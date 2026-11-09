"""Standalone RAVDESS training: video/audio features + fixed first-layer gradient gates.

Random initialization by default; --checkpoint optionally initializes encoders.
No dependency on another experiment runner. --prepare-only needs only stdlib.
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

EMOTIONS = ['happy', 'sad', 'angry', 'fearful', 'surprised', 'disgusted']
EMOTION_CODES = {3: 0, 4: 1, 5: 2, 6: 3, 8: 4, 7: 5}
NUM_CLASSES = len(EMOTIONS)
TEST_CONDITIONS = ('both', 'video_noise', 'audio_neutral')
LOGIT_TERMS = ('mlp',)
ARCHITECTURE = 'indicator_mlp'
TEST_INDICATORS = {'both': [1, 1], 'video_noise': [1, 0], 'audio_neutral': [1, 1]}


def resolve_gpu_selection(device, gpu_ids=None, available_count=None):
    """Select visible CUDA IDs without importing torch (also used in preparation)."""
    requested = list(gpu_ids) if gpu_ids else None
    if not device.startswith('cuda'):
        if requested:
            raise ValueError('--gpu-ids requires a CUDA --device')
        return device, []
    if device != 'cuda':
        parts = device.split(':')
        if len(parts) != 2 or parts[0] != 'cuda' or not parts[1].isdigit():
            raise ValueError('Use --device cuda or cuda:N')
    selected = requested if requested is not None else [0 if device == 'cuda' else int(device.split(':')[1])]
    if any(type(index) is not int or index < 0 for index in selected) or len(set(selected)) != len(selected):
        raise ValueError('--gpu-ids must contain unique nonnegative GPU indices')
    if available_count is not None and any(index >= available_count for index in selected):
        raise ValueError('Requested GPUs %s, but PyTorch sees %d CUDA devices' % (selected, available_count))
    return 'cuda:%d' % selected[0], selected


def configure_parallel_model(raw_model, args):
    """Single-process DataParallel; raw_model stays suitable for checkpoint/heatmap use."""
    import torch
    count = torch.cuda.device_count() if args.device.startswith('cuda') else 0
    primary, selected = resolve_gpu_selection(args.device, getattr(args, 'gpu_ids', None), count)
    args.device, args.gpu_ids = primary, selected
    raw_model.to(torch.device(primary))
    model = (torch.nn.DataParallel(raw_model, device_ids=selected, output_device=selected[0])
             if len(selected) > 1 else raw_model)
    info = dict(mode='data_parallel' if len(selected) > 1 else 'single_device', primary_device=primary,
                gpu_ids=selected, gpu_names=[torch.cuda.get_device_name(index) for index in selected],
                global_batch_size=args.batch_size,
                maximum_samples_per_gpu=math.ceil(args.batch_size / max(1, len(selected))))
    print('Device configuration:', json.dumps(info), flush=True)
    if len(selected) > 1 and args.batch_size <= len(selected):
        print('Small global batch: consider --batch-size 16 for four GPUs if memory permits; '
              'parallel overhead can dominate small batches.', flush=True)
    return model, info


def indicators_for_conditions(conditions, training, test_policy=None):
    available = {'both': (1, 1), 'video_noise': (1, 0), 'audio_neutral': (0, 1)}
    if any(condition not in available for condition in conditions):
        raise ValueError('Unknown input condition')
    policy = TEST_INDICATORS if test_policy is None else test_policy
    return [list(available[condition] if training else policy[condition] if isinstance(policy, dict) else policy)
            for condition in conditions]


def first_layer_half_width(width):
    if width < 2 or width % 2:
        raise ValueError('The first MLP hidden width must be even and at least 2 for fixed half-neuron gating')
    return width // 2


def first_layer_gate_sizes(width, visual=None, audio=None):
    """Both supplied: their sum sets width; one supplied: infer the remainder."""
    if visual is None and audio is None:
        visual = audio = first_layer_half_width(width)
    elif visual is None:
        visual = width - audio
    elif audio is None:
        audio = width - visual
    if any(type(value) is not int or value < 1 for value in (visual, audio)):
        raise ValueError('Visual/audio gated neuron counts must both be positive integers')
    return visual, audio


def condition_gate_sizes(av=128, visual=64, audio=64):
    if any(type(value) is not int or value < 1 for value in (av, visual, audio)):
        raise ValueError('AV, visual-only and audio-only gated neuron counts must be positive integers')
    return {'av': av, 'visual': visual, 'audio': audio}


def condition_group_gates(indicators):
    mapping = {(1, 1): [1, 0, 0], (1, 0): [0, 1, 0], (0, 1): [0, 0, 1]}
    try:
        return [mapping[tuple(bits)] for bits in indicators]
    except KeyError as error:
        raise ValueError('Training indicators must be [1,1], [1,0] or [0,1]') from error


def discover(video_root, audio_root):
    """Pair speech MP4s with WAVs using the six non-modality filename fields."""
    audios = {}
    audio_priority = {'03': 0, '01': 1, '02': 2}
    for path in sorted(Path(audio_root).rglob('*')):
        if not path.is_file() or path.suffix.lower() != '.wav':
            continue
        parts = path.stem.split('-')
        if len(parts) == 7 and parts[0] in audio_priority and parts[1] == '01':
            key = '-'.join(parts[1:])
            if key in audios and audios[key].stem.split('-')[0] == parts[0]:
                raise ValueError(f'Duplicate audio trial: {key}')
            # Official audio-only exports take priority when several versions exist.
            if key not in audios or audio_priority[parts[0]] < audio_priority[audios[key].stem.split('-')[0]]:
                audios[key] = path.resolve()
    videos = {}
    for path in sorted(Path(video_root).rglob('*')):
        if not path.is_file() or path.suffix.lower() != '.mp4':
            continue
        parts = path.stem.split('-')
        if len(parts) != 7 or parts[0] not in ('01', '02') or parts[1] != '01':
            continue
        key = '-'.join(parts[1:])
        # Prefer video-only when both modalities are downloaded.
        if key in videos and videos[key].stem.split('-')[0] == parts[0]:
            raise ValueError(f'Duplicate video trial: {key}')
        if key not in videos or parts[0] == '02':
            videos[key] = path.resolve()
    missing = sorted(set(videos) - set(audios))
    if missing:
        raise ValueError(f'{len(missing)} videos have no matching WAV; first: {missing[0]}')
    rows = []
    for key, video in sorted(videos.items()):
        parts = video.stem.split('-')
        code, actor = int(parts[2]), int(parts[6])
        if not 1 <= code <= 8 or not 1 <= actor <= 24:
            raise ValueError(f'Invalid RAVDESS filename: {video}')
        # Neutral is a stimulus control only; calm is excluded entirely later.
        label = EMOTION_CODES.get(code, -1 if code == 1 else -2)
        rows.append(dict(video=str(video), audio=str(audios[key]), label=label, actor=actor))
    if not rows:
        raise ValueError('No paired RAVDESS speech MP4/WAV files found.')
    return rows


def balanced_split(rows, seed):
    """Hold out four complete actors; balance emotions within each partition."""
    emotional = [row for row in rows if row['label'] in range(NUM_CLASSES)]
    if len({row['video'] for row in emotional}) != len(emotional):
        raise ValueError('Duplicate emotional trial')
    actors = sorted({row['actor'] for row in emotional})
    if len(actors) != 24:
        raise ValueError('The 20:4 actor split requires 24 actors with emotional trials; found %d' % len(actors))
    rng = random.Random(seed)
    rng.shuffle(actors)
    actor_groups = {'train': set(actors[4:]), 'test': set(actors[:4])}
    result, report = {}, {}
    for name, selected in actor_groups.items():
        buckets = [sorted([row for row in emotional if row['actor'] in selected and row['label'] == label],
                          key=lambda row: row['video']) for label in range(NUM_CLASSES)]
        counts = [len(bucket) for bucket in buckets]
        count = min(counts)
        if count == 0:
            raise ValueError('Missing emotion in %s actor partition: %s' % (name, counts))
        result[name] = []
        for bucket in buckets:
            rng.shuffle(bucket)
            result[name].extend(dict(row) for row in bucket[:count])
        rng.shuffle(result[name])
        report[name] = dict(actors=sorted(selected), original_counts=counts,
                            balanced_counts=[count] * NUM_CLASSES, total=count * NUM_CLASSES,
                            unused_counts=[value-count for value in counts])
    if actor_groups['train'] & actor_groups['test']:
        raise RuntimeError('Actor overlap between training and testing')
    if {row['video'] for row in result['train']} & {row['video'] for row in result['test']}:
        raise RuntimeError('Trial overlap between training and testing')
    original_counts = [sum(row['label'] == label for row in emotional) for label in range(NUM_CLASSES)]
    report['design'] = dict(method='actor_disjoint_20_4', seed=seed,
                            train_actor_count=20, test_actor_count=4, original_counts=original_counts,
                            unused_counts=[report['train']['unused_counts'][i] + report['test']['unused_counts'][i]
                                           for i in range(NUM_CLASSES)], shared_actors=[],
                            validation=False, checkpoint_selection='final_epoch')
    return result, report


def attach_test_controls(splits, rows, seed):
    """Attach neutral references and disjoint noise seeds to train/test base trials."""
    references = {}
    for row in rows:
        if row['label'] == -1:
            parts = Path(row['video']).stem.split('-')
            key = (row['actor'], parts[4], parts[5])
            if key in references:
                raise ValueError(f'Duplicate neutral reference: {key}')
            references[key] = row['video']
    for name, offset in [('train', 0), ('test', 100000)]:
        for i, row in enumerate(splits.get(name, [])):
            parts = Path(row['video']).stem.split('-')
            key = (row['actor'], parts[4], parts[5])
            if key not in references:
                raise ValueError(f'Missing neutral speech video for {name} actor/statement/repetition {key}')
            row['neutral_video'] = references[key]
            row['noise_seed'] = seed + offset + i


def expand_training_rows(rows):
    """Each training trial appears once under each condition; loader shuffles all rows."""
    return [dict(row, condition=condition) for row in rows for condition in TEST_CONDITIONS]


def white_noise(num_samples, seed, rms):
    """Fixed-RMS Gaussian white noise generated in waveform space, not filterbank space."""
    import torch
    generator = torch.Generator().manual_seed(seed)
    wave = torch.randn(1, num_samples, generator=generator)
    wave = wave - wave.mean()
    return wave * (rms / wave.square().mean().sqrt().clamp_min(1e-12))


def gaussian_noise_image(seed, mean=.5, std=.2, size=160):
    """One 2D Gaussian noise realization, replicated across RGB channels."""
    import numpy as np
    noise = np.random.default_rng(seed).normal(mean, std, size=(size, size))
    gray = np.rint(np.clip(noise, 0., 1.) * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def make_loaders(splits, args):
    import numpy as np
    import torch
    import torchaudio
    from decord import VideoReader, cpu
    from torch.utils.data import Dataset, DataLoader

    class PairedDataset(Dataset):
        def __init__(self, rows, training, condition='both'):
            self.rows, self.training, self.condition = rows, training, condition
            self.epoch = 0

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            condition = row.get('condition', self.condition)
            vr = VideoReader(row['video'], ctx=cpu(0), num_threads=1)
            waveform, sr = torchaudio.load(row['audio'])
            waveform = waveform.mean(0, keepdim=True)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            fps = vr.get_avg_fps()
            duration = min(len(vr) / fps, waveform.shape[1] / 16000)
            if duration <= 0:
                raise ValueError(f'Empty media: {row}')
            # A synchronized 2.56 s window; repeat video edge / pad audio for short clips.
            span = min(2.56, duration)
            start = random.uniform(0, duration - span) if self.training else (duration - span) / 2
            times = start + np.arange(16) * span / 16
            indices = np.clip((times * fps).astype(int), 0, len(vr) - 1)
            if (condition == 'audio_neutral' and self.training
                    and getattr(args, 'training_audio_only_visual', 'neutral') == 'gaussian_noise'):
                visual_seed = row['noise_seed'] + self.epoch * 1000000 + 2000000000
                image = gaussian_noise_image(visual_seed, getattr(args, 'visual_noise_mean', .5),
                                             getattr(args, 'visual_noise_std', .2))
                pixels = np.repeat(image[None], 16, axis=0)
            elif (condition == 'audio_neutral' and self.training
                    and getattr(args, 'training_audio_only_visual', 'neutral') == 'white'):
                # Pure RGB white before the same normal preprocessing as all video.
                pixels = np.full((16, 160, 160, 3), 255, dtype=np.uint8)
            elif condition == 'audio_neutral':
                neutral = VideoReader(row['neutral_video'], ctx=cpu(0), num_threads=1)
                # New runs use frame 0 as a static image, repeated across time.
                # Old configs omit this policy and retain their midpoint control
                # when regenerating explanations from an existing checkpoint.
                frame_policy = getattr(args, 'neutral_frame_policy', 'midpoint')
                neutral_index = 0 if frame_policy == 'first' else len(neutral) // 2
                pixels = np.repeat(neutral.get_batch([neutral_index]).asnumpy(), 16, axis=0)
            else:
                pixels = vr.get_batch(indices).asnumpy()
            frames = torch.from_numpy(pixels).permute(0, 3, 1, 2).float() / 255
            _, _, h, w = frames.shape
            side = min(h, w)
            frames = frames[:, :, (h-side)//2:(h+side)//2, (w-side)//2:(w+side)//2]
            frames = torch.nn.functional.interpolate(frames, (160, 160), mode='bilinear', align_corners=False)
            mean = frames.new_tensor([.485, .456, .406])[None, :, None, None]
            std = frames.new_tensor([.229, .224, .225])[None, :, None, None]
            video = ((frames - mean) / std).permute(1, 0, 2, 3)
            wave = waveform[:, int(start*16000):int((start+span)*16000)]
            wave = torch.nn.functional.pad(wave, (0, max(0, 40960-wave.shape[1])))[:, :40960]
            wave = wave - wave.mean()
            if condition == 'video_noise':
                noise_seed = row['noise_seed'] + (self.epoch * 1000000 if self.training else 0)
                wave = white_noise(40960, noise_seed, args.noise_rms)
            fb = torchaudio.compliance.kaldi.fbank(
                wave, htk_compat=True, sample_frequency=16000, use_energy=False,
                window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
            fb = torch.nn.functional.pad(fb[:256], (0, 0, 0, max(0, 256-fb.shape[0])))
            audio = ((fb + 4.2677393) / (4.5689974 * 2)).unsqueeze(0)
            return video, audio, row['label'], index

    # num_workers=0 default also supports Windows: nested Dataset is not spawn-picklable.
    def loader(rows, training=False, condition='both'):
        return DataLoader(PairedDataset(rows, training, condition), batch_size=args.batch_size,
                          shuffle=training, num_workers=0,
                          pin_memory=args.device.startswith('cuda'), drop_last=False)
    loaders = dict(train=loader(expand_training_rows(splits['train']), training=True))
    for condition in TEST_CONDITIONS:
        loaders['test_' + condition] = loader(splits['test'], condition=condition)
    return loaders

def make_model(args):
    import torch
    from torch import nn
    from functools import partial
    from modeling_finetune_av import AudioVisionTransformer

    class IndicatorMLP(nn.Module):
        architecture = ARCHITECTURE
        logit_terms = LOGIT_TERMS

        def __init__(self):
            super().__init__()
            mode = getattr(args, 'mlp_input_mode', 'concat_indicators')
            self.condition_gradient_gating = mode == 'condition_gradient_gated'
            self.gradient_gating = mode in ('gradient_gated', 'condition_gradient_gated')
            if not self.gradient_gating:
                # Retain the old architecture only for loading previous runs.
                self.test_indicators = getattr(args, 'test_indicators', [1, 1])
            backbone = AudioVisionTransformer(
                img_size=160, patch_size=16, embed_dim=args.embed_dim,
                num_heads=args.embed_dim // 64, depth=args.depth,
                img_size_audio=(256, 128), patch_size_audio=16,
                embed_dim_audio=args.embed_dim, num_heads_audio=args.embed_dim // 64,
                depth_audio=args.depth, fusion_depth=0, num_classes=NUM_CLASSES,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                all_frames=16, drop_path_rate=args.drop_path)
            self.encoder, self.encoder_audio = backbone.encoder, backbone.encoder_audio
            self.video_layer_weights = backbone.video_layer_weights
            self.audio_layer_weights = backbone.audio_layer_weights
            def feature_head():
                return nn.Sequential(nn.LayerNorm(args.embed_dim), nn.Linear(args.embed_dim, args.head_dim), nn.GELU())
            self.video_head, self.audio_head = feature_head(), feature_head()
            hidden_dims = list(args.mlp_hidden_dims)
            if self.gradient_gating:
                if self.condition_gradient_gating:
                    self.gate_group_sizes = condition_gate_sizes(getattr(args, 'av_gated_neurons', 128),
                        getattr(args, 'visual_gated_neurons', 64), getattr(args, 'audio_gated_neurons', 64))
                else:
                    visual, audio = first_layer_gate_sizes(hidden_dims[0], getattr(args, 'visual_gated_neurons', None),
                                                          getattr(args, 'audio_gated_neurons', None))
                    self.gate_group_sizes = {'visual': visual, 'audio': audio}
                self.visual_gated_neurons, self.audio_gated_neurons = self.gate_group_sizes['visual'], self.gate_group_sizes['audio']
                hidden_dims[0] = sum(self.gate_group_sizes.values())
            layers, width = [], 2 * args.head_dim + (0 if self.gradient_gating else 2)
            for index, hidden in enumerate(hidden_dims):
                if self.gradient_gating and index == 0:
                    class FixedGroups(nn.Module):
                        def __init__(self, inputs, group_sizes):
                            super().__init__()
                            # Separate parameters allow AdamW to skip a wholly
                            # inactive group, including momentum and weight decay.
                            self.order = tuple(group_sizes)
                            for name, count in group_sizes.items():
                                setattr(self, name, nn.Linear(inputs, count))
                        def forward(self, features):
                            return torch.cat([getattr(self, name)(features) for name in self.order], dim=-1)
                    linear = FixedGroups(width, self.gate_group_sizes)
                else:
                    linear = nn.Linear(width, hidden)
                layers.extend([linear, nn.GELU(), nn.Dropout(args.mlp_dropout)])
                width = hidden
            layers.append(nn.Linear(width, NUM_CLASSES))
            self.classifier = nn.Sequential(*layers)

        def indicators_for_test_condition(self, condition):
            if self.gradient_gating:
                return None
            return indicators_for_conditions([condition], False, self.test_indicators)[0]

        def training_group_gates(self, indicators):
            if not self.condition_gradient_gating:
                return indicators
            if not (indicators.sum(dim=1) >= 1).all():
                raise ValueError('No training condition corresponds to indicators [0,0]')
            visual, audio = indicators[:, :1], indicators[:, 1:2]
            return torch.cat((visual * audio, visual * (1-audio), (1-visual) * audio), dim=1)

        def prepare_gradient_gate_step(self, indicators):
            """Use GLOBAL batch eligibility, after DataParallel gradient reduction."""
            if self.gradient_gating:
                active = self.training_group_gates(indicators.detach()).bool().any(dim=0).tolist()
                for enabled, name in zip(active, self.gate_group_sizes):
                    if not enabled:
                        for parameter in getattr(self.classifier[0], name).parameters():
                            parameter.grad = None

        def forward(self, video, audio, indicators=None, return_terms=False):
            v = self.encoder(video).mean(1)
            a = self.encoder_audio(audio).mean(1)
            ov = (v * self.video_layer_weights.softmax(0)[None, :, None]).sum(1)
            oa = (a * self.audio_layer_weights.softmax(0)[None, :, None]).sum(1)
            hv, ha = self.video_head(ov), self.audio_head(oa)
            if self.training or not self.gradient_gating:
                if indicators is None:
                    if self.training:
                        raise ValueError('Training requires explicit on_v/on_a indicators')
                    indicators = hv.new_ones((len(hv), 2))
                indicators = torch.as_tensor(indicators, dtype=hv.dtype, device=hv.device)
                if indicators.shape != (len(hv), 2) or not ((indicators == 0) | (indicators == 1)).all():
                    raise ValueError('Indicators must be a B-by-2 tensor of zeros/ones ordered [on_v, on_a]')
            if self.gradient_gating:
                hidden = self.classifier[1](self.classifier[0](torch.cat((hv, ha), dim=-1)))
                if self.training:
                    groups = self.training_group_gates(indicators)
                    gate = torch.cat([groups[:, i:i+1].expand(-1, count)
                                      for i, count in enumerate(self.gate_group_sizes.values())], dim=-1)
                    # Identity in the forward pass; only incoming gradients are gated.
                    hidden = hidden.detach() + gate * (hidden - hidden.detach())
                for layer in list(self.classifier.children())[2:]:
                    hidden = layer(hidden)
                logits = hidden
            else:
                logits = self.classifier(torch.cat((hv, ha, indicators), dim=-1))
            return (logits, (logits,)) if return_terms else logits

    return IndicatorMLP()


def load_pretrained(model, path):
    """Optional encoder initialization; all MLP/feature heads stay newly initialized."""
    import torch
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('model', checkpoint.get('module', checkpoint))
    state = {(key[7:] if key.startswith('module.') else key): value for key, value in state.items()}
    target = model.state_dict()
    prefixes = ('encoder.', 'encoder_audio.')
    required = [key for key in target if key.startswith(prefixes)]
    missing = [key for key in required if key not in state]
    mismatch = [key for key in required if key in state and state[key].shape != target[key].shape]
    if missing or mismatch:
        raise ValueError('Encoder checkpoint incompatible; missing: %s; shape mismatches: %s. Match depth/embed-dim.'
                         % (missing[:8], mismatch[:8]))
    kept = {key: value for key, value in state.items()
            if (key.startswith(prefixes) or key in ('video_layer_weights', 'audio_layer_weights'))
            and key in target and value.shape == target[key].shape}
    model.load_state_dict(kept, strict=False)
    print('Initialized encoders from checkpoint (%d tensors); new feature heads and MLP.' % len(kept))


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def classification_metrics(records):
    import numpy as np
    from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
    labels = [row['label'] for row in records]
    predictions = [row['prediction'] for row in records]
    return dict(loss=float(np.mean([row['loss'] for row in records])),
                accuracy=float(accuracy_score(labels, predictions)), correct=sum(row['correct'] for row in records), total=len(records),
                uar=float(recall_score(labels, predictions, labels=list(range(NUM_CLASSES)), average='macro', zero_division=0)),
                macro_f1=float(f1_score(labels, predictions, labels=list(range(NUM_CLASSES)), average='macro', zero_division=0)),
                confusion_matrix=confusion_matrix(labels, predictions, labels=list(range(NUM_CLASSES))).tolist())


def train(splits, args, output):
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    raw_model = make_model(args)
    if args.checkpoint:
        load_pretrained(raw_model, args.checkpoint)
    else:
        print('Training both encoders, feature heads and MLP from random initialization.')
    model, parallelism = configure_parallel_model(raw_model, args)
    gradient_gating = getattr(raw_model, 'gradient_gating', False)
    device = torch.device(args.device)
    (output / 'parallelism.json').write_text(json.dumps(parallelism, indent=2))
    loaders = make_loaders(splits, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    predictions_dir = output / 'epoch_predictions'
    predictions_dir.mkdir()

    def run_pass(loader, training=False):
        model.train(training)
        records = []
        with torch.set_grad_enabled(training):
            for video, audio, label, indices in loader:
                index_list = indices.tolist()
                rows = [loader.dataset.rows[index] for index in index_list]
                conditions = [row.get('condition', loader.dataset.condition) for row in rows]
                bits = (indicators_for_conditions(conditions, training, getattr(args, 'test_indicators', TEST_INDICATORS))
                        if training or not gradient_gating else None)
                indicators = torch.tensor(bits, device=device, dtype=torch.float32) if bits is not None else None
                forward_args = dict(indicators=indicators) if indicators is not None else {}
                logits, terms = model(video.to(device), audio.to(device), return_terms=True, **forward_args)
                torch.testing.assert_close(logits, sum(terms))
                losses = torch.nn.functional.cross_entropy(logits, label.to(device), reduction='none')
                loss = losses.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite emotion loss')
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if gradient_gating:
                        raw_model.prepare_gradient_gate_step(indicators)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                predictions = logits.detach().argmax(1).cpu().tolist()
                probabilities = logits.detach().softmax(1).cpu().tolist()
                scores = logits.detach().cpu().tolist()
                loss_values = losses.detach().cpu().tolist()
                for i, row in enumerate(rows):
                    available = indicators_for_conditions([conditions[i]], True)[0]
                    noise_seed = row.get('noise_seed', '')
                    if training and conditions[i] == 'video_noise' and noise_seed != '':
                        noise_seed += getattr(loader.dataset, 'epoch', 0) * 1000000
                    target = int(label[i].item())
                    visual_seed = (row['noise_seed'] + getattr(loader.dataset, 'epoch', 0) * 1000000 + 2000000000
                                   if training and conditions[i] == 'audio_neutral'
                                   and getattr(args, 'training_audio_only_visual', '') == 'gaussian_noise' else '')
                    record = dict(video=row['video'], actor=row['actor'], condition=conditions[i],
                                  neutral_video=row.get('neutral_video', ''), noise_seed=noise_seed, visual_noise_seed=visual_seed,
                                  available_v=available[0], available_a=available[1],
                                  on_v=bits[i][0] if bits is not None else '', on_a=bits[i][1] if bits is not None else '',
                                  label=target, prediction=predictions[i], correct=int(target == predictions[i]), loss=loss_values[i])
                    if getattr(raw_model, 'condition_gradient_gating', False):
                        record['gradient_gate_group'] = ({'both': 'av', 'video_noise': 'visual', 'audio_neutral': 'audio'}[conditions[i]]
                                                         if training else '')
                    for j, emotion in enumerate(EMOTIONS):
                        record['p_' + emotion] = probabilities[i][j]
                        record['logit_' + emotion] = record['mlp_logit_' + emotion] = scores[i][j]
                    records.append(record)
        return classification_metrics(records), records

    def save_epoch(epoch, split, metrics, records, learning_rate):
        path = output / 'epoch_metrics.csv'
        exists = path.exists()
        with path.open('a', newline='') as f:
            fields = ['epoch', 'split', 'loss', 'accuracy', 'correct', 'total', 'uar', 'macro_f1', 'learning_rate']
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(dict(epoch=epoch, split=split, learning_rate=learning_rate,
                                 **{key: metrics[key] for key in fields if key in metrics}))
        write_csv(predictions_dir / ('%s_epoch_%03d.csv' % (split, epoch)), records)

    for epoch in range(args.epochs):
        learning_rate = optimizer.param_groups[0]['lr']
        loaders['train'].dataset.epoch = epoch
        train_metrics, train_records = run_pass(loaders['train'], training=True)
        save_epoch(epoch+1, 'train', train_metrics, train_records, learning_rate)
        by_condition = {}
        for condition in TEST_CONDITIONS:
            subset = [row for row in train_records if row['condition'] == condition]
            by_condition[condition] = classification_metrics(subset)
            save_epoch(epoch+1, 'train_' + condition, by_condition[condition], subset, learning_rate)
        tested = {}
        if not args.skip_epoch_tests:
            for condition in TEST_CONDITIONS:
                tested[condition], rows = run_pass(loaders['test_' + condition])
                save_epoch(epoch+1, 'test_' + condition, tested[condition], rows, learning_rate)
        record = dict(epoch=epoch+1, train=train_metrics, train_conditions=by_condition, tests=tested, learning_rate=learning_rate)
        print(json.dumps(record), flush=True)
        with (output / 'history.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        scheduler.step()

    # Save the underlying model, never DataParallel's module.-prefixed state dict.
    torch.save(dict(model=raw_model.state_dict(), args=vars(args), epoch=args.epochs, emotions=EMOTIONS,
                    checkpoint_selection='final_epoch'), output / 'final.pth')
    matrices = {key.replace('.', '_'): value.detach().cpu().numpy() for key, value in raw_model.state_dict().items()
                if key.startswith(('classifier.', 'video_head.', 'audio_head.'))}
    np.savez_compressed(output / 'classifier_weights.npz', emotions=np.asarray(EMOTIONS), logit_terms=np.asarray(LOGIT_TERMS), **matrices)
    final = {}
    for condition in TEST_CONDITIONS:
        final[condition], records = run_pass(loaders['test_' + condition])
        write_csv(output / ('test_predictions_' + condition + '.csv'), records)
        (output / ('test_metrics_' + condition + '.json')).write_text(json.dumps(final[condition], indent=2))
    summary = dict(emotions=EMOTIONS, architecture=ARCHITECTURE, logit_terms=list(LOGIT_TERMS),
                   initialization='pretrained_encoders' if args.checkpoint else 'random',
                   checkpoint_epoch=args.epochs, checkpoint_file='final.pth', checkpoint_selection='final_epoch',
                   parallelism=parallelism,
                   mlp_input_mode=getattr(args, 'mlp_input_mode', 'concat_indicators'),
                   test_indicators=None if gradient_gating else getattr(args, 'test_indicators', TEST_INDICATORS), conditions=final)
    (output / 'test_metrics.json').write_text(json.dumps(summary, indent=2))
    if not args.skip_explanations:
        from video_loss_explanations import export_test_explanations
        # Native attention hooks and one-trial explanations run on the primary GPU.
        export_test_explanations(raw_model, loaders, splits['test'], TEST_CONDITIONS, device, output, args.explain_limit)
    print('Final-epoch test results:', json.dumps(final), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-root', required=True)
    parser.add_argument('--audio-root', required=True)
    parser.add_argument('--output-dir', default='runs/ravdess_indicator_mlp')
    parser.add_argument('--checkpoint', help='Optional encoder initialization; omit to train from scratch')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embed-dim', type=int, choices=[384, 512], default=512)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--head-dim', type=int, default=256)
    parser.add_argument('--mlp-hidden-dims', nargs='+', type=int, default=[256, 128])
    parser.add_argument('--visual-gated-neurons', type=int, default=None,
                        help='Visual gradient group size; default half the first hidden width')
    parser.add_argument('--audio-gated-neurons', type=int, default=None,
                        help='Audio gradient group size; default half the first hidden width')
    parser.add_argument('--mlp-dropout', type=float, default=.1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=.05)
    parser.add_argument('--drop-path', type=float, default=.1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu-ids', nargs='+', type=int,
                        help='CUDA GPU indices, e.g. 2 or 0 1 2 3; first is primary. Omit for one GPU.')
    parser.add_argument('--noise-rms', type=float, default=.05)
    parser.add_argument('--skip-epoch-tests', action='store_true')
    parser.add_argument('--skip-explanations', action='store_true')
    parser.add_argument('--explain-limit', type=int, default=0)
    args = parser.parse_args()
    try:
        args.device, args.gpu_ids = resolve_gpu_selection(args.device, args.gpu_ids)
    except ValueError as error:
        parser.error(str(error))
    if min(args.epochs, args.batch_size, args.depth, args.head_dim, *args.mlp_hidden_dims) < 1:
        parser.error('Epochs, batch size, depth and MLP dimensions must be positive')
    try:
        args.visual_gated_neurons, args.audio_gated_neurons = first_layer_gate_sizes(
            args.mlp_hidden_dims[0], args.visual_gated_neurons, args.audio_gated_neurons)
        args.mlp_hidden_dims[0] = args.visual_gated_neurons + args.audio_gated_neurons
    except ValueError as error:
        parser.error(str(error))
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('Learning rate must be finite/positive; weight decay finite/nonnegative')
    if any(not math.isfinite(value) or not 0 <= value < 1 for value in (args.mlp_dropout, args.drop_path)):
        parser.error('Dropout/drop-path must be finite and in [0,1)')
    if not math.isfinite(args.noise_rms) or not 0 < args.noise_rms <= 1 or args.explain_limit < 0:
        parser.error('Noise RMS must be in (0,1] and explanation limit nonnegative')
    args.architecture, args.logit_terms = ARCHITECTURE, list(LOGIT_TERMS)
    args.neutral_frame_policy = 'first'
    args.training_audio_only_visual = 'gaussian_noise'
    args.visual_noise_mean, args.visual_noise_std = .5, .2
    args.testing_audio_only_visual = 'neutral'
    args.training_conditions = list(TEST_CONDITIONS)
    args.training_indicators = {'both': [1, 1], 'video_noise': [1, 0], 'audio_neutral': [0, 1]}
    args.mlp_input_mode = 'gradient_gated'
    args.test_indicators = None
    args.first_layer_gate_assignment = dict(visual=[0, args.visual_gated_neurons],
        audio=[args.visual_gated_neurons, args.mlp_hidden_dims[0]],
        ranges='start_inclusive_end_exclusive', fixed=True)
    args.training_gate_groups = {'both': ['visual', 'audio'], 'video_noise': ['visual'], 'audio_neutral': ['audio']}
    args.initialization = 'pretrained_encoders' if args.checkpoint else 'random'
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error('Use a new/empty output directory')
    rows = discover(args.video_root, args.audio_root)
    splits, report = balanced_split(rows, args.seed)
    attach_test_controls(splits, rows, args.seed)
    args.split_method = 'actor_disjoint_20_4'
    args.train_actors = report['train']['actors']
    args.test_actors = report['test']['actors']
    expanded = expand_training_rows(splits['train'])
    for row in expanded:
        row['on_v'], row['on_a'] = indicators_for_conditions([row['condition']], True)[0]
    report['train_expanded'] = dict(total=len(expanded), conditions=list(TEST_CONDITIONS),
                                   counts_per_condition={condition: len(splits['train']) for condition in TEST_CONDITIONS},
                                   counts_per_emotion=[3*count for count in report['train']['balanced_counts']])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
    (output / 'split_report.json').write_text(json.dumps(dict(emotions=EMOTIONS, splits=report), indent=2))
    for name, subset in splits.items():
        write_csv(output / (name+'.csv'), subset)
    write_csv(output / 'train_expanded.csv', expanded)
    print(json.dumps(report, indent=2), flush=True)
    if not args.prepare_only:
        train(splits, args, output)


if __name__ == '__main__':
    main()
