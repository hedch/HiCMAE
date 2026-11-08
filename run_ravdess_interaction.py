"""Fine-tune HiCMAE encoders with additive + Hadamard interaction logits.

Run --help for options; see RAVDESS_INTERACTION.md for data preparation.
Only --prepare-only can run without the upstream ML dependencies.
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

EMOTIONS = ['happy', 'sad', 'angry', 'fearful', 'surprised', 'disgusted']
# RAVDESS codes: 07=disgust, 08=surprised. Preserve the requested class order.
EMOTION_CODES = {3: 0, 4: 1, 5: 2, 6: 3, 8: 4, 7: 5}
NUM_CLASSES = len(EMOTIONS)
TEST_CONDITIONS = ('both', 'video_noise', 'audio_neutral')
LOGIT_TERMS = ('video', 'interaction')
INTERACTION_TERMS = ('video', 'audio', 'interaction')


def normalize_terms(names):
    names = ['video' if name == 'visual' else name for name in names]
    if not names or any(name not in INTERACTION_TERMS for name in names):
        raise ValueError('Choose at least one term from visual, audio, interaction')
    if len(set(names)) != len(names):
        raise ValueError('Each term may be selected only once (visual and video are aliases)')
    return tuple(name for name in INTERACTION_TERMS if name in names)


def auxiliary_weights(args):
    weights = {name: float(getattr(args, 'aux_' + name + '_weight', 0.)) for name in ('video', 'audio')}
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError('Auxiliary weights must be finite and nonnegative')
    return weights


def training_objective(logits, target, auxiliary, av_mask, weights):
    """Main CE on every presentation; auxiliary CE only on real AV examples.

    Auxiliary logits contain only the rows selected by av_mask. Empty-AV batches
    contribute main CE only. Auxiliary means are conditional on AV rows.
    """
    import torch.nn.functional as F
    main = F.cross_entropy(logits, target, reduction='none')
    loss, auxiliary_losses = main.mean(), {}
    if av_mask.any():
        for name, scores in auxiliary.items():
            values = F.cross_entropy(scores, target[av_mask], reduction='none')
            loss = loss + weights[name] * values.mean()
            auxiliary_losses[name] = values
    return loss, main, auxiliary_losses


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
    """Mix actors; independently shuffle each emotion into 128 train / 64 test trials."""
    rng = random.Random(seed)
    result, report = {'train': [], 'test': []}, {}
    original_counts = []
    for label, emotion in enumerate(EMOTIONS):
        bucket = sorted([row for row in rows if row['label'] == label], key=lambda r: r['video'])
        if len({row['video'] for row in bucket}) != len(bucket):
            raise ValueError('Duplicate emotional trial in ' + emotion)
        original_counts.append(len(bucket))
        if len(bucket) < 192:
            raise ValueError('%s has %d trials; exactly 128 training + 64 testing requires at least 192.'
                             % (emotion, len(bucket)))
        rng.shuffle(bucket)
        result['train'].extend(dict(row) for row in bucket[:128])
        result['test'].extend(dict(row) for row in bucket[128:192])
    train_ids = {row['video'] for row in result['train']}
    test_ids = {row['video'] for row in result['test']}
    if train_ids & test_ids:
        raise ValueError('A trial occurs in both training and testing')
    for name, n in [('train', 128), ('test', 64)]:
        rng.shuffle(result[name])
        report[name] = dict(actors=sorted({row['actor'] for row in result[name]}),
                            balanced_counts=[n] * NUM_CLASSES, total=n * NUM_CLASSES)
    report['design'] = dict(method='mixed_actor_trial_split', seed=seed,
                            original_counts=original_counts,
                            unused_counts=[n-192 for n in original_counts],
                            shared_actors=sorted({r['actor'] for r in result['train']} &
                                                 {r['actor'] for r in result['test']}),
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


def make_model(args):
    import torch
    from torch import nn
    from functools import partial
    from modeling_finetune_av import AudioVisionTransformer

    class InteractionModel(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = AudioVisionTransformer(
                img_size=160, patch_size=16, embed_dim=args.embed_dim,
                num_heads=args.embed_dim // 64, depth=args.depth,
                img_size_audio=(256, 128), patch_size_audio=16,
                embed_dim_audio=args.embed_dim, num_heads_audio=args.embed_dim // 64,
                depth_audio=args.depth, fusion_depth=0, num_classes=NUM_CLASSES,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                all_frames=16, drop_path_rate=args.drop_path)
            # Register only encoders and layer pooling weights: no fusion module.
            self.encoder = backbone.encoder
            self.encoder_audio = backbone.encoder_audio
            self.video_layer_weights = backbone.video_layer_weights
            self.audio_layer_weights = backbone.audio_layer_weights
            self.logit_terms = normalize_terms(getattr(args, 'logit_terms', LOGIT_TERMS))
            if 'video' in self.logit_terms:
                self.wv = nn.Linear(args.embed_dim, NUM_CLASSES, bias=False)
            if 'audio' in self.logit_terms:
                self.wa = nn.Linear(args.embed_dim, NUM_CLASSES, bias=False)
            if 'interaction' in self.logit_terms:
                self.wf = nn.Linear(args.embed_dim, NUM_CLASSES, bias=False)
            # Separate training-only classifiers: never summed into main logits.
            self.aux_heads = nn.ModuleDict({name: nn.Linear(args.embed_dim, NUM_CLASSES, bias=False)
                                           for name, weight in auxiliary_weights(args).items() if weight > 0})

        def forward(self, video, audio, return_terms=False, return_aux=False, aux_mask=None):
            values = {}
            if 'video' in self.logit_terms or 'interaction' in self.logit_terms or (return_aux and 'video' in self.aux_heads):
                v = self.encoder(video).mean(dim=1)  # B,L,D
                ov = (v * self.video_layer_weights.softmax(0)[None, :, None]).sum(1)
            if 'audio' in self.logit_terms or 'interaction' in self.logit_terms or (return_aux and 'audio' in self.aux_heads):
                a = self.encoder_audio(audio).mean(dim=1)
                oa = (a * self.audio_layer_weights.softmax(0)[None, :, None]).sum(1)
            if 'video' in self.logit_terms:
                values['video'] = self.wv(ov)
            if 'audio' in self.logit_terms:
                values['audio'] = self.wa(oa)
            if 'interaction' in self.logit_terms:
                values['interaction'] = self.wf(ov * oa)
            terms = tuple(values[name] for name in self.logit_terms)
            logits = sum(terms)
            if return_aux:
                features = {}
                if 'video' in self.aux_heads:
                    features['video'] = ov
                if 'audio' in self.aux_heads:
                    features['audio'] = oa
                auxiliary = {name: head(features[name] if aux_mask is None else features[name][aux_mask])
                             for name, head in self.aux_heads.items()}
                return logits, terms, auxiliary
            return (logits, terms) if return_terms else logits

    return InteractionModel()


def load_pretrained(model, path):
    import torch
    # Only use trusted checkpoints. Supports upstream {'model': state_dict}.
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('model', checkpoint.get('module', checkpoint))
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    target = model.state_dict()
    kept = {k: v for k, v in state.items()
            if k in target and not k.startswith(('wv.', 'wa.', 'wf.', 'aux_heads.'))}
    required = [k for k in target if k.startswith(('encoder.', 'encoder_audio.'))]
    missing = [k for k in required if k not in kept]
    mismatched = [k for k, v in kept.items() if v.shape != target[k].shape]
    if missing or mismatched:
        raise ValueError(f'Checkpoint incompatible; missing encoder keys: {missing[:8]}; '
                         f'shape mismatches: {mismatched[:8]}. Match --embed-dim and --depth.')
    model.load_state_dict(kept, strict=False)
    print(f'Loaded {len(kept)} tensors; discarded fusion/decoder/old head. New interaction heads initialized.')


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
            if condition == 'audio_neutral':
                neutral = VideoReader(row['neutral_video'], ctx=cpu(0), num_threads=1)
                # One neutral midpoint face repeated across 16 frames: no mouth-motion cue.
                pixels = neutral.get_batch([len(neutral) // 2] * 16).asnumpy()
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


def train(splits, args, output, model_factory=None, pretrained_loader=None):
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = (model_factory or make_model)(args)
    (pretrained_loader or load_pretrained)(model, args.checkpoint)
    model.to(device)
    loaders = make_loaders(splits, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    prediction_dir = output / 'epoch_predictions'
    prediction_dir.mkdir()
    aux_weights = auxiliary_weights(args)
    use_auxiliary = any(aux_weights.values())
    if use_auxiliary and not hasattr(model, 'aux_heads'):
        raise ValueError('Auxiliary losses require the interaction model with auxiliary heads')

    def run_pass(loader, training=False):
        model.train(training)
        labels, predictions, losses, records = [], [], [], []
        objective_sum = 0.
        auxiliary_stats = {name: dict(loss_sum=0., correct=0, total=0)
                           for name, weight in aux_weights.items() if weight > 0}
        with torch.set_grad_enabled(training):
            for v, a, y, indices in loader:
                target = y.to(device)
                av_mask = torch.tensor([loader.dataset.rows[index].get('condition', 'both') == 'both'
                                        for index in indices.tolist()], device=device, dtype=torch.bool)
                auxiliary = {}
                if training and use_auxiliary:
                    logits, terms, auxiliary = model(v.to(device), a.to(device), return_terms=True,
                                                    return_aux=True, aux_mask=av_mask)
                else:
                    logits, terms = model(v.to(device), a.to(device), return_terms=True)
                # Terms come from this exact forward pass, including during training.
                torch.testing.assert_close(logits, sum(terms))
                loss, batch_losses, aux_losses = training_objective(logits, target, auxiliary, av_mask, aux_weights)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                objective_sum += loss.detach().item() * len(y)
                for name, values_aux in aux_losses.items():
                    stats = auxiliary_stats[name]
                    stats['loss_sum'] += values_aux.detach().sum().item()
                    stats['correct'] += (auxiliary[name].detach().argmax(1) == target[av_mask]).sum().item()
                    stats['total'] += len(values_aux)
                pred = logits.detach().argmax(1).cpu().tolist()
                probs = logits.detach().softmax(1).cpu().tolist()
                values = batch_losses.detach().cpu().tolist()
                term_values = [term.detach().cpu().tolist() for term in terms]
                logit_values = logits.detach().cpu().tolist()
                labels.extend(y.tolist())
                predictions.extend(pred)
                losses.extend(values)
                for offset, (index, label, prediction, value, probability) in enumerate(
                        zip(indices.tolist(), y.tolist(), pred, values, probs)):
                    row = loader.dataset.rows[index]
                    condition = row.get('condition', loader.dataset.condition if hasattr(loader.dataset, 'condition') else 'both')
                    noise_seed = row.get('noise_seed', '')
                    if training and condition == 'video_noise' and noise_seed != '':
                        noise_seed += getattr(loader.dataset, 'epoch', 0) * 1000000
                    record = dict(video=row['video'], actor=row['actor'], condition=condition,
                                  neutral_video=row.get('neutral_video', ''), noise_seed=noise_seed,
                                  label=label, prediction=prediction, correct=int(label == prediction), loss=value)
                    record.update(zip(['p_' + e for e in EMOTIONS], probability))
                    record.update(zip(['logit_' + e for e in EMOTIONS], logit_values[offset]))
                    for branch, branch_values in zip(model.logit_terms, term_values):
                        record.update(zip([branch + '_logit_' + e for e in EMOTIONS], branch_values[offset]))
                    records.append(record)
        metrics = dict(loss=float(np.mean(losses)), accuracy=float(accuracy_score(labels, predictions)),
                       correct=sum(y == p for y, p in zip(labels, predictions)), total=len(labels),
                       uar=float(recall_score(labels, predictions, labels=list(range(NUM_CLASSES)), average='macro', zero_division=0)),
                       macro_f1=float(f1_score(labels, predictions, labels=list(range(NUM_CLASSES)), average='macro', zero_division=0)),
                       confusion_matrix=confusion_matrix(labels, predictions, labels=list(range(NUM_CLASSES))).tolist())
        if training and use_auxiliary:
            metrics['objective_loss'] = objective_sum / len(labels)
            metrics['auxiliary'] = {name: dict(loss=stats['loss_sum'] / stats['total'] if stats['total'] else None,
                                               accuracy=stats['correct'] / stats['total'] if stats['total'] else None,
                                               correct=stats['correct'], total=stats['total'], weight=aux_weights[name])
                                    for name, stats in auxiliary_stats.items()}
        return metrics, records

    def save_predictions(path, records):
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    def save_epoch(epoch, name, metrics, records, learning_rate):
        path = output / 'epoch_metrics.csv'
        exists = path.exists()
        with path.open('a', newline='') as f:
            fields = ['epoch', 'split', 'loss', 'accuracy', 'correct', 'total', 'uar', 'macro_f1', 'learning_rate']
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(dict(epoch=epoch, split=name, learning_rate=learning_rate,
                                 **{k: metrics[k] for k in fields if k in metrics}))
        save_predictions(prediction_dir / ('%s_epoch_%03d.csv' % (name, epoch)), records)

    for epoch in range(args.epochs):
        learning_rate = optimizer.param_groups[0]['lr']
        loaders['train'].dataset.epoch = epoch
        train_metrics, train_records = run_pass(loaders['train'], training=True)
        save_epoch(epoch+1, 'train', train_metrics, train_records, learning_rate)
        if use_auxiliary:
            path = output / 'auxiliary_metrics.csv'
            exists = path.exists()
            with path.open('a', newline='') as f:
                fields = ['epoch', 'main_loss', 'objective_loss', 'aux_video_loss', 'aux_video_accuracy',
                          'aux_video_total', 'aux_audio_loss', 'aux_audio_accuracy', 'aux_audio_total']
                writer = csv.DictWriter(f, fieldnames=fields)
                if not exists:
                    writer.writeheader()
                values = dict(epoch=epoch+1, main_loss=train_metrics['loss'], objective_loss=train_metrics['objective_loss'])
                for name, stats in train_metrics['auxiliary'].items():
                    values.update({'aux_' + name + '_' + key: stats[key] for key in ('loss', 'accuracy', 'total')})
                writer.writerow(values)
        training_conditions = {}
        for condition in TEST_CONDITIONS:
            subset = [row for row in train_records if row['condition'] == condition]
            if not subset:
                continue
            true = [row['label'] for row in subset]
            pred = [row['prediction'] for row in subset]
            condition_metrics = dict(
                loss=float(np.mean([row['loss'] for row in subset])),
                accuracy=float(accuracy_score(true, pred)), correct=sum(row['correct'] for row in subset),
                total=len(subset),
                uar=float(recall_score(true, pred, labels=list(range(NUM_CLASSES)), average='macro', zero_division=0)),
                macro_f1=float(f1_score(true, pred, labels=list(range(NUM_CLASSES)), average='macro', zero_division=0)),
                confusion_matrix=confusion_matrix(true, pred, labels=list(range(NUM_CLASSES))).tolist())
            training_conditions[condition] = condition_metrics
            save_epoch(epoch+1, 'train_' + condition, condition_metrics, subset, learning_rate)
        tests = {}
        if not args.skip_epoch_tests:
            for condition in TEST_CONDITIONS:
                name = 'test_' + condition
                tests[condition], records = run_pass(loaders[name])
                save_epoch(epoch+1, name, tests[condition], records, learning_rate)
        record = dict(epoch=epoch+1, train=train_metrics, train_conditions=training_conditions, tests=tests,
                      learning_rate=learning_rate)
        print(json.dumps(record), flush=True)
        with (output / 'history.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        scheduler.step()
    # Fixed final epoch: test monitoring never selects a checkpoint.
    torch.save(dict(model=model.state_dict(), args=vars(args), epoch=args.epochs,
                    emotions=EMOTIONS, checkpoint_selection='final_epoch'), output / 'final.pth')
    weights = dict(emotions=np.asarray(EMOTIONS), logit_terms=np.asarray(model.logit_terms))
    if hasattr(model, 'heads'):
        weights.update({name: head.weight.detach().cpu().numpy() for name, head in model.heads.items()})
    else:
        for term, attribute in (('video', 'wv'), ('audio', 'wa'), ('interaction', 'wf')):
            if term in model.logit_terms:
                weights[attribute] = getattr(model, attribute).weight.detach().cpu().numpy()
    if hasattr(model, 'aux_heads'):
        weights.update({'aux_' + name: head.weight.detach().cpu().numpy() for name, head in model.aux_heads.items()})
    np.savez_compressed(output / 'classifier_weights.npz', **weights)
    summary = {}
    for condition in TEST_CONDITIONS:
        summary[condition], records = run_pass(loaders['test_' + condition])
        (output / f'test_metrics_{condition}.json').write_text(json.dumps(summary[condition], indent=2))
        save_predictions(output / f'test_predictions_{condition}.csv', records)
    (output / 'test_metrics.json').write_text(json.dumps(
        dict(emotions=EMOTIONS, checkpoint_epoch=args.epochs, checkpoint_file='final.pth',
             checkpoint_selection='final_epoch', architecture=getattr(args, 'architecture', 'interaction'),
             logit_terms=list(model.logit_terms), conditions=summary), indent=2))
    print('Final-epoch checkpoint test results:', json.dumps(summary))
    if not args.skip_explanations:
        from video_loss_explanations import export_test_explanations
        export_test_explanations(model, loaders, splits['test'], TEST_CONDITIONS,
                                 device, output, args.explain_limit)


def main(model_factory=None, pretrained_loader=None, architecture='interaction', logit_terms=LOGIT_TERMS):
    description = ('Fine-tune HiCMAE with selectable video/audio/fusion output heads.'
                   if architecture == 'fusion' else __doc__)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--video-root', required=True)
    parser.add_argument('--audio-root', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--output-dir', default='runs/ravdess_' + architecture)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embed-dim', type=int, choices=[384, 512], default=512)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--fusion-depth', type=int, default=2,
                        help='Cross-modal fusion layers (used by the separate fusion experiment)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--drop-path', type=float, default=.1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--noise-rms', type=float, default=0.05,
                        help='White-noise waveform RMS in video-only testing (default .05)')
    parser.add_argument('--skip-epoch-tests', action='store_true',
                        help='Evaluate test conditions only after training, not every epoch')
    parser.add_argument('--skip-explanations', action='store_true')
    parser.add_argument('--explain-limit', type=int, default=0,
                        help='Explain first N test trials in each condition; 0 means all')
    if architecture == 'fusion':
        parser.add_argument('--heads', nargs='+', choices=['visual', 'video', 'audio', 'fusion_video', 'fusion_audio'],
                            default=['video', 'audio', 'fusion_video', 'fusion_audio'],
                            help='Output heads to sum; visual is an alias for video. Default: all four.')
    else:
        parser.add_argument('--terms', '--heads', dest='terms', nargs='+',
                            choices=['visual', 'video', 'audio', 'interaction'], default=list(LOGIT_TERMS),
                            help='Terms to sum; visual aliases video. Default: visual interaction.')
        parser.add_argument('--aux-video-weight', type=float, default=.5,
                            help='Training-only visual CE weight on real AV examples; 0 disables (default .5)')
        parser.add_argument('--aux-audio-weight', type=float, default=.5,
                            help='Training-only audio CE weight on real AV examples; 0 disables (default .5)')
    args = parser.parse_args()
    args.training_conditions = list(TEST_CONDITIONS)
    args.logit_terms = list(logit_terms)
    if architecture == 'fusion':
        from run_ravdess_fusion import normalize_heads
        try:
            args.heads = list(normalize_heads(args.heads))
        except ValueError as error:
            parser.error(str(error))
        args.logit_terms = list(args.heads)
    else:
        try:
            args.terms = list(normalize_terms(args.terms))
        except ValueError as error:
            parser.error(str(error))
        args.logit_terms = list(args.terms)
    args.architecture = architecture
    try:
        auxiliary_weights(args)
    except ValueError as error:
        parser.error(str(error))
    if args.explain_limit < 0:
        parser.error('--explain-limit must be nonnegative')
    if args.epochs < 1 or args.batch_size < 1 or args.depth < 1 or args.lr <= 0:
        parser.error('epochs, batch-size, depth and lr must be positive')
    if architecture == 'fusion' and args.fusion_depth < 1:
        parser.error('--fusion-depth must be positive for the fusion experiment')
    if not args.prepare_only and not args.checkpoint:
        parser.error('--checkpoint is required for fine-tuning')
    if not 0 < args.noise_rms <= 1:
        parser.error('--noise-rms must be in (0, 1]')
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error('Use a new/empty output directory to avoid overwriting an experiment')
    rows = discover(args.video_root, args.audio_root)
    splits, report = balanced_split(rows, args.seed)
    attach_test_controls(splits, rows, args.seed)
    expanded_train = expand_training_rows(splits['train'])
    report['train_expanded'] = dict(total=len(expanded_train), conditions=list(TEST_CONDITIONS),
                                   counts_per_condition={c: len(splits['train']) for c in TEST_CONDITIONS},
                                   counts_per_emotion=[384] * NUM_CLASSES)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
    (output / 'split_report.json').write_text(json.dumps(dict(emotions=EMOTIONS, split_method='mixed_actor_trial_split', splits=report), indent=2))
    for name, rows in splits.items():
        with (output / f'{name}.csv').open('w', newline='') as f:
            fields = ['video', 'audio', 'label', 'actor']
            fields += ['neutral_video', 'noise_seed']
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with (output / 'train_expanded.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['video', 'audio', 'label', 'actor',
                                              'neutral_video', 'noise_seed', 'condition'])
        writer.writeheader()
        writer.writerows(expanded_train)
    print(json.dumps(report, indent=2))
    if not args.prepare_only:
        train(splits, args, output, model_factory=model_factory, pretrained_loader=pretrained_loader)


if __name__ == '__main__':
    main()
