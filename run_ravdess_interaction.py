"""Fine-tune HiCMAE encoders with additive + Hadamard interaction logits.

Run --help for options; see RAVDESS_INTERACTION.md for data preparation.
Only --prepare-only can run without the upstream ML dependencies.
"""
import argparse
import csv
import json
import random
from pathlib import Path

EMOTIONS = ['happy', 'sad', 'angry', 'fearful', 'surprised', 'disgusted']
# RAVDESS codes: 07=disgust, 08=surprised. Preserve the requested class order.
EMOTION_CODES = {3: 0, 4: 1, 5: 2, 6: 3, 8: 4, 7: 5}
NUM_CLASSES = len(EMOTIONS)
TEST_CONDITIONS = ('both', 'video_noise', 'audio_neutral')


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
    """Use neutral controls matched on actor, statement and repetition; controls are not training targets."""
    references = {}
    for row in rows:
        if row['label'] == -1:
            parts = Path(row['video']).stem.split('-')
            key = (row['actor'], parts[4], parts[5])
            if key in references:
                raise ValueError(f'Duplicate neutral reference: {key}')
            references[key] = row['video']
    for i, row in enumerate(splits['test']):
        parts = Path(row['video']).stem.split('-')
        key = (row['actor'], parts[4], parts[5])
        if key not in references:
            raise ValueError(f'Missing neutral speech video for test actor/statement/repetition {key}')
        row['neutral_video'] = references[key]
        row['noise_seed'] = seed + 100000 + i


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
            self.wv = nn.Linear(args.embed_dim, NUM_CLASSES, bias=False)
            self.wa = nn.Linear(args.embed_dim, NUM_CLASSES, bias=False)
            self.wf = nn.Linear(args.embed_dim, NUM_CLASSES, bias=False)

        def forward(self, video, audio, return_terms=False):
            v = self.encoder(video).mean(dim=1)  # B,L,D
            a = self.encoder_audio(audio).mean(dim=1)
            ov = (v * self.video_layer_weights.softmax(0)[None, :, None]).sum(1)
            oa = (a * self.audio_layer_weights.softmax(0)[None, :, None]).sum(1)
            terms = (self.wv(ov), self.wa(oa), self.wf(ov * oa))
            logits = terms[0] + terms[1] + terms[2]
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
            if k in target and not k.startswith(('wv.', 'wa.', 'wf.'))}
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

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
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
            if self.condition == 'audio_neutral':
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
            if self.condition == 'video_noise':
                wave = white_noise(40960, row['noise_seed'], args.noise_rms)
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
    loaders = dict(train=loader(splits['train'], training=True))
    for condition in TEST_CONDITIONS:
        loaders['test_' + condition] = loader(splits['test'], condition=condition)
    return loaders


def train(splits, args, output):
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = make_model(args)
    load_pretrained(model, args.checkpoint)
    model.to(device)
    loaders = make_loaders(splits, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    prediction_dir = output / 'epoch_predictions'
    prediction_dir.mkdir()

    def run_pass(loader, training=False):
        model.train(training)
        labels, predictions, losses, records = [], [], [], []
        with torch.set_grad_enabled(training):
            for v, a, y, indices in loader:
                target = y.to(device)
                logits = model(v.to(device), a.to(device))
                batch_losses = torch.nn.functional.cross_entropy(logits, target, reduction='none')
                loss = batch_losses.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                pred = logits.detach().argmax(1).cpu().tolist()
                probs = logits.detach().softmax(1).cpu().tolist()
                values = batch_losses.detach().cpu().tolist()
                labels.extend(y.tolist())
                predictions.extend(pred)
                losses.extend(values)
                for index, label, prediction, value, probability in zip(indices.tolist(), y.tolist(), pred, values, probs):
                    row = loader.dataset.rows[index]
                    record = dict(video=row['video'], actor=row['actor'],
                                  neutral_video=row.get('neutral_video', ''), noise_seed=row.get('noise_seed', ''),
                                  label=label, prediction=prediction, correct=int(label == prediction), loss=value)
                    record.update(zip(['p_' + e for e in EMOTIONS], probability))
                    records.append(record)
        metrics = dict(loss=float(np.mean(losses)), accuracy=float(accuracy_score(labels, predictions)),
                       correct=sum(y == p for y, p in zip(labels, predictions)), total=len(labels),
                       uar=float(recall_score(labels, predictions, labels=list(range(NUM_CLASSES)), average='macro', zero_division=0)),
                       macro_f1=float(f1_score(labels, predictions, labels=list(range(NUM_CLASSES)), average='macro', zero_division=0)),
                       confusion_matrix=confusion_matrix(labels, predictions, labels=list(range(NUM_CLASSES))).tolist())
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
        train_metrics, train_records = run_pass(loaders['train'], training=True)
        save_epoch(epoch+1, 'train', train_metrics, train_records, learning_rate)
        tests = {}
        if not args.skip_epoch_tests:
            for condition in TEST_CONDITIONS:
                name = 'test_' + condition
                tests[condition], records = run_pass(loaders[name])
                save_epoch(epoch+1, name, tests[condition], records, learning_rate)
        record = dict(epoch=epoch+1, train=train_metrics, tests=tests,
                      learning_rate=learning_rate)
        print(json.dumps(record), flush=True)
        with (output / 'history.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        scheduler.step()
    # Fixed final epoch: test monitoring never selects a checkpoint.
    torch.save(dict(model=model.state_dict(), args=vars(args), epoch=args.epochs,
                    emotions=EMOTIONS, checkpoint_selection='final_epoch'), output / 'final.pth')
    summary = {}
    for condition in TEST_CONDITIONS:
        summary[condition], records = run_pass(loaders['test_' + condition])
        (output / f'test_metrics_{condition}.json').write_text(json.dumps(summary[condition], indent=2))
        save_predictions(output / f'test_predictions_{condition}.csv', records)
    (output / 'test_metrics.json').write_text(json.dumps(
        dict(emotions=EMOTIONS, checkpoint_epoch=args.epochs, checkpoint_file='final.pth',
             checkpoint_selection='final_epoch', conditions=summary), indent=2))
    print('Final-epoch checkpoint test results:', json.dumps(summary))
    if not args.skip_explanations:
        from video_loss_explanations import export_test_explanations
        export_test_explanations(model, loaders, splits['test'], TEST_CONDITIONS,
                                 device, output, args.explain_limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-root', required=True)
    parser.add_argument('--audio-root', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--output-dir', default='runs/ravdess_interaction')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embed-dim', type=int, choices=[384, 512], default=512)
    parser.add_argument('--depth', type=int, default=10)
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
    args = parser.parse_args()
    if args.explain_limit < 0:
        parser.error('--explain-limit must be nonnegative')
    if args.epochs < 1 or args.batch_size < 1 or args.depth < 1 or args.lr <= 0:
        parser.error('epochs, batch-size, depth and lr must be positive')
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
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
    (output / 'split_report.json').write_text(json.dumps(dict(emotions=EMOTIONS, split_method='mixed_actor_trial_split', splits=report), indent=2))
    for name, rows in splits.items():
        with (output / f'{name}.csv').open('w', newline='') as f:
            fields = ['video', 'audio', 'label', 'actor']
            if name == 'test':
                fields += ['neutral_video', 'noise_seed']
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(report, indent=2))
    if not args.prepare_only:
        train(splits, args, output)


if __name__ == '__main__':
    main()
