"""Fine-tune HiCMAE encoders with additive + Hadamard interaction logits.

Run --help for options; see RAVDESS_INTERACTION.md for data preparation.
Only --prepare-only can run without the upstream ML dependencies.
"""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def discover(video_root, audio_root):
    """Pair speech MP4s with WAVs using the six non-modality filename fields."""
    audios = {}
    for path in sorted(Path(audio_root).rglob('*.wav')):
        parts = path.stem.split('-')
        if len(parts) == 7 and parts[:2] == ['03', '01']:
            key = '-'.join(parts[1:])
            if key in audios:
                raise ValueError(f'Duplicate audio trial: {key}')
            audios[key] = path.resolve()
    videos = {}
    for path in sorted(Path(video_root).rglob('*.mp4')):
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
        label, actor = int(parts[2]) - 1, int(parts[6])
        if not 0 <= label < 8 or not 1 <= actor <= 24:
            raise ValueError(f'Invalid RAVDESS filename: {video}')
        rows.append(dict(video=str(video), audio=str(audios[key]), label=label, actor=actor))
    if not rows:
        raise ValueError('No paired RAVDESS speech MP4/WAV files found.')
    return rows


def balanced_split(rows, seed, val_actors=None, test_actors=None):
    actors = sorted({r['actor'] for r in rows})
    if val_actors is None and test_actors is None:
        if len(actors) != 24:
            raise ValueError('Default split requires all 24 actors; otherwise specify both actor lists.')
        random.Random(seed).shuffle(actors)
        val_actors, test_actors = actors[16:20], actors[20:24]
    if not val_actors or not test_actors:
        raise ValueError('Specify both nonempty --val-actors and --test-actors.')
    val, test = set(val_actors), set(test_actors)
    train = set(actors) - val - test
    if val & test or not (val | test) <= set(actors) or not train:
        raise ValueError('Actor groups must be nonempty, present, and disjoint.')
    result, report = {}, {}
    for i, (name, group) in enumerate([('train', train), ('val', val), ('test', test)]):
        buckets = [[r for r in rows if r['actor'] in group and r['label'] == c] for c in range(8)]
        n = min(map(len, buckets))
        if n == 0:
            raise ValueError(f'{name}: at least one emotion is absent.')
        rng = random.Random(seed + i)
        selected = []
        for bucket in buckets:
            selected.extend(rng.sample(bucket, n))
        rng.shuffle(selected)
        result[name] = selected
        report[name] = dict(actors=sorted(group), original_counts=[len(b) for b in buckets],
                            balanced_counts=[n] * 8, total=8 * n)
    return result, report


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
                depth_audio=args.depth, fusion_depth=0, num_classes=8,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                all_frames=16, drop_path_rate=args.drop_path)
            # Register only encoders and layer pooling weights: no fusion module.
            self.encoder = backbone.encoder
            self.encoder_audio = backbone.encoder_audio
            self.video_layer_weights = backbone.video_layer_weights
            self.audio_layer_weights = backbone.audio_layer_weights
            self.wv = nn.Linear(args.embed_dim, 8, bias=False)
            self.wa = nn.Linear(args.embed_dim, 8, bias=False)
            self.wf = nn.Linear(args.embed_dim, 8, bias=False)

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
        def __init__(self, rows, training):
            self.rows, self.training = rows, training

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
            frames = torch.from_numpy(vr.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).float() / 255
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
            fb = torchaudio.compliance.kaldi.fbank(
                wave, htk_compat=True, sample_frequency=16000, use_energy=False,
                window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
            fb = torch.nn.functional.pad(fb[:256], (0, 0, 0, max(0, 256-fb.shape[0])))
            audio = ((fb + 4.2677393) / (4.5689974 * 2)).unsqueeze(0)
            return video, audio, row['label']

    # num_workers=0 default also supports Windows: nested Dataset is not spawn-picklable.
    return {name: DataLoader(PairedDataset(rows, name == 'train'), batch_size=args.batch_size,
                            shuffle=name == 'train', num_workers=0,
                            pin_memory=args.device.startswith('cuda'), drop_last=False)
            for name, rows in splits.items()}


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
    criterion = torch.nn.CrossEntropyLoss()

    def evaluate(loader):
        model.eval()
        labels, predictions, probabilities = [], [], []
        with torch.no_grad():
            for v, a, y in loader:
                logits = model(v.to(device), a.to(device))
                labels.extend(y.tolist())
                predictions.extend(logits.argmax(1).cpu().tolist())
                probabilities.extend(logits.softmax(1).cpu().tolist())
        metrics = dict(accuracy=accuracy_score(labels, predictions),
                       uar=recall_score(labels, predictions, labels=list(range(8)), average='macro', zero_division=0),
                       macro_f1=f1_score(labels, predictions, labels=list(range(8)), average='macro', zero_division=0),
                       confusion_matrix=confusion_matrix(labels, predictions, labels=list(range(8))).tolist())
        return metrics, labels, predictions, probabilities

    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for v, a, y in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(v.to(device), a.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
        scheduler.step()
        metrics, *_ = evaluate(loaders['val'])
        record = dict(epoch=epoch+1, train_loss=total_loss/len(splits['train']), val=metrics)
        print(json.dumps(record), flush=True)
        with (output / 'history.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        if metrics['uar'] > best:
            best = metrics['uar']
            torch.save(dict(model=model.state_dict(), args=vars(args), epoch=epoch+1,
                            emotions=EMOTIONS, val=metrics), output / 'best.pth')
    best_state = torch.load(output / 'best.pth', map_location=device)
    model.load_state_dict(best_state['model'])
    metrics, labels, predictions, probabilities = evaluate(loaders['test'])
    (output / 'test_metrics.json').write_text(json.dumps(metrics, indent=2))
    with (output / 'test_predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video', 'actor', 'label', 'prediction'] + [f'p_{e}' for e in EMOTIONS])
        for row, y, pred, prob in zip(splits['test'], labels, predictions, probabilities):
            writer.writerow([row['video'], row['actor'], y, pred] + prob)
    print('Best-validation checkpoint test results:', json.dumps(metrics))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-root', required=True)
    parser.add_argument('--audio-root', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--output-dir', default='runs/ravdess_interaction')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--val-actors', type=int, nargs='+')
    parser.add_argument('--test-actors', type=int, nargs='+')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embed-dim', type=int, choices=[384, 512], default=512)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--drop-path', type=float, default=.1)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.depth < 1 or args.lr <= 0:
        parser.error('epochs, batch-size, depth and lr must be positive')
    if not args.prepare_only and not args.checkpoint:
        parser.error('--checkpoint is required for fine-tuning')
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error('Use a new/empty output directory to avoid overwriting an experiment')
    splits, report = balanced_split(discover(args.video_root, args.audio_root), args.seed,
                                    args.val_actors, args.test_actors)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
    (output / 'split_report.json').write_text(json.dumps(dict(emotions=EMOTIONS, splits=report), indent=2))
    for name, rows in splits.items():
        with (output / f'{name}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['video', 'audio', 'label', 'actor'])
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(report, indent=2))
    if not args.prepare_only:
        train(splits, args, output)


if __name__ == '__main__':
    main()
