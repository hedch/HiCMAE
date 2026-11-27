"""Train independent ungated audiovisual MLPs on gender-balanced actor holdouts."""
import argparse
import json
import random
from functools import partial
from pathlib import Path

import run_ravdess_indicator_mlp as data

TRAIN = ('audio_noise', 'video_noise', 'both')
TEST = data.TEST_CONDITIONS
EMOTIONS = data.EMOTIONS


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def actor_groups(seed):
    rng = random.Random(seed)
    male, female = list(range(1, 25, 2)), list(range(2, 25, 2))
    rng.shuffle(male)
    rng.shuffle(female)
    return [sorted(male[i:i+2] + female[i:i+2]) for i in range(0, 12, 2)]


def allocate(actors, counts, seed):
    if len(counts) != 3 or min(counts) < 1 or sum(counts) != 20:
        raise ValueError('Supply three positive actor counts summing to 20.')
    rng = random.Random(seed)
    male = sorted(a for a in actors if a % 2)
    female = sorted(set(actors) - set(male))
    rng.shuffle(male)
    rng.shuffle(female)
    odd = [i for i, n in enumerate(counts) if n % 2]
    extra = set(rng.sample(odd, len(odd)//2))
    result = {}
    for i, (condition, count) in enumerate(zip(TRAIN, counts)):
        m = count//2 + int(i in extra)
        f = count-m
        result[condition] = sorted(male[:m] + female[:f])
        del male[:m], female[:f]
    return result


def experiment_units(root, settings):
    """Yield every expected model, including legacy one-model-per-fold runs."""
    root = Path(root)
    for fold in settings['holdout_groups']:
        folder = root / f'fold_{fold:02d}'
        if 'blocks_per_fold' not in settings:
            yield fold, 1, folder
        else:
            for block in range(1, settings['blocks_per_fold'] + 1):
                yield fold, block, folder / f'block_{block:02d}'


def block_assignments(groups, fold, counts, seed, blocks):
    """Draw distinct gender-balanced assignments reproducibly within a fold."""
    actors = set(range(1, 25)) - set(groups[fold-1])
    seen = set()
    assignments = []
    for attempt in range(10000):
        assignment_seed = seed + fold * 10000 + attempt
        assigned = allocate(actors, counts, assignment_seed)
        signature = tuple(tuple(assigned[c]) for c in TRAIN)
        if signature in seen:
            continue
        seen.add(signature)
        assignments.append((assignment_seed, assigned))
        if len(assignments) == blocks:
            return assignments
    raise ValueError('Could not generate the requested distinct actor assignments.')


def prepare(rows, groups, fold, args, assigned=None):
    test = groups[fold-1]
    if assigned is None:
        assigned = allocate(set(range(1, 25))-set(test), args.train_actors, args.seed+fold)
    condition = {a: c for c, actors in assigned.items() for a in actors}
    splits = {'train': [], 'test': []}
    # Equalize emotions within each actor, preserving the actor allocation.
    rng = random.Random(args.seed + fold)
    for actor in range(1, 25):
        buckets = [[r for r in rows if r['actor'] == actor and r['label'] == j]
                   for j in range(len(EMOTIONS))]
        count = min(map(len, buckets))
        if count == 0:
            raise ValueError(f'Actor {actor} is missing a target emotion.')
        for bucket in buckets:
            for row in rng.sample(bucket, count):
                if actor in condition:
                    splits['train'].append(dict(row, condition=condition[actor]))
                else:
                    splits['test'].append(dict(row))
    data.attach_test_controls({'test': splits['test']}, rows, args.seed+fold)
    for index, row in enumerate(splits['train']):
        row['noise_seed'] = args.seed + fold + index
    return splits, dict(test=test, training=assigned)


def make_model(args):
    import torch
    from torch import nn
    from modeling_finetune_av import AudioVisionTransformer

    class ActorMLP(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = AudioVisionTransformer(img_size=160, patch_size=16, embed_dim=args.embed_dim,
                num_heads=args.embed_dim//64, depth=args.depth, img_size_audio=(256, 128),
                patch_size_audio=16, embed_dim_audio=args.embed_dim, num_heads_audio=args.embed_dim//64,
                depth_audio=args.depth, fusion_depth=0, num_classes=6, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), all_frames=16, drop_path_rate=args.drop_path)
            self.encoder, self.encoder_audio = backbone.encoder, backbone.encoder_audio
            self.video_layer_weights = backbone.video_layer_weights
            self.audio_layer_weights = backbone.audio_layer_weights
            width = 2*args.embed_dim
            layers = []
            for size in args.mlp_hidden_dims:
                layers.extend([nn.Linear(width, size), nn.GELU(), nn.Dropout(args.mlp_dropout)])
                width = size
            self.classifier = nn.Sequential(*layers, nn.Linear(width, len(EMOTIONS)))

        def forward(self, video, audio):
            v = self.encoder(video).mean(1)
            a = self.encoder_audio(audio).mean(1)
            v = (v * self.video_layer_weights.softmax(0)[None, :, None]).sum(1)
            a = (a * self.audio_layer_weights.softmax(0)[None, :, None]).sum(1)
            return self.classifier(torch.cat((v, a), dim=-1))

    return ActorMLP()


def trial_losses(logits, labels, mode):
    from torch.nn import functional as F
    if mode == 'six-way':
        return F.cross_entropy(logits, labels, reduction='none')
    targets = F.one_hot(labels, len(EMOTIONS)).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='none').mean(1)


def run_pass(model, loader, args, optimizer=None):
    import torch
    training = optimizer is not None
    model.train(training)
    records = []
    with torch.set_grad_enabled(training):
        for video, audio, labels, indices in loader:
            labels = labels.to(args.device)
            logits = model(video.to(args.device), audio.to(args.device))
            losses = trial_losses(logits, labels, args.classification)
            if not torch.isfinite(losses).all():
                raise ValueError('Nonfinite emotion loss')
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses.mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            scores = logits.detach()
            probs = scores.softmax(1) if args.classification == 'six-way' else scores.sigmoid()
            for i, pred, loss, outputs, probabilities in zip(indices.tolist(), scores.argmax(1).tolist(),
                    losses.tolist(), scores.tolist(), probs.tolist()):
                row = dict(loader.dataset.rows[i])
                row.setdefault('condition', loader.dataset.condition)
                row.update(prediction=pred, correct=int(pred == row['label']), loss=loss)
                row.update(zip(('logit_'+e for e in EMOTIONS), outputs))
                row.update(zip(('p_'+e for e in EMOTIONS), probabilities))
                records.append(row)
    return records


def train_fold(splits, args, folder):
    import numpy as np
    import torch
    random.seed(args.fold_seed)
    np.random.seed(args.fold_seed)
    torch.manual_seed(args.fold_seed)
    raw = make_model(args)
    if args.checkpoint:
        data.load_pretrained(raw, args.checkpoint)
    model, devices = data.configure_parallel_model(raw, args)
    save_json(folder/'parallelism.json', devices)
    loaders = data.make_loaders(splits, args, assigned_conditions=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    history = []
    for epoch in range(1, args.epochs+1):
        loaders['train'].dataset.epoch = epoch-1
        epoch_dir = folder/f'epoch_{epoch:03d}'
        epoch_dir.mkdir()
        for split, loader in loaders.items():
            training = split == 'train'
            records = run_pass(model, loader, args, optimizer if training else None)
            data.write_csv(epoch_dir/(split+'.csv'), records)
            subsets = {split: records}
            if training:
                subsets.update({'train_'+c: [r for r in records if r['condition']==c] for c in TRAIN})
            for name, subset in subsets.items():
                metrics = data.classification_metrics(subset)
                matrix = metrics.pop('confusion_matrix')
                save_json(epoch_dir/(name+'_confusion.json'), matrix)
                history.append(dict(fold=args.fold, block=getattr(args, 'block', 1), epoch=epoch, split=name, **metrics))
        scheduler.step()
        data.write_csv(folder/'metrics.csv', history)
        av_accuracy = next(r['accuracy'] for r in reversed(history) if r['split'] == 'test_both')
        print(f'Fold {args.fold}, block {args.block}, epoch {epoch}/{args.epochs}, AV test accuracy {av_accuracy:.3f}', flush=True)
    torch.save(raw.state_dict(), folder/'final.pth')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('video-root','audio-root','output-dir'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--train-actors', type=int, nargs=3, metavar=('A_NOISE','V_NOISE','AV'), default=[6,6,8])
    p.add_argument('--holdout-groups', type=int, nargs='+', default=[1,2,3,4,5,6], help='Distinct group IDs 1 to 6.')
    p.add_argument('--blocks-per-fold', type=int, default=4, help='Fresh models with distinct training actor allocations per fold.')
    p.add_argument('--classification', choices=('six-way','binary'), default='six-way')
    p.add_argument('--mlp-hidden-dims', type=int, nargs='+', default=[256,128])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=.05)
    p.add_argument('--checkpoint')
    p.add_argument('--embed-dim', type=int, choices=(384,512), default=512)
    p.add_argument('--depth', type=int, default=10)
    p.add_argument('--drop-path', type=float, default=.1)
    p.add_argument('--mlp-dropout', type=float, default=.1)
    p.add_argument('--noise-rms', type=float, default=.05)
    p.add_argument('--device', default='cuda')
    p.add_argument('--gpu-ids', type=int, nargs='+')
    p.add_argument('--skip-explanations', action='store_true', help=argparse.SUPPRESS)  # Legacy no-op.
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    if (len(set(args.holdout_groups)) != len(args.holdout_groups)
            or not set(args.holdout_groups) <= set(range(1,7))):
        p.error('Use distinct holdout group IDs between 1 and 6.')
    if sum(args.train_actors)!=20 or min(args.train_actors)<1:
        p.error('The three positive training actor counts must sum to 20.')
    if min(args.batch_size,args.epochs,args.depth,args.blocks_per_fold,*args.mlp_hidden_dims)<1 or args.seed<0:
        p.error('Positive dimensions, epochs, batch size and a nonnegative seed are required.')
    args.neutral_frame_policy = 'first'
    args.training_audio_only_visual = 'gaussian_noise'
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        p.error('Choose an empty experiment output directory.')
    rows = data.discover(args.video_root,args.audio_root)
    groups = actor_groups(args.seed)
    root.mkdir(parents=True, exist_ok=True)
    save_json(root/'experiment.json', dict(vars(args), groups=groups, emotions=EMOTIONS))
    # Prepare every manifest before starting expensive training.
    for fold in args.holdout_groups:
        assignments = block_assignments(groups, fold, args.train_actors, args.seed, args.blocks_per_fold)
        for block, (assignment_seed, assigned) in enumerate(assignments, start=1):
            # Same initialization and training RNG seed within a fold isolates
            # changes in actor allocation. Every block creates a fresh model.
            config = argparse.Namespace(**vars(args), fold=fold, block=block,
                                        fold_seed=args.seed+fold, assignment_seed=assignment_seed)
            folder = root / f'fold_{fold:02d}' / f'block_{block:02d}'
            folder.mkdir(parents=True)
            splits, actors = prepare(rows, groups, fold, config, assigned=assigned)
            save_json(folder/'actors.json', actors)
            save_json(folder/'config.json', vars(config))
            for split, records in splits.items():
                data.write_csv(folder/(split+'.csv'), records)
    if not args.prepare_only:
        import csv
        for fold, block, folder in experiment_units(root, vars(args)):
            config = argparse.Namespace(**json.loads((folder/'config.json').read_text()))
            splits = {}
            for split in ('train', 'test'):
                with (folder/(split+'.csv')).open(newline='') as stream:
                    records = list(csv.DictReader(stream))
                for row in records:
                    for key in ('actor', 'label', 'noise_seed'):
                        row[key] = int(row[key])
                splits[split] = records
            train_fold(splits, config, folder)
    if not args.prepare_only:
        from analyze_ravdess_actor_mlp import analyze
        analyze(root)


if __name__ == '__main__':
    main()
