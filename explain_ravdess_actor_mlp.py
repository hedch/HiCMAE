"""Export video loss explanations for each selected actor holdout."""
import argparse
import csv
import json
from pathlib import Path
from run_ravdess_actor_mlp import TEST, data, make_model, trial_losses


def export_fold(model, loaders, rows, args, folder, limit):
    from video_loss_explanations import export_test_explanations
    def loss_fn(logits, labels):
        return trial_losses(logits, labels, args.classification).mean()
    target = 'six-way cross entropy' if args.classification == 'six-way' else 'mean of six one-vs-rest binary cross entropies'
    export_test_explanations(model, loaders, rows, TEST, args.device, folder, limit,
                             loss_fn=loss_fn, target_name=target)


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--explain-limit', type=int, default=0, help='Trials per fold; zero means all.')
    p.add_argument('--fold', type=int, nargs='+')
    options = p.parse_args()
    if options.explain_limit < 0:
        p.error('Use a nonnegative explanation limit.')
    settings = json.loads((options.output_dir/'experiment.json').read_text())
    folds = options.fold or settings['holdout_groups']
    if not set(folds) <= set(settings['holdout_groups']):
        p.error('A selected fold is absent from the experiment.')
    for fold in folds:
        folder = options.output_dir/f'fold_{fold:02d}'
        args = argparse.Namespace(**json.loads((folder/'config.json').read_text()))
        args.device = options.device
        splits = {}
        for name in ('train', 'test'):
            with (folder/(name+'.csv')).open(newline='') as f:
                splits[name] = list(csv.DictReader(f))
            for row in splits[name]:
                for k in ('label','actor','noise_seed'):
                    row[k] = int(row[k])
        model = make_model(args)
        model.load_state_dict(torch.load(folder/'final.pth', map_location='cpu'))
        model.to(args.device).eval()
        loaders = data.make_loaders(splits, args, assigned_conditions=True)
        export_fold(model, loaders, splits['test'], args, folder, options.explain_limit)
        del model, loaders


if __name__ == '__main__':
    main()
