"""Retest final actor-MLP checkpoints with one full-width final video head and explain its loss."""
import argparse
import csv
import json
from pathlib import Path

from run_ravdess_actor_mlp import TEST, data, experiment_units, make_model, run_pass, save_json, trial_losses


def retest(folder, device, limit, batch_size):
    import torch
    from video_loss_explanations import export_test_explanations

    args = argparse.Namespace(**json.loads((folder / 'config.json').read_text()))
    args.device = device
    if batch_size is not None:
        args.batch_size = batch_size
    with (folder / 'test.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ('label', 'actor', 'noise_seed'):
            row[key] = int(row[key])
    loaders = data.make_loaders({'test': rows}, args)
    model = make_model(args)
    model.load_state_dict(torch.load(folder / 'final.pth', map_location='cpu'), strict=True)
    attention = model.encoder.blocks[-1].attn
    trained_heads = attention.num_heads
    # Native Attention.forward reshapes the learned Q/K/V using num_heads.
    # Reuse every channel and the learned output projection with one softmax.
    width = attention.proj.in_features
    attention.num_heads = 1
    attention.scale = width ** -0.5
    model.to(device).eval()

    output = folder / 'single_head_retest'
    output.mkdir(exist_ok=True)
    summary = dict(checkpoint='final.pth', epoch=args.epochs, classification=args.classification,
                   trained_final_video_heads=trained_heads, tested_final_video_heads=1,
                   head_width=width, conditions={})
    for condition in TEST:
        split = 'test_' + condition
        records = run_pass(model, loaders[split], args)
        data.write_csv(output / (split + '.csv'), records)
        summary['conditions'][condition] = data.classification_metrics(records)
        print(f"{folder.name} {condition} accuracy {summary['conditions'][condition]['accuracy']:.3f}", flush=True)
    save_json(output / 'metrics.json', summary)

    def loss_fn(logits, labels):
        return trial_losses(logits, labels, args.classification).mean()

    export_test_explanations(model, loaders, rows, TEST, device, output, limit,
                             loss_fn=loss_fn, target_name=args.classification + ' emotion loss',
                             last_layer_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path, help='Existing training experiment directory.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fold', nargs='+', type=int)
    parser.add_argument('--block', nargs='+', type=int)
    parser.add_argument('--batch-size', type=int, help='Retest batch size. Defaults to the training value.')
    parser.add_argument('--explain-limit', type=int, default=0, help='Trials per condition to explain. Zero explains all. Testing always covers every trial.')
    options = parser.parse_args()
    if options.explain_limit < 0 or (options.batch_size is not None and options.batch_size < 1):
        parser.error('Use a nonnegative explanation limit and a positive batch size.')
    settings = json.loads((options.output_dir / 'experiment.json').read_text())
    units = list(experiment_units(options.output_dir, settings))
    for selection, available in [(options.fold, {f for f, b, p in units}),
                                 (options.block, {b for f, b, p in units})]:
        if selection and not set(selection) <= available:
            parser.error(f'Selected IDs {selection} outside available IDs {sorted(available)}.')
    for fold, block, folder in units:
        if (options.fold is None or fold in options.fold) and (options.block is None or block in options.block):
            print(f'Retesting fold {fold}, block {block}', flush=True)
            retest(folder, options.device, options.explain_limit, options.batch_size)


if __name__ == '__main__':
    main()
