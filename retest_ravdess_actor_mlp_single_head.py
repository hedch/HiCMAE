"""Retest final actor-MLP checkpoints with one full-width final video head and explain its loss."""
import argparse
import csv
import json
from pathlib import Path

from run_ravdess_actor_mlp import EMOTIONS, TEST, data, experiment_units, make_model, run_pass, save_json, trial_losses


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
                             last_layer_only=True, save_frame_images=False)


def neutral_template(video_root):
    """Read Actor 01's first frame in the same spatial coordinates as model inputs."""
    import torch
    from decord import VideoReader, cpu

    candidates = []
    for path in Path(video_root).rglob('*'):
        parts = path.stem.split('-')
        if (path.suffix.lower() == '.mp4' and len(parts) == 7
                and parts[0] in ('01', '02') and parts[1:4] == ['01', '01', '01']
                and parts[-1] == '01' and path.is_file()):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f'No RAVDESS Actor 01 neutral speech MP4 found under {video_root}')
    path = min(candidates, key=lambda item: (item.stem[3:], item.stem[:2] != '02', str(item)))
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    frame = torch.from_numpy(reader[0].asnumpy()).permute(2, 0, 1).float() / 255
    height, width = frame.shape[-2:]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    frame = frame[:, top:top + side, left:left + side]
    frame = torch.nn.functional.interpolate(frame[None], (160, 160), mode='bilinear', align_corners=False)
    return frame[0].permute(1, 2, 0).numpy(), path


def summarize_heatmaps(run_folder, folders, template, template_path):
    """Average time and trials by true emotion and test condition, then render one grid."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.figure import Figure

    totals, counts = {}, {}
    for folder in folders:
        root = folder / 'single_head_retest' / 'explanations'
        manifest = root / 'index.jsonl'
        if not manifest.is_file() or not manifest.stat().st_size:
            raise ValueError(f'No saved heatmaps indexed in {manifest}')
        with manifest.open(encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line)
                key = (record['label'], record['condition'])
                # Rebase stored paths so summaries work if a run folder has moved.
                trial_folder = record['directory'].replace(chr(92), '/').rstrip('/').split('/')[-1]
                with np.load(root / record['condition'] / trial_folder / 'heatmaps.npz') as arrays:
                    scale = max(record['shared_positive_scale'], 1e-12)
                    spatial = np.clip(arrays['positive'] / scale, 0, 1).mean(axis=0, dtype=np.float64)
                totals[key] = totals.get(key, 0) + spatial
                counts[key] = counts.get(key, 0) + 1

    means = {key: total / counts[key] for key, total in totals.items()}
    high = max(float(matrix.max()) for matrix in means.values())
    norm = Normalize(vmin=0, vmax=high if high > 0 else 1)
    colors = LinearSegmentedColormap.from_list('relevance', ['#2166ac', '#f7f7f7', '#b2182b'])
    fig = Figure(figsize=(8.2, 13), layout='constrained')
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(EMOTIONS), len(TEST), squeeze=False)
    titles = {'both': 'Audiovisual', 'video_noise': 'Video only', 'audio_neutral': 'Audio only'}
    extent = (0, template.shape[1], template.shape[0], 0)
    for label, emotion in enumerate(EMOTIONS):
        for column, condition in enumerate(TEST):
            ax = axes[label, column]
            ax.imshow(template, extent=extent)
            key = (label, condition)
            if key in means:
                im = ax.imshow(means[key], cmap=colors, norm=norm, alpha=.45,
                               interpolation='bilinear', extent=extent)
            else:
                ax.text(.5, .5, 'No trials', transform=ax.transAxes, ha='center',
                        va='center', bbox={'facecolor': 'white', 'alpha': .8})
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if label == 0:
                ax.set_title(titles[condition])
            if column == 0:
                ax.set_ylabel(emotion.capitalize())
    fig.suptitle(f'{run_folder.name} | Mean video relevance\nActor 01 neutral template · {len(folders)} blocks')
    fig.colorbar(im, ax=axes, shrink=.7, label='Mean normalized positive loss relevance')
    fig.savefig(run_folder / 'average_heatmaps.png', dpi=180)
    save_json(run_folder / 'average_heatmaps.json', {
        'template_video': str(template_path), 'template_frame': 0, 'blocks': len(folders),
        'averaging': 'Mean over tubelets and explained trials, grouped by true emotion and condition',
        'scale': 'Each trial uses a common maximum across the three conditions; all panels share a color scale.',
        'trial_counts': {emotion: {condition: counts.get((label, condition), 0) for condition in TEST}
                         for label, emotion in enumerate(EMOTIONS)},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path, help='Existing training experiment directory.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fold', nargs='+', type=int)
    parser.add_argument('--block', nargs='+', type=int)
    parser.add_argument('--batch-size', type=int, help='Retest batch size. Defaults to the training value.')
    parser.add_argument('--explain-limit', type=int, default=0, help='Trials per condition to explain. Zero explains all. Testing always covers every trial.')
    parser.add_argument('--summarize-only', action='store_true',
                        help='Build the run summary from existing heatmap arrays without retesting.')
    options = parser.parse_args()
    if options.explain_limit < 0 or (options.batch_size is not None and options.batch_size < 1):
        parser.error('Use a nonnegative explanation limit and a positive batch size.')
    settings = json.loads((options.output_dir / 'experiment.json').read_text())
    units = list(experiment_units(options.output_dir, settings))
    for selection, available in [(options.fold, {f for f, b, p in units}),
                                 (options.block, {b for f, b, p in units})]:
        if selection and not set(selection) <= available:
            parser.error(f'Selected IDs {selection} outside available IDs {sorted(available)}.')
    folders = [folder for fold, block, folder in units
               if (options.fold is None or fold in options.fold)
               and (options.block is None or block in options.block)]
    if not folders:
        parser.error('No training blocks selected.')
    template, template_path = neutral_template(settings['video_root'])
    if not options.summarize_only:
        for folder in folders:
            print(f'Retesting {folder}', flush=True)
            retest(folder, options.device, options.explain_limit, options.batch_size)
    summarize_heatmaps(options.output_dir, folders, template, template_path)
    print(f'Saved {options.output_dir / "average_heatmaps.png"}', flush=True)


if __name__ == '__main__':
    main()
