"""Analyze saved RAVDESS results without loading a model or rerunning training.

Usage:
  python analyze_ravdess_results.py --run-dir /path/to/trained_run
Dependencies: numpy, matplotlib (no PyTorch required).
Figures are saved as PNG and PDF; numeric summaries and mean heatmaps are saved.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

EMOTIONS = ['happy', 'sad', 'angry', 'fearful', 'surprised', 'disgusted']
CONDITIONS = ('both', 'video_noise', 'audio_neutral')
NAMES = {'both': 'Audio + video', 'video_noise': 'Video + white noise',
         'audio_neutral': 'Audio + neutral face'}
COLORS = {'both': '#2864B4', 'video_noise': '#D37B20', 'audio_neutral': '#23856B'}


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, path, dpi):
    fig.savefig(str(path) + '.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(str(path) + '.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def load_epochs(run):
    path = run / 'epoch_metrics.csv'
    if path.exists():
        rows = read_csv(path)
    else:
        rows = []
        with (run / 'history.jsonl').open() as f:
            for line in f:
                record = json.loads(line)
                for name in ('train', 'val'):
                    if isinstance(record.get(name), dict) and 'accuracy' in record[name]:
                        rows.append(dict(epoch=record['epoch'], split=name, **record[name]))
                for name, metrics in record.get('tests', {}).items():
                    rows.append(dict(epoch=record['epoch'], split='test_' + name, **metrics))
    groups, seen = defaultdict(list), set()
    for row in rows:
        name, epoch = row['split'], int(row['epoch'])
        accuracy = float(row['accuracy'])
        if (name, epoch) in seen:
            raise ValueError('Duplicate epoch result: %s, epoch %d' % (name, epoch))
        if not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError('Accuracy must be a finite proportion between zero and one')
        groups[name].append((epoch, accuracy))
        seen.add((name, epoch))
    for values in groups.values():
        values.sort()
    return groups


def plot_epoch_curves(groups, save, best_epoch, dpi, warnings, checkpoint_label="Best validation epoch"):
    specifications = [
        ('training_validation_accuracy', [('train', 'Training (online)', '#2864B4'),
                                           ('val', 'Validation', '#D37B20')],
         'Training and validation accuracy'),
        ('testing_accuracy_conditions', [('test_' + c, NAMES[c], COLORS[c]) for c in CONDITIONS],
         'Test accuracy across conditions')]
    if not groups.get('val'):
        specifications[0] = ('training_accuracy', [('train', 'Training (online)', '#2864B4')],
                             'Training accuracy — no validation split')
    for filename, curves, title in specifications:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        plotted = 0
        for key, label, color in curves:
            if not groups.get(key):
                warnings.append('No epoch accuracies for ' + key)
                continue
            epochs, accuracy = zip(*groups[key])
            ax.plot(epochs, np.asarray(accuracy) * 100, label=label, color=color,
                    linewidth=2, marker='o' if len(epochs) <= 10 else None, markersize=4)
            plotted += 1
        ax.axhline(100 / len(EMOTIONS), color='.65', linestyle=':', label='Chance (16.7%)')
        if best_epoch is not None:
            ax.axvline(best_epoch, color='.35', linestyle='--', linewidth=1,
                       label=checkpoint_label + ' (%d)' % best_epoch)
        if not plotted:
            ax.text(.5, .55, 'No per-epoch accuracy records available', ha='center', transform=ax.transAxes)
        ax.set(xlabel='Epoch', ylabel='Accuracy (%)', title=title, ylim=(0, 100))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=9, loc='best')
        fig.tight_layout()
        save_figure(fig, save / filename, dpi)


def analyze_predictions(run, save, dpi, checkpoint_description="best-validation checkpoint"):
    summaries, targets = [], {}
    for condition in CONDITIONS:
        rows = read_csv(run / ('test_predictions_' + condition + '.csv'))
        ids = [row['video'] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate test trial IDs in ' + condition)
        targets[condition] = {row['video']: int(row['label']) for row in rows}
        for row in rows:
            if not 0 <= int(row['label']) < len(EMOTIONS) or not 0 <= int(row['prediction']) < len(EMOTIONS):
                raise ValueError('Invalid class index in ' + condition)
        for label, emotion in enumerate(EMOTIONS):
            subset = [row for row in rows if int(row['label']) == label]
            if not subset:
                raise ValueError('Missing test emotion %s in %s' % (emotion, condition))
            correct = sum(int(row['prediction']) == label for row in subset)
            summaries.append(dict(condition=condition, label=label, emotion=emotion,
                                  correct=correct, total=len(subset), accuracy=correct / len(subset)))
    if any(targets[c] != targets['both'] for c in CONDITIONS):
        raise ValueError('Test conditions do not contain identical trial IDs and true labels')
    write_csv(save / 'testing_accuracy_by_emotion.csv', summaries,
              ['condition', 'label', 'emotion', 'correct', 'total', 'accuracy'])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x, width = np.arange(len(EMOTIONS)), .24
    for i, condition in enumerate(CONDITIONS):
        subset = [row for row in summaries if row['condition'] == condition]
        bars = ax.bar(x + (i-1)*width, [100*r['accuracy'] for r in subset], width,
                      color=COLORS[condition], label=NAMES[condition])
        for bar, row in zip(bars, subset):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    '%d/%d' % (row['correct'], row['total']), ha='center', va='bottom', fontsize=8)
    ax.axhline(100 / len(EMOTIONS), color='.65', linestyle=':', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in EMOTIONS])
    ax.set(ylabel='Accuracy within true emotion (%)', ylim=(0, 110),
           title='Test accuracy by emotion — ' + checkpoint_description)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=3, fontsize=9)
    fig.tight_layout()
    save_figure(fig, save / 'testing_accuracy_by_emotion', dpi)
    return targets


def aggregate_heatmaps(run, save, targets, dpi, warnings):
    """Average per-trial shared-scaled positive maps, equally over time and trials.

    Also save raw positive/signed means. Trial-level scales are shared between
    conditions, so a condition's weaker relevance is not independently stretched.
    Native maps are aligned model-input coordinates, not registered facial anatomy.
    """
    sums, counts, seen, shape = {}, defaultdict(int), set(), None
    for condition in CONDITIONS:
        folder = run / 'explanations' / condition
        for metadata_path in sorted(folder.glob('*/metadata.json')):
            metadata = json.loads(metadata_path.read_text())
            if metadata.get('condition') != condition:
                raise ValueError('Condition mismatch: ' + str(metadata_path))
            trial_id = metadata['trial']['video']
            label = int(metadata['label'])
            if targets[condition].get(trial_id) != label:
                raise ValueError('Heatmap trial not matched to final test labels: ' + trial_id)
            if (condition, trial_id) in seen:
                raise ValueError('Duplicate heatmaps for trial: ' + trial_id)
            seen.add((condition, trial_id))
            path = metadata_path.parent / 'heatmaps.npz'
            with np.load(path, allow_pickle=False) as arrays:
                signed = np.asarray(arrays['signed'], dtype=np.float64)
                positive = np.asarray(arrays['positive'], dtype=np.float64)
            if signed.ndim != 3 or signed.shape != positive.shape:
                raise ValueError('Expected T x H x W signed/positive maps: ' + str(path))
            if shape is None:
                shape = signed.shape
            if signed.shape != shape or not np.isfinite(signed).all() or not np.isfinite(positive).all():
                raise ValueError('Incompatible or nonfinite heatmap: ' + str(path))
            scale = float(metadata['shared_positive_scale'])
            if not np.isfinite(scale) or scale < 0:
                raise ValueError('Invalid shared heatmap scale: ' + str(path))
            scaled = np.clip(positive / max(scale, 1e-12), 0, 1)
            key = (condition, label)
            if key not in sums:
                sums[key] = dict(shared=np.zeros(shape), positive=np.zeros(shape), signed=np.zeros(shape))
            for name, value in [('shared', scaled), ('positive', positive), ('signed', signed)]:
                sums[key][name] += value
            counts[key] += 1
    heatmap_dir = save / 'average_heatmaps'
    heatmap_dir.mkdir(exist_ok=True)
    means = {key: {name: total / counts[key] for name, total in values.items()}
             for key, values in sums.items()}
    # One color range over every emotion/condition; no independent panel normalization.
    vmax = max([float(values['shared'].mean(0).max()) for values in means.values()] + [1e-12])
    report = []
    fig, axes = plt.subplots(len(EMOTIONS), len(CONDITIONS), figsize=(10, 16), squeeze=False,
                              constrained_layout=True)
    last_image = None
    for label, emotion in enumerate(EMOTIONS):
        for column, condition in enumerate(CONDITIONS):
            key, ax = (condition, label), axes[label, column]
            expected = sum(value == label for value in targets[condition].values())
            n = counts[key]
            report.append(dict(condition=condition, emotion=emotion, label=label,
                               heatmap_trials=n, test_trials=expected, complete=n == expected))
            if n != expected:
                warnings.append('%s / %s: heatmaps for %d of %d trials' % (condition, emotion, n, expected))
            title = '%s | n=%d/%d' % (emotion.capitalize(), n, expected)
            if key not in means:
                ax.text(.5, .5, 'No saved heatmaps', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=10)
                ax.axis('off')
                continue
            values = means[key]
            mask = values['shared'].mean(0)
            np.savez_compressed(heatmap_dir / (condition + '_' + emotion + '.npz'),
                                mean_mask=mask, mean_mask_by_tubelet=values['shared'],
                                mean_raw_positive=values['positive'].mean(0),
                                mean_raw_positive_by_tubelet=values['positive'],
                                mean_raw_signed=values['signed'].mean(0),
                                mean_raw_signed_by_tubelet=values['signed'], trial_count=n)
            last_image = ax.imshow(mask, cmap='jet', vmin=0, vmax=vmax, interpolation='bilinear')
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            single, single_ax = plt.subplots(figsize=(4, 4))
            im = single_ax.imshow(mask, cmap='jet', vmin=0, vmax=vmax, interpolation='bilinear')
            single_ax.set(title=NAMES[condition] + '\n' + title)
            single_ax.set_xticks([])
            single_ax.set_yticks([])
            single.colorbar(im, ax=single_ax, fraction=.046, pad=.04, label='Mean relative loss relevance')
            save_figure(single, heatmap_dir / (condition + '_' + emotion), dpi)
    for column, condition in enumerate(CONDITIONS):
        axes[0, column].set_title(NAMES[condition] + '\n' + axes[0, column].get_title(), fontsize=10)
    fig.suptitle('Average video loss-relevance masks\nEqual weighting of trials and tubelets; common color scale', fontsize=13)
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=.025, pad=.02,
                     label='Mean relative positive loss relevance')
    save_figure(fig, save / 'average_heatmaps_all_emotions_conditions', dpi)
    write_csv(save / 'heatmap_counts.csv', report,
              ['condition', 'emotion', 'label', 'heatmap_trials', 'test_trials', 'complete'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, help='Existing trained-run output folder')
    parser.add_argument('--save-dir', help='Default: RUN_DIR/analysis')
    parser.add_argument('--dpi', type=int, default=250)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error('--dpi must be positive')
    run = Path(args.run_dir).expanduser().resolve()
    save = Path(args.save_dir).expanduser().resolve() if args.save_dir else run / 'analysis'
    save.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    warnings = []
    summary_path = run / 'test_metrics.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if summary.get('emotions', EMOTIONS) != EMOTIONS:
        raise ValueError('Saved class order differs from expected six-emotion experiment')
    groups = load_epochs(run)
    final_epoch = summary.get('checkpoint_selection') == 'final_epoch'
    checkpoint_epoch = summary.get('checkpoint_epoch', summary.get('best_epoch'))
    description = 'final-epoch checkpoint' if final_epoch else 'best-validation checkpoint'
    plot_epoch_curves(groups, save, checkpoint_epoch, args.dpi, warnings,
                      'Final epoch' if final_epoch else 'Best validation epoch')
    targets = analyze_predictions(run, save, args.dpi, description)
    counts = aggregate_heatmaps(run, save, targets, args.dpi, warnings)
    report = dict(run_dir=str(run), checkpoint_epoch=checkpoint_epoch,
                  checkpoint_selection=summary.get('checkpoint_selection', 'best_validation'), emotions=EMOTIONS,
                  conditions=NAMES, warnings=warnings, heatmap_counts=counts,
                  heatmap_averaging='Positive native map / saved shared trial scale, averaged over tubelets then trials',
                  interpretation='Positive loss relevance; model-input coordinates without facial registration',
                  training_accuracy='Online training crops and changing within-epoch weights',
                  bar_plot=description + ' predictions, grouped by true emotion')
    (save / 'analysis_report.json').write_text(json.dumps(report, indent=2))
    print('Saved analysis to:', save)
    for warning in warnings:
        print('WARNING:', warning)
    print('Done. Training/model files were not modified.')


if __name__ == '__main__':
    main()
