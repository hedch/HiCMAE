"""Summarize trained folds. Error bars show one sample standard deviation across fold/block estimates."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_ravdess_actor_mlp import EMOTIONS, TEST, experiment_units


def summarize(frame, keys, values):
    grouped = frame.groupby(keys, sort=False)
    result = grouped[values].agg(['mean','std'])
    result.columns = ['_'.join(c) for c in result.columns]
    result['n_folds'] = grouped['fold'].nunique()
    result['n_models'] = grouped.size()
    return result.reset_index()


def heatmaps(units, folder):
    # Average trials within each model before averaging models. Use each trial's common
    # display scale to preserve differences among the three test conditions.
    sums, counts = {}, {}
    for fold, block, path in units:
        index = path/'explanations'/'index.jsonl'
        if not index.exists():
            continue
        for line in index.read_text().splitlines():
            record = json.loads(line)
            c, label = record['condition'], record['label']
            path = index.parent/c/Path(record['directory']).name/'heatmaps.npz'
            with np.load(path) as maps:
                m = maps['positive'].mean(axis=0)/max(record['shared_positive_scale'],1e-12)
            key = (fold,block,c,label)
            sums[key] = sums.get(key,0.) + m
            counts[key] = counts.get(key,0) + 1
    if not sums:
        return
    fig, axes = plt.subplots(len(TEST),len(EMOTIONS), figsize=(13,7), constrained_layout=True)
    coverage = []
    for i,c in enumerate(TEST):
        for j,e in enumerate(EMOTIONS):
            means = []
            for fold, block, path in units:
                key = (fold,block,c,j)
                coverage.append(dict(fold=fold,block=block,condition=c,emotion=e,trials=counts.get(key,0)))
                if key in sums:
                    means.append(sums[key]/counts[key])
            ax = axes[i,j]
            ax.set(xticks=[],yticks=[], title=e if i==0 else '', ylabel=c if j==0 else '')
            if means:
                mean = np.mean(means,axis=0)
                std = np.std(means,axis=0,ddof=1) if len(means)>1 else np.full_like(mean,np.nan)
                np.savez_compressed(folder/f'heatmap_{c}_{e}.npz',mean=mean,std=std,n_models=len(means))
                im=ax.imshow(mean,vmin=0,vmax=1,cmap='inferno')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Mean positive loss relevance')
    fig.savefig(folder/'mean_heatmaps.png',dpi=180)
    plt.close(fig)
    pd.DataFrame(coverage).to_csv(folder/'heatmap_counts.csv',index=False)


def analyze(root):
    root = Path(root)
    settings = json.loads((root/'experiment.json').read_text())
    epochs = settings['epochs']
    units = list(experiment_units(root, settings))
    folder = root/'analysis'
    folder.mkdir(exist_ok=True)
    metrics, predictions = [], []
    for fold, block, path in units:
        frame = pd.read_csv(path/'metrics.csv')
        if frame.epoch.max()!=epochs or not (path/'final.pth').exists():
            raise ValueError(f'Fold {fold}, block {block} is incomplete')
        expected = {(epoch, split) for epoch in range(1, epochs+1)
                    for split in ['train', 'train_audio_noise', 'train_video_noise', 'train_both',
                                  'test_both', 'test_video_noise', 'test_audio_neutral']}
        actual = set(zip(frame.epoch, frame.split))
        if actual != expected or frame.duplicated(['epoch', 'split']).any():
            raise ValueError(f'Incomplete or duplicate metrics in {path}')
        metrics.append(frame.assign(fold=fold, block=block))
        for c in TEST:
            final = pd.read_csv(path/f'epoch_{epochs:03d}'/f'test_{c}.csv')
            predictions.append(final.assign(fold=fold, block=block))
    metrics, predictions = pd.concat(metrics), pd.concat(predictions)
    metrics.to_csv(folder/'all_block_metrics.csv',index=False)
    predictions.to_csv(folder/'final_test_predictions.csv',index=False)
    avg = summarize(metrics,['epoch','split'],['accuracy','loss','uar','macro_f1'])
    avg.to_csv(folder/'epoch_mean_std.csv',index=False)
    final_metrics = avg[(avg.epoch==epochs) & avg.split.str.startswith('test_')]
    final_metrics.to_csv(folder/'final_mean_std.csv',index=False)
    print(final_metrics.to_string(index=False))
    for phase in ('train','test'):
        fig,ax=plt.subplots(figsize=(8,5), constrained_layout=True)
        for name, group in avg[avg.split.str.startswith(phase)].groupby('split'):
            g=group.sort_values('epoch')
            ax.errorbar(g.epoch,g.accuracy_mean,yerr=g.accuracy_std.fillna(0),capsize=2,label=name)
        ax.set(xlabel='Epoch',ylabel='Accuracy (block mean ± SD)',ylim=(0,1.03))
        ax.legend()
        fig.savefig(folder/(phase+'_accuracy.png'),dpi=180)
        plt.close(fig)
    scores=predictions.groupby(['fold','block','condition','label']).correct.agg(['mean','sum','count']).reset_index()
    scores=scores.rename(columns={'mean':'accuracy','sum':'correct','count':'total'})
    scores['emotion'] = scores.label.map(dict(enumerate(EMOTIONS)))
    scores.to_csv(folder/'per_emotion_by_block.csv',index=False)
    summary=summarize(scores,['condition','emotion'],['accuracy'])
    summary.to_csv(folder/'per_emotion_mean_std.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    x=np.arange(len(EMOTIONS))
    for i,c in enumerate(TEST):
        g=summary[summary.condition==c].set_index('emotion').loc[EMOTIONS]
        ax.bar(x+(i-1)*.25,g.accuracy_mean,width=.24,yerr=g.accuracy_std.fillna(0),capsize=3,label=c)
    ax.set(xticks=x,xticklabels=EMOTIONS,ylabel='Final test accuracy (block mean ± SD)',ylim=(0,1.05))
    ax.legend()
    fig.savefig(folder/'test_accuracy_by_emotion.png',dpi=180)
    plt.close(fig)
    heatmaps(units, folder)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    analyze(parser.parse_args().output_dir)

