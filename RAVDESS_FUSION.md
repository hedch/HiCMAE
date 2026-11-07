# Selectable output heads with original HiCMAE fusion

Use `run_ravdess_fusion.py`. Download the updated repository because this runner
reuses the shared training, explanation and analysis files.

## Select heads

Pass one or more names after `--heads`. By default, all four are selected.

| Parameter | Feature projected to six emotion logits |
| --- | --- |
| `visual` (alias: `video`) | Hierarchical pooling of video encoder layers |
| `audio` | Hierarchical pooling of audio encoder layers |
| `fusion_video` | Mean of the video-stream tokens after the original fusion encoder |
| `fusion_audio` | Mean of the audio-stream tokens after the original fusion encoder |

Each selected head has its own learned, bias-free D-to-6 projection matrix W.
The final logits are the sum of the selected projected features:

```
logits = sum(W_head(feature_head) for head in selected_heads)
```

These matrices supply the weights, as in the previous experiment; there are no
additional scalar mixture coefficients. The two fusion heads use the two separate
streams of the original bidirectional cross-modal fusion encoder. Both fusion
streams can depend on both inputs. There is no Hadamard interaction term.

Only selected classifier heads are created. Encoder paths required by those heads
run and receive gradients; unused paths do not run. At least one head is required,
and duplicate names (including selecting both `visual` and `video`) are rejected.

## Run

Select this script in PyCharm, using the same environment and data parameters as
before. Use a new output directory and retrain when changing the head selection.

```bash
python run_ravdess_fusion.py --video-root /path/to/videos --audio-root /path/to/audio --checkpoint /path/to/checkpoint-99.pth --output-dir runs/four_heads --epochs 50 --batch-size 4 --heads visual audio fusion_video fusion_audio
```

Example values to append to your usual data/checkpoint/output arguments:

```text
--heads visual fusion_video
--heads audio
--heads fusion_video fusion_audio
```

Omit `--heads` to select all four. `visual` is saved internally as `video` in the
configuration, checkpoints and prediction columns. Selection order does not change
the sum.

For a file/split check without ML dependencies:

```bash
python run_ravdess_fusion.py --video-root /path/to/videos --audio-root /path/to/audio --output-dir runs/fusion_prepare --heads visual fusion_audio --prepare-only
```

Use the original HiCMAE pretraining checkpoint including all three encoders' weights.
The loader checks names/shapes for video, audio and fusion encoders, discards old
classifiers/decoders, and initializes the selected classifiers. Match `--depth`,
`--embed-dim` and `--fusion-depth` to the checkpoint. Defaults are 10 layers per
unimodal encoder, 2 fusion layers, and dimension 512 with 8 attention heads.

The same mixed-actor split has 128 training and 64 test trials per emotion: 768
base training trials and 384 held-out test trials, with no validation set. Each
training trial appears in all three conditions (both inputs, video with white
noise, audio with a static neutral face), shuffled across 2304 presentations per
epoch. Training noise changes reproducibly each epoch; test noise stays fixed.

## Outputs and analysis

The final epoch checkpoint is `final.pth`. Configuration and checkpoint metadata
record the head selection. Prediction CSVs contain each selected head's class
logits and their total. The weights archive stores each selected head's matrix
under its canonical name. Per-epoch metrics, trial correctness, manifests and
three-condition test outputs retain their existing format.

```bash
python video_loss_explanations.py --output-dir runs/four_heads --device cuda
python analyze_ravdess_results.py --run-dir runs/four_heads
```

The shared analysis still produces accuracy curves, emotion bars and global
average video heatmaps. Contribution, term-only and score-removal analyses adapt
to the selected heads. With a single head, removing it leaves no classifier;
the removal figure explains this instead of reporting an arbitrary zero-logit
prediction accuracy. Removing one fusion head is an output-term removal, not an
input-modality removal.

Video explanations retain the same loss target, pooled-query algorithm,
aggregation and colorful overlays. Hooks capture video-encoder self-attention;
gradients include whichever selected heads depend on video. With only `audio`,
the loss has no video dependency: exported video heatmaps are zero and overlays
show the unchanged frames. The three test input conditions are still evaluated.

Explanation-only loading automatically reconstructs the head selection from a
checkpoint. Previous combined-fusion checkpoints (video plus a single head over
concatenated fusion streams) and older interaction runs remain supported.

The existing `run_ravdess_interaction.py` architecture remains available. Other
data and explanation settings in [RAVDESS_INTERACTION.md](RAVDESS_INTERACTION.md)
also apply.

## Validation

Tests cover all 15 nonempty head combinations in numerical result analysis and
optional PyTorch forward/gradient checks, as well as aliases, invalid selections,
legacy results and single-head analysis. Full training requires the real media,
pretrained checkpoint and PyTorch environment.
