# Video + cross-modal fusion experiment

This separate runner is `run_ravdess_fusion.py`. The interaction-only runner remains
available as `run_ravdess_interaction.py`. Download the updated repository because
the fusion runner reuses its data/training helpers and the shared analysis/explainer.

## Architecture

All three original HiCMAE encoders are used: video, audio, and cross-modal fusion.
The fusion encoder is the original `VisionTransformerEncoderForFusion`, with both
video and audio streams using cross-attention, self-attention, MLPs and normalization.
Default depths are 10 video layers, 10 audio layers and 2 fusion layers. Default
embedding dimension is 512 with 8 heads. There is no Hadamard interaction term and
no independent audio classifier.

The direct video feature ov is the learned hierarchical pooling of the video
encoder layers, as in the existing experiment. The fusion encoder receives the
last-layer video and audio tokens, as in the original model. Its outputs Fv and Fa
are token-mean-pooled and concatenated:

```
ov = hierarchical_pool(video_encoder_layers)     # B,D
Fv, Fa = fusion_encoder(last_video_tokens, last_audio_tokens)
of = cat(mean_tokens(Fv), mean_tokens(Fa))        # B,2D
logits = wv(ov) + wf(of)                         # B,6
```

wv is a learned, bias-free D-to-6 linear map; wf is a learned, bias-free 2D-to-6
linear map. Thus weighted summation means two learned projection matrices, not
scalar mixture weights. Both pooled fusion streams contribute, rather than only
the fused video stream. Loss gradients train all three encoders and both heads.

Use the original HiCMAE pretraining checkpoint including fusion weights. The loader
requires matching names and shapes for all three encoders. A checkpoint from the
interaction-only experiment is intentionally rejected because it lacks the fusion
encoder. Only reconstruction decoders and old classifiers are discarded. Set
`--fusion-depth` to match the checkpoint (released base default: 2).

## Run

The parameters and Conda environment are the same as the existing experiment.
Select `run_ravdess_fusion.py` as the script in PyCharm, with a NEW output folder:

```bash
python run_ravdess_fusion.py --video-root /path/to/videos --audio-root /path/to/audio --checkpoint /path/to/checkpoint-99.pth --output-dir runs/ravdess_fusion01 --epochs 50 --batch-size 4 --fusion-depth 2
```

For a file/split check without importing the ML dependencies:

```bash
python run_ravdess_fusion.py --video-root /path/to/videos --audio-root /path/to/audio --output-dir runs/fusion_prepare --prepare-only
```

The same 128/64 mixed-actor split per emotion is used: 768 base training trials,
384 held-out test trials, no validation. Every training trial expands into original
audio/video, video with white noise, and audio with static neutral face, shuffled
across all 2304 presentations each epoch. Training noise changes reproducibly by
epoch; test noise stays fixed. No test emotional trial enters training.

## Saved results, explanations and analysis

The final epoch is saved as `final.pth`, with `architecture=fusion` in run configuration
and test summary. Prediction CSVs store `video_logit_{emotion}` and
`fusion_logit_{emotion}` plus their total. Weight archives contain wv and wf.
Per-epoch condition metrics, accuracy, correctness, losses, manifests and final
test outputs keep the existing file structure.

Video explanations use the SAME pooled-query algorithm, layer aggregation,
cross-entropy target, positive mask scaling and image overlays. Hooks still capture
only the video encoder's self-attention layers. Gradients pass through the complete
new classifier and fusion encoder; fusion attention itself is not separately mapped.
The explanation-only entrypoint detects fusion checkpoints automatically:

```bash
python video_loss_explanations.py --output-dir runs/ravdess_fusion01 --device cuda
python analyze_ravdess_results.py --run-dir runs/ravdess_fusion01
```

The same analysis script produces all accuracy curves, emotion bars, global average
heatmaps and contribution plots. Branch labels are video and fusion. Score-removal
analyses include drop_video and drop_fusion. Removing the direct video head leaves
video information in the fusion branch. These are output-term analyses rather than
independent modality removals. Older interaction and three-term results remain supported.

All settings documented in RAVDESS_INTERACTION.md (noise RMS, explanation limits,
input naming, visual controls, saved maps, analysis coverage) also apply.

## Validation

Python compilation, synthetic preparation and numerical fusion-output analysis tests
are available without PyTorch. Optional tests check the summed-logit formula, audio
and fusion gradient flow and strict fusion-checkpoint loading using the original
fusion module. Full training and attribution cannot be verified in the editing
environment because its PyTorch dependencies and real media/checkpoints are absent.
