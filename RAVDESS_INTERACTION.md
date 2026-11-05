# Six-emotion RAVDESS interaction experiment

Fine-tune both HiCMAE encoders using paired audio/video. Evaluate the SAME selected
test trials with the SAME best-validation checkpoint in three conditions:

| Condition | Video | Audio |
| --- | --- | --- |
| `both` | Original emotional clip | Original speech |
| `video_noise` | Original emotional clip | Gaussian white noise |
| `audio_neutral` | Static neutral face, repeated for 16 frames | Original speech |

The model still receives both tensors in every condition, and all three terms in
its logit formula remain active. These are stimulus-replacement conditions, not
branch disabling: neutral-face and noise embeddings can contribute to the logits.

## Model and labels

The cross-modal fusion encoder is absent. Each encoder's tokens are mean-pooled
and layers combined using learned softmax weights, as in HiCMAE. The B x D output
vectors feed three unconstrained, bias-free learned D-to-6 linear maps:

```
logits = wv(ov) + wa(oa) + wf(ov * oa)
```

`*` is elementwise multiplication. The weights are matrices, not scalar mixture
weights. Cross entropy receives raw logits. Training and validation always use
original paired emotional audio/video; no noise, neutral substitutions,
reconstruction loss, or contrastive loss are used during fine-tuning.

Class order (labels 0 through 5): **happy, sad, angry, fearful, surprised,
disgusted**. RAVDESS codes map 03->0, 04->1, 05->2, 06->3, 08->4, 07->5.
Neutral and calm are excluded from all classification targets. Neutral speech
videos are retained solely as test control stimuli.

Defaults match the released base checkpoint: embedding dimension 512, 10 layers
per encoder, 16 video frames at 160 x 160, 256 x 128 audio filterbanks. Other
checkpoints can use `--embed-dim 384` and `--depth`. All encoder parameter names
and shapes must match. Fusion, decoder, and old head weights are discarded.

## Data

Use the repository's existing Python 3.8 / PyTorch / torchaudio / timm 0.4.12 /
einops / decord / NumPy / scikit-learn / Pillow environment.

Download RAVDESS speech videos and WAVs from https://zenodo.org/records/1188976.
Keep neutral speech videos available for the test control. Extract the archives
and pass the video and audio parent directories; subdirectories are searched.

Example paired filenames:

```
videos/Actor_01/02-01-05-01-01-01-01.mp4
audio/Actor_01/03-01-05-01-01-01-01.wav
```

Both video-only (02) and full-AV (01) MP4 modalities are accepted; video-only is
preferred if both exist. WAV files with prefixes 03 (official audio-only), 01 (full-AV exports), or 02 (video-only-named exports) are paired using the remaining six filename fields. If multiple audio versions exist, preference is 03, then 01, then 02. File extensions are case-insensitive. Identically named 01-prefixed MP4/WAV pairs in flat folders are supported. Song
files are excluded. All discovered speech videos, including neutral controls,
currently require matching WAVs. Audio is resampled to 16 kHz internally.

Face-cropped MP4s retaining the original filename, full duration, and frame rate
are preferable for transfer from the face-pretrained model. Raw MP4s are accepted,
but the center square crop is NOT a face detector. Extracted image folders are
not supported by this standalone script. Video and audio must share a time origin.

Emotional clips use synchronized 2.56-second windows: random in training and
centered in validation/testing. Video is sampled into 16 frames; short audio is
zero-padded. Video uses ImageNet normalization and filterbanks use the original
AudioSet normalization. A data-loading failure stops execution.

## Test controls

White noise is generated in the **waveform domain** at 16 kHz for 2.56 seconds,
then passed through the same filterbank and normalization as speech. Every trial
has a saved noise seed. The default waveform RMS is 0.05 on the conventional
floating audio scale; change it with `--noise-rms`. This is fixed across trials,
not matched to original emotional speech amplitude. No original speech enters the
noise replacement. This amplitude is an experimental choice, not a calibrated
sound-pressure level.

The neutral face is the midpoint frame of the neutral speech clip from the **same
test actor**, matched on statement and repetition. That single frame is repeated
across all 16 video frames. This avoids articulatory motion in the visual control.
It is a neutral-labeled frame, not a synthesized or independently validated face.
The neutral reference never comes from a training or validation actor. Missing
references produce an explicit error. `test.csv` saves the reference path and
noise seed used for each trial.

All conditions retain exactly the same emotional speech/video trial IDs, labels,
and test ordering. Only the specified input is replaced. The original six-way
label stays unchanged in all conditions.

## Balancing and split

A seeded shuffle assigns 16 actors to training, 4 to validation, and 4 to testing.
Choose custom groups by passing BOTH `--val-actors` and `--test-actors`; remaining
actors train. Within EACH split, all six target emotions are deterministically
undersampled without replacement to the smallest class count. On complete speech
data these six emotions are already equal, so all 1152 target trials are retained:
128 per class in training, 32 per class in validation, and 32 per class in testing.
The saved manifests/report record actual selections; incomplete data may reduce counts.

## Run

Check discovery, balancing, and neutral references without ML dependencies:

```bash
python run_ravdess_interaction.py --video-root /path/to/videos --audio-root /path/to/audio --output-dir runs/ravdess_check --prepare-only
```

Train and automatically run all three test conditions:

```bash
python run_ravdess_interaction.py --video-root /path/to/videos --audio-root /path/to/audio --checkpoint /path/to/checkpoint-99.pth --output-dir runs/ravdess_six --batch-size 4 --epochs 50 --lr 1e-4 --noise-rms 0.05
```

Use a fresh output directory every time, including after preparation. Only load
trusted checkpoint files. Default device is CUDA; `--device cpu` runs on CPU.
This is a single-device, full-precision script with zero loader workers for
Windows support. Multi-GPU/DDP, automatic download, and resume are not implemented.

## Saved epoch and final results

Every epoch saves training, validation, and all three test conditions to
`epoch_metrics.csv` and `history.jsonl`: mean cross-entropy loss, accuracy,
correct/total counts, UAR, macro F1, and learning rate. Confusion matrices are
included in JSON. Each split/condition also gets a per-trial CSV in
`epoch_predictions/`, containing trial ID, true/predicted label, correctness,
loss, and six class probabilities.

Training metrics are online values collected on the random training crops before
each optimizer update, across changing model weights. Validation/test values use
fixed centered inputs and the end-of-epoch weights in evaluation mode.

Checkpoint selection always uses paired-input validation UAR only. Epoch test
results are monitoring outputs and never used to choose the checkpoint. To keep
test evaluation until the end instead, pass `--skip-epoch-tests`.

After training, the best-validation checkpoint is loaded and all three test
conditions are evaluated again. `test_metrics.json` records the selected epoch
and all condition metrics. Each condition also gets `test_metrics_{condition}.json`
and `test_predictions_{condition}.csv`, including per-trial correctness and loss.
These final results can differ from the last epoch because the selected epoch
may be earlier. `best.pth`, configuration, split report, and manifests are saved.

## Loss-based video explanations and masked frames

Explanations run automatically AFTER training using the best-validation checkpoint.
Only video is attributed, separately for every test trial and all three conditions.
The target is each trial's true-label cross-entropy loss. Positive relevance means
local contribution toward increasing the loss; it is not evidence supporting the
correct emotion, nor a causal guarantee that masking changes the prediction.

This is a **Grad-ECLIP-inspired pooled-query adaptation**, based on
https://github.com/Cyang-Zhao/Grad-Eclip. It does not use CLIP or introduce a CLS
token. For every video transformer layer/head, the native attention output before
its projection is captured. The channel weight for query i is the gradient of
loss with respect to that query's per-head attention output. The source-token j
score sums gradient/value channel products multiplied by per-query min-max-scaled
cosine similarity between query i and key j. Scores sum across queries, heads,
and layers. No learned pooling weights are multiplied a second time. Constant
similarity rows receive zero spatial weights; normalization denominators are guarded.
Layer aggregation is an attribution heuristic, not an exact loss decomposition.

Hooks observe the native attention computation without changing head counts,
scaling, or model predictions. Every explanation checks hooked logits against
an ordinary forward pass. Model weights remain fixed; parameter gradients are
not accumulated. Explanations require autograd even though the model is in eval mode.

Output folders are:

- `explanations/both/{trial}/`
- `explanations/video_noise/{trial}/`
- `explanations/audio_neutral/{trial}/`

Each trial folder contains 16 `frame_XX_masked.png` images showing the ACTUAL
160 x 160 center-cropped model input multiplied by its positive relevance mask.
For audio-only tests these are the repeated static neutral face, not the replaced
emotional video. Native maps are 8 x 10 x 10: each tubelet covers two frames.
Masks are resized spatially and repeated for the two frames, without temporal
interpolation or a claim of independent frame-level resolution.

`heatmaps.npz` contains signed native scores, signed per-layer scores, positive
native scores, a per-condition normalized native map, and the shared-scale masks
actually used to save images. Masked images use the maximum positive native score
across all three conditions for THAT trial, preserving relative magnitudes within
that comparison. The scale is not shared across different trials. All-zero positive
scores produce black masked frames and zero masks, not NaNs.

`metadata.json` records logits, loss, correctness, target label, forward-equivalence
error, neutral reference, noise seed, and display scale. `explanations/index.jsonl`
indexes all exports. Masking is a visualization step only: these masked images are
not fed back through the classifier and do not alter saved test metrics.

For a quick export check, add `--explain-limit 2` to process only two test trials
in each condition. Default 0 explains all test trials. `--skip-explanations` skips
exports. Attribution runs one trial at a time and chunks query aggregation to
limit temporary memory; it still adds computation and image files for each trial.

## Verification

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Data preparation tests need only Python. Optional dependency-backed tests cover
head logits/gradients, seeded noise, chunked attribution against an explicit formula,
constant/zero-map stability, tubelet-to-frame mapping, native forward equality,
hook removal, saved image exports, and a miniature two-epoch run checking result
files for all three conditions. In the editing environment six data tests passed;
six dependency-backed tests were skipped because PyTorch is unavailable. Python
compilation passed. Real media decoding, full-model attribution, and fine-tuning
have not been executed here; run these checks in the project environment before
a full experiment.

## Retry explanation export without retraining

If training and final testing completed but explanation export was interrupted,
update `video_loss_explanations.py` and run it directly, pointing to the EXISTING
run output directory (containing config.json, best.pth, and the three manifests):

```bash
python video_loss_explanations.py --output-dir /path/to/existing/run --device cuda --explain-limit 2
```

Omit `--explain-limit 2` to export all test trials. This loads the best checkpoint
and saved trial selections, keeps training/test metrics untouched, and rebuilds
`explanations/index.jsonl` while overwriting the corresponding explanation files.
The original media paths in the manifests must still exist. In PyCharm, set this
module as the script path and put the options in Parameters. This mode does not
apply the training runner's empty-output-directory restriction.


## Colored overlays

Every explanation now additionally saves `frame_XX_overlay.png`: a jet-style color heatmap blended over the actual input frame. Maximum opacity is 0.45, multiplied by normalized positive relevance, so zero-relevance regions show the unchanged original image and the face remains visible at high relevance. The same per-trial scale across conditions is used. `frame_XX_masked.png` remains available as the multiplicative black-background visualization. Rerun the explanation-only command to add overlays to an existing trained run; no retraining is needed.

The current runner uses one seeded 16/4/4 actor split. It does not implement the original repository's cross-validation folds (the repository supplies six RAVDESS split folders). Results from this runner are single-split results, not fold averages.
