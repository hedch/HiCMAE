# Six-emotion RAVDESS interaction experiment

## Current split: mixed actors, no validation

Each emotion is shuffled independently with `--seed` (default 42). The first 128
trials train and the next 64 test. Actors are mixed across the sets, and no
emotional trial can appear in both. This evaluates new trials with familiar actors,
rather than holding out actor identities.

| Emotion | Training | Testing |
| --- | ---: | ---: |
| Happy | 128 | 64 |
| Sad | 128 | 64 |
| Angry | 128 | 64 |
| Fearful | 128 | 64 |
| Surprised | 128 | 64 |
| Disgusted | 128 | 64 |
| Total | 768 | 384 |

No validation set is created or evaluated. A class with fewer than 192 unique
trials produces an error instead of changing the requested counts. Extra trials
beyond 192 per class are left unused and reported. With complete RAVDESS speech
data, all 1152 selected emotion trials are retained. Neutral and calm are excluded
from classification targets; neutral clips supply visual input controls in training and testing.

`train.csv`, `test.csv`, and `split_report.json` record selections, counts, seed,
actor overlap and any unused trials. Existing runs are not changed: use a NEW
output directory and retrain for this split. The old `--val-actors` and
`--test-actors` arguments have been removed.

## Model

The cross-modal fusion encoder is absent. Video and audio encoder outputs are
mean-pooled over tokens and combined across layers with learned softmax weights.
The B x D pooled vectors feed independent bias-free D-to-6 linear maps:

```
logits = wv(ov) + wa(oa) + wf(ov * oa)
```

The product is elementwise. Both encoders train with a shuffled mixture of paired emotional audio/video,
emotional video with white noise, and emotional audio with a static neutral face,
using cross entropy on raw logits without reconstruction or contrastive loss.
Class labels 0..5 are happy, sad, angry, fearful, surprised, disgusted.
RAVDESS codes are 03, 04, 05, 06, 08, 07 in that order.

Defaults match the released base checkpoint: D=512, 10 layers per encoder,
16 video frames at 160 x 160, audio filterbanks of 256 x 128. Use `--embed-dim 384`
and a matching `--depth` for another checkpoint. Encoder names/shapes must match;
fusion, reconstruction decoder and old classifier weights are ignored.

## Data and dependencies

Use the upstream Python/PyTorch environment with torchaudio, timm 0.4.12, einops,
decord, NumPy, scikit-learn and Pillow. Analysis also uses Matplotlib.
Download the speech videos and WAVs from https://zenodo.org/records/1188976.
Keep neutral clips available. Flat folders and actor subfolders both work.

```
videos/01-01-05-01-01-01-01.mp4
audio/01-01-05-01-01-01-01.wav
```

MP4 prefixes 01 and 02 are accepted; prefix 02 is preferred if both exist.
WAV prefixes 03, 01, and 02 are accepted in that priority order. Pairing uses the
six non-modality filename fields. Extensions are case-insensitive. All discovered
speech videos currently require matching WAVs, including neutral clips. Song is
excluded. Preserve original filenames and a shared video/audio time origin.

Audio is resampled to 16 kHz. Face-cropped MP4s retaining full duration/frame rate
are preferable for the face-pretrained model; raw MP4s work but the center square
crop is not a face detector. Extracted image folders are not supported.

A synchronized 2.56-second window is random during training and centered in
testing. Video is sampled into 16 frames; short audio is zero-padded. Video uses
ImageNet normalization and filterbanks use upstream AudioSet normalization.

## Run in a new output folder

Preparation check (without ML dependencies):

```bash
python run_ravdess_interaction.py --video-root /path/to/videos --audio-root /path/to/audio --output-dir runs/mixed_check --prepare-only
```

Fine-tune and run all tests/explanations:

```bash
python run_ravdess_interaction.py --video-root /path/to/videos --audio-root /path/to/audio --checkpoint /path/to/checkpoint-99.pth --output-dir runs/ravdess_mixed --epochs 50 --batch-size 4 --device cuda
```

Use a new/empty output directory for every invocation. Output directories are
created automatically. This single-device script uses full precision and zero
loader workers for Windows support. Reduce batch size for GPU memory limits.
Use `--device cpu` for CPU execution. No resume or DDP is implemented.

There is no validation-based or test-based checkpoint selection. The final
requested epoch is saved as `final.pth` and used for final metrics/explanations.
The new runner does not create `best.pth` or `val.csv`.

## Three test conditions and saved metrics

| Condition | Video | Audio |
| --- | --- | --- |
| both | Original emotional clip | Original speech |
| video_noise | Original emotional clip | Seeded Gaussian white noise |
| audio_neutral | Static neutral face repeated 16 times | Original speech |

The same 384 test trials, labels and ordering are used in each condition. All
three terms in the classifier stay active. White noise is generated in waveform
space before audio preprocessing; `--noise-rms` defaults to 0.05, fixed across
trials, with seeds saved per trial. Neutral clips match the actor, statement and
repetition of the emotional trial. The midpoint frame is repeated. These controls
are used as inputs during training and testing; neutral is never an emotion target.

`epoch_metrics.csv` and `history.jsonl` record training and all three test conditions
each epoch: loss, accuracy, correct/total, UAR, macro F1, learning rate and (in JSON)
confusion matrices. `epoch_predictions/` saves trial predictions, correctness,
loss and six probabilities. Training values are online predictions on random crops
with changing within-epoch weights. Test values use fixed inputs and epoch-end
weights. Tests are monitoring only; `--skip-epoch-tests` delays testing until the end.

Final results use `final.pth`: `test_metrics.json` includes `checkpoint_epoch` and
`checkpoint_selection=final_epoch`; per-condition JSON/CSV files save metrics and
predictions. `config.json` records the run options.

## Explanations and overlays

Video explanations run after training, using per-trial true-label cross-entropy
loss. The Grad-ECLIP-inspired pooled-query adaptation captures native per-head
attention outputs without replacing attention. Each query's loss gradient weights
source value channels with min-max cosine query-key similarity. Contributions sum
across queries, heads and video layers; learned pooling weights are accounted for
by backpropagation, not multiplied again. This is a local attribution heuristic,
not an exact or causal loss decomposition. Constant similarities yield zero weights.
Hooked and ordinary logits are checked for equality. Weights stay fixed.

`explanations/{condition}/{trial}/` contains:

- `frame_XX_overlay.png`: translucent jet-style positive relevance over actual input.
- `frame_XX_masked.png`: actual input multiplied by its positive mask.
- `heatmaps.npz`: signed per-layer/native maps, positive/normalized maps and display masks.
- `metadata.json`: labels, logits, loss, correctness, input controls, seeds and scale.

Positive relevance means locally increasing the classification loss. Native maps
are 8 x 10 x 10: each two-frame tubelet mask is spatially enlarged and repeated
for those frames. Audio-only images show the actual neutral control face. The
three conditions share a per-trial positive display maximum; scales are not shared
across trials. Zero positive relevance leaves overlay images unchanged and produces
black multiplicative masked images. Masked images are not reclassified.

Default explains all trials. `--explain-limit 2` exports only two trials per condition;
`--skip-explanations` disables exports. To re-export without retraining:

```bash
python video_loss_explanations.py --output-dir /path/to/existing/run --device cuda
```

This uses `final.pth`, falling back to `best.pth` for older runs, and needs only
train/test manifests. It rebuilds the explanation index and corresponding files
without changing training/test metrics.

## Analysis

```bash
python analyze_ravdess_results.py --run-dir /path/to/existing/run
```

All analysis is in this single script; it needs NumPy and Matplotlib, not PyTorch.
Outputs default to RUN_DIR/analysis (`--save-dir` overrides). New runs get a
training-only accuracy plot and three-condition test accuracy curves. Older runs
with validation still get training/validation curves. Bars by true emotion use
final predictions from the selected checkpoint (final epoch for new runs).

Figures are PNG/PDF. Numeric summaries are CSV/JSON. Average heatmaps divide each
positive native map by its saved shared per-trial scale, average over tubelets,
then average trials equally. All 18 groups use a common color range. Raw signed
and positive means and tubelet means are also saved as NPZ. These are model-input
coordinates, not anatomically registered facial maps. All exported trials contribute,
including incorrect predictions. Counts/warnings identify partial exports; missing
groups are marked instead of invented. Export all explanations before global analysis.

## Checks

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Data/metadata tests and syntax checks run in the editing environment. The preparation
CLI was checked against a synthetic complete 1440-trial dataset: 128/64 per target
emotion, no validation, no overlapping trials, all actors mixed, controls matched.
Analysis was checked with synthetic known results. Full training/attribution checks
require PyTorch and real data, unavailable in the editing environment.

## Modality contributions recorded during retraining

The runner now calls the model with `return_terms=True` and records the actual
feature-dependent video, audio, and interaction logits for every class. These are
saved in both epoch prediction CSVs and final test prediction CSVs as
`video_logit_{emotion}`, `audio_logit_{emotion}`, `interaction_logit_{emotion}`,
and total `logit_{emotion}`. The runner checks the terms sum to the total. No
additional forward pass or inference is required for the analysis. Final weight
matrices are also saved in `classifier_weights.npz` (wv, wa, wf and class order),
but weight magnitudes alone are not treated as modality reliance.

The same single `analyze_ravdess_results.py` script automatically adds a
`modality_contributions/` output folder containing:

- `true_logit_contributions.png/.pdf`: mean contributions to the true emotion's score.
- `margin_contributions.png/.pdf`: signed contributions to the true-versus-competitor margin.
- `absolute_margin_shares.png/.pdf`: mean fractional shares of absolute margin magnitude.
- `term_removal_accuracy_drop.png/.pdf`: accuracy change when one score term is removed.
- `trial_contributions.csv`, `contributions_by_emotion.csv`, and `term_ablation_accuracy.csv`.

All summaries use final-epoch test outputs and group by true emotion, separately
for all three testing conditions. Every trial is included, whether correctly
classified or not. The competing emotion is chosen once per trial as the highest
wrong-class score under the original audio+video condition; it stays fixed across
branches and conditions. This makes each three-term margin decomposition exact
and comparable across conditions. Positive margins support the true class over
that competitor; negative margins support the competitor. Means may cancel opposing
contributions, so mean absolute magnitudes and positive fractions are saved too.
Absolute shares ignore direction and are not probabilities of reliance. All-zero
margins receive zero shares and are flagged at trial level.

Term ablations recompute predictions by omitting selected recorded score terms,
without changing features or retraining. The CSV also includes full scores and
video-only, audio-only, and interaction-only score predictions. Dropping the video
term still leaves visual information in the interaction; it is not a complete
video-modality removal. Likewise for audio. A positive accuracy drop means omission
hurt performance; negative values mean it improved performance. These describe
the trained model's behavior, not human emotion processing or causal attribution
to individual input pixels. The existing three-condition accuracy plots remain
the input-replacement comparison.

No additional analysis command is needed. Older CSVs without the new columns still
produce the existing plots and a clear warning that contribution analyses were
skipped. Use a new output directory for retraining to obtain the contribution data.


## Expanded three-condition training

After the original 128/64-per-emotion split, every TRAINING trial is expanded into
three examples: `both`, `video_noise`, and `audio_neutral`. Targets remain the
original emotional label. The loader uses `shuffle=True` over the full expanded
list each epoch; training is not grouped by condition. Every epoch includes:

| Condition | Per emotion | All six emotions |
| --- | ---: | ---: |
| Original audio + video | 128 | 768 |
| Emotional video + white noise | 128 | 768 |
| Emotional audio + static neutral face | 128 | 768 |
| Total presented training examples | 384 | 2304 |

These are 768 unique emotional training trials presented under three conditions,
not 2304 independent original recordings. Test membership remains 64 unique trials
per emotion, 384 total, evaluated under each condition. Expansion occurs only AFTER
splitting and never uses a held-out emotional trial. Static neutral references match
actor, statement and repetition in both sets. Since actors are mixed across sets,
neutral control references can be shared across sets; emotional trials remain disjoint.

`train.csv` retains the 768 unique training trials, with neutral references and noise
base seeds. `train_expanded.csv` records all 2304 condition-labeled examples. The
split report records both unique-trial counts and expanded-example counts.
Training white noise uses base_seed + zero_based_epoch * 1000000, giving fresh but
reproducible noise each epoch, with train/test seed ranges kept separate. Test noise
stays fixed for comparison. The actual seed and condition are saved in prediction
CSVs. No new command-line options are needed.

Aggregate training metrics summarize all 2304 presentations. Additional `train_both`,
`train_video_noise`, and `train_audio_neutral` records in epoch metrics/predictions
summarize each condition separately. The analysis script automatically adds
`training_accuracy_conditions.png/.pdf`. Modality-contribution recording, final-epoch
checkpoint selection, test conditions and explanation export are preserved.

With the same batch size, each epoch now has approximately three times as many
optimizer steps. Use a NEW output directory and retrain; old completed runs are
not modified. Preparation/expansion tests verify exact counts, neutral references
and no overlap with held-out emotional trials. Full PyTorch execution remains
unverified in the editing environment.
