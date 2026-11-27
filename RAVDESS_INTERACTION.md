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
logits = wv(ov) + wf(ov * oa)
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

The same 384 test trials, labels and ordering are used in each condition. Both
terms in the classifier stay active; audio enters through the interaction only. White noise is generated in waveform
space before audio preprocessing; `--noise-rms` defaults to 0.05, fixed across
trials, with seeds saved per trial. Neutral clips match the actor, statement and
repetition of the emotional trial. For new runs, frame 0 (the first decoded image)
is repeated across all 16 frames. These controls
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

## Current two-term classifier and modality contributions

The current model REMOVES the independent `wa(oa)` term. New runs have only
`wv(ov) + wf(ov * oa)`: a direct video term and a joint audio-video interaction.
There is no trainable audio-only output head. The audio encoder remains trainable
through the interaction. Training/test input conditions remain the same, including
emotional audio with a neutral face; in that condition audio changes the
interaction with the neutral-face representation. The independent video term
can provide a constant offset for a fixed neutral input.

New run configuration, checkpoint summaries and weight archives identify
`logit_terms = [video, interaction]`. Prediction CSVs record total `logit_{emotion}`,
`video_logit_{emotion}` and `interaction_logit_{emotion}` for all six emotions.
The runner checks their sum reconstructs the total. `classifier_weights.npz`
contains wv and wf, with no wa. Fine-tuning from the original pretrained checkpoint
still loads encoder weights and initializes the two new output heads.

The same analysis script automatically detects two-term versus older three-term
CSVs and creates `modality_contributions/` with four figures (PNG/PDF): raw
true-class logit contributions, signed true-versus-competitor margins, absolute
margin shares, and accuracy changes when a score term is removed. Trial and
emotion-level contributions and term-ablation accuracies are saved as CSVs.
For new runs the plots show ONLY video and interaction; no fictitious audio term
or audio-term ablation is reported. Older results retain their actual three terms.

The competitor is the highest-scoring wrong class under original audio+video,
chosen once per trial and fixed across branches and conditions. Contributions sum
exactly to the true-minus-competitor margin. Positive values support the true class
over that competitor; negative values support the competitor. Mean absolute values
and positive fractions accompany signed averages. Absolute shares ignore direction
and are not probabilities of reliance. Zero-magnitude margins get zero shares and
are flagged. All test trials, correct and incorrect, contribute to the summaries.

Term removal omits recorded score terms without retraining or replacing inputs.
The new two-term model also reports video-term-only and interaction-term-only scores.
Dropping the direct video term still leaves video information in the interaction.
Dropping the interaction removes the model's only audio-dependent path. Positive
accuracy drop means omission hurt performance; negative drop means it helped.
These describe model behavior, not human modality reliance or causal pixel effects.

Use a NEW output directory and retrain for the new architecture. No new parameters
are needed. Existing checkpoints/results are not rewritten. Explanation-only reruns
inspect old saved checkpoints and reconstruct their original three-term head when
needed, so old runs are not silently interpreted as the new two-term model.

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

## Alternate original-fusion experiment

`run_ravdess_fusion.py` is a separate experiment restoring the original video,
audio and cross-modal fusion encoders. Use `--heads` to select any combination of
`visual`, `audio`, `fusion_video` and `fusion_audio`; their learned projections are
summed as logits (all four by default). It uses this same split,
three-condition training mix, heatmaps and analysis workflow. See
[RAVDESS_FUSION.md](RAVDESS_FUSION.md) for checkpoint requirements and run commands.
The existing interaction-only script keeps its current two-term architecture.

## Select interaction-model output terms

`run_ravdess_interaction.py` accepts `--terms` (alias `--heads`) followed by one
or more of `visual`, `audio`, and `interaction`. `video` aliases `visual`.
Omitting this option preserves the previous default: visual plus interaction.

Append one of these examples to your usual data/checkpoint/output arguments:

```text
--terms visual audio interaction
--terms visual interaction
--terms audio interaction
--terms visual audio
--terms interaction
--terms audio
--terms visual
```

The logits sum exactly the selected terms:

```
visual:       Wv(ov)
audio:        Wa(oa)
interaction:  Wf(ov * oa)    # elementwise feature product
```

Each W is a learned, bias-free D-to-6 matrix, with no additional scalar mixture
weights. Only selected classifier heads are created. The interaction term uses
both encoders even when the visual/audio standalone terms are omitted. A direct
visual-only selection runs only the video encoder; an audio-only selection runs
only the audio encoder. Duplicate aliases and empty selections are rejected.

Use a new output directory and retrain when changing terms. The configuration,
prediction contribution columns and classifier_weights.npz record the selection.
The same three-condition training/testing protocol, data split, optimizer,
per-epoch results and final-epoch checkpoint rule apply.

Use the updated video_loss_explanations.py and analyze_ravdess_results.py alongside
the runner. They support every term subset and older checkpoints/results. With
only the audio term, video heatmaps are zero because the loss does not depend on
video; overlays show the original input frames. A single-term run has no meaningful
term-removal classifier, so analysis does not assign an accuracy to removing its
only term. The fusion runner's separate four-head selection remains available.

## Training-only auxiliary supervision

New interaction runs enable two separate auxiliary classifiers by default:

```text
--aux-video-weight 0.5 --aux-audio-weight 0.5
```

For the interaction-only experiment, append this to your existing command:

```text
--terms interaction --aux-video-weight 0.5 --aux-audio-weight 0.5
```

The main logits still contain exactly the selected `--terms`. Auxiliary classifiers
are independent bias-free D-to-6 projections from the hierarchical video/audio
encoder features. They are training-only and never contribute to predictions,
score-removal analysis or the video-explanation target.

For each shuffled training minibatch:

```
objective = mean(main CE over every presentation)
          + aux_video_weight * mean(auxiliary visual CE on real AV presentations)
          + aux_audio_weight * mean(auxiliary audio CE on real AV presentations)
```

Only rows with `condition=both` receive auxiliary supervision. White-noise audio
and neutral-face video are not taught to predict the target emotion through an
auxiliary classifier. A batch with no real AV rows uses main CE alone. Each
auxiliary loss is averaged over eligible rows in that batch, not all batch rows.
The auxiliary CE gradients update both the auxiliary classifier and its encoder;
there is no feature detachment. This encourages discriminative representations
in each modality but does not guarantee equal modality reliance or AV>A=V accuracy.

Set either weight to zero to disable its auxiliary classifier, or both to zero
for the previous training objective. Weights must be finite and nonnegative.
Existing trained checkpoints are unchanged. New runs require a new output folder
and retraining. Original media splits, shuffled three-condition presentations,
test conditions, optimizer and final-epoch checkpoint selection stay the same.
The original-fusion runner does not enable these auxiliary losses.

Existing `loss` fields and curves continue to describe the main classifier CE,
so they remain comparable across training/testing conditions. Additional outputs:

- `auxiliary_metrics.csv`: epoch, main CE, optimized objective, and each auxiliary
  classifier's CE, accuracy and eligible trial count. There are 768 eligible AV
  presentations per full epoch (128 per emotion) for each enabled auxiliary head.
- `history.jsonl`: the training record also contains `objective_loss` and an
  `auxiliary` object with loss, accuracy, correct count, total and weight.
- `classifier_weights.npz`: `aux_video` and/or `aux_audio` contain the auxiliary
  projection matrices, separate from the selected main matrices.
- `final.pth`: includes auxiliary parameters and their configured weights, allowing
  strict checkpoint reconstruction. The updated explanation runner supports both
  these checkpoints and older runs without auxiliary classifiers.

The logged `objective_loss` is the sample-weighted average of the actual minibatch
objectives. Epoch auxiliary losses are averaged over eligible AV presentations;
because minibatches have different AV counts, recombining the epoch component
means need not reproduce the logged optimized objective exactly.

New runs record `neutral_frame_policy=first` in config.json: the shared loader
decodes only frame 0 of the matched neutral sample and repeats that image 16 times
for audio-neutral training, testing and explanations. Original emotional audio
is retained. Configurations from older runs without this setting retain their
midpoint-frame control when regenerating heatmaps. Use a new output folder and
retrain to adopt the first-frame control. The shared fusion and DBM runners use
the same setting.
