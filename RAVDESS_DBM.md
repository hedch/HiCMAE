# DBM fusion with adversarial condition confusion

Run `run_ravdess_dbm.py` as a separate experiment. Keep the updated repository's
shared runner, analysis and explanation files alongside it. Existing interaction
and original-fusion runners remain available.

## Structure

1. Original pretrained HiCMAE video and audio encoders, without its fusion encoder.
2. Learned hierarchical pooling of each encoder's layer outputs.
3. Separate feature heads: Linear(D, visible_dim), LayerNorm, tanh.
4. A two-hidden-layer undirected Gaussian-visible/Bernoulli-hidden DBM.
5. Six-emotion linear classifier from the inferred top hidden probabilities.
6. A separate three-condition classifier from the same top representation, reached
   through a gradient reversal operation.

The encoder feature heads supply embeddings, not six-class emotion logits, to the
DBM. There is no direct visual/audio summation or Hadamard interaction head.
The DBM has two visible groups, visual v and audio a, both connected to hidden
layer h1; h1 connects bidirectionally to h2. Its energy is:

```
E(v,a,h1,h2) = .5||v||² + .5||a||²
              - v Wv h1 - a Wa h1 - h1 W12 h2 - b1 h1 - b2 h2
```

Visible inputs are clamped; unit visible variance is fixed. Hidden inference starts
bottom-up and iterates these updates using the SAME W12 in both directions:

```
mean_h1 = sigmoid(v Wv + a Wa + mean_h2 W12.T + b1)
mean_h2 = sigmoid(mean_h1 W12 + b2)
```

Default sizes: visible=128 per modality, h1=256, h2=128, five refinement steps.
The model backpropagates through the unrolled mean-field updates. It is trained
**discriminatively** with emotion/adversarial losses, not generative maximum
likelihood: there is no RBM pretraining, Gibbs/negative phase, or contrastive-divergence objective. Fixed-iteration mean-field inference is approximate and is
not asserted to have converged. This distinction matters when describing the
method in a paper. The learned fusion parameters are new; only HiCMAE encoders
and matching pooling weights are initialized from the checkpoint.

## Condition adversary and objective

Condition targets describe the inputs actually constructed by the data loader:

| Saved ID | Internal condition | Meaning |
| --- | --- | --- |
| 0 | both | Original visual + original audio |
| 1 | video_noise | Original visual + white-noise audio |
| 2 | audio_neutral | Static neutral face + original audio |

All three conditions receive emotion and condition cross-entropy. The two players
have different gradient objectives:

```
encoders + feature heads + DBM: minimize L_emotion - lambda * L_condition
emotion classifier:            minimize L_emotion
condition classifier:          minimize L_condition
```

The implementation backpropagates `L_emotion + L_condition`, with a gradient
reversal layer whose backward multiplier is `-lambda` on the condition path to
the shared representation. Thus the classifier learns to identify the condition,
while the feature extractor learns to confuse it. Naively minimizing the negative
condition loss for the classifier itself would train a deliberately wrong
classifier rather than an adversarial one.

`--condition-lambda` defaults to 0.1 and must be finite/nonnegative. Setting it to
0 removes the adversarial feature gradient; the condition classifier still learns
as a diagnostic. No auxiliary emotion heads from the interaction experiment are
used. The condition classifier is absent from emotion prediction and the video
heatmap target. Confusing this classifier does not by itself establish equal
modality reliance or guarantee AV>A=V performance.

## Run

```bash
python run_ravdess_dbm.py --video-root /path/to/videos --audio-root /path/to/audio --checkpoint /path/to/checkpoint-99.pth --output-dir runs/ravdess_dbm01 --epochs 50 --batch-size 4 --condition-lambda 0.1 --dbm-visible-dim 128 --dbm-hidden-dims 256 128 --dbm-steps 5
```

Use a new output folder and retrain. Match `--embed-dim` and `--depth` to your
HiCMAE checkpoint; defaults remain 512 and 10. `--condition-hidden-dim` controls
the adversary's one ReLU hidden layer (default 128). The shared `--fusion-depth`
option has no effect in this runner because the original fusion encoder is absent.
There is a single DBM emotion head, so `--terms`/`--heads` are not options here.

Check pairing, split and configuration without ML dependencies:

```bash
python run_ravdess_dbm.py --video-root /path/to/videos --audio-root /path/to/audio --output-dir runs/dbm_prepare --prepare-only
```

The protocol is unchanged: six emotions; mixed-actor trial split with 128 training
and 64 test trials per emotion; 768 base train/384 test; no validation; each
training trial expanded to all three conditions (2304 presentations per epoch)
and shuffled. Training noise changes each epoch, test noise is fixed. The final
epoch is saved, without test-based checkpoint selection.

## Outputs, heatmaps and analysis

Existing epoch emotion metrics, trial predictions, final metrics, manifests,
`final.pth`, and configuration are retained. Config records `architecture=dbm`,
DBM settings, adversarial coefficient, condition-class order and training method.
Each prediction row includes the sole `dbm_logit_{emotion}` term plus the final
logits. There are no independently attributable additive audio/video terms.
The classifier archive contains the DBM emotion projection; the checkpoint contains
all encoder, DBM and classifier parameters.

Additional diagnostics:

- `condition_metrics.csv`: epoch and split, emotion CE, condition CE, condition
  accuracy/correct/total, prediction entropy, coefficient, effective feature
  objective (`emotion CE - lambda * condition CE`) and backward surrogate.
- `history.jsonl` and final test JSONs: `condition_adversary` metrics.
- Prediction CSVs: `condition_target`, `condition_prediction`, `condition_correct`
  and `condition_loss` for each presentation.

The backward surrogate is emotion CE plus condition CE during training. Its value
is not the feature player's objective; gradient reversal gives the two parameter
sets their proper different signs. Evaluation's `loss` remains emotion CE.

```bash
python video_loss_explanations.py --output-dir runs/ravdess_dbm01 --device cuda
python analyze_ravdess_results.py --run-dir runs/ravdess_dbm01
```

The explanation script detects DBM checkpoints automatically and retains the same
video-attention capture, emotion-loss target, colorful overlays and three-condition
folders. Gradients propagate through DBM mean-field inference; condition CE is not
part of the explanation target. Average heatmaps and existing emotion accuracy
figures remain supported.

Analysis adds `condition_adversary_learning.png/.pdf`: condition accuracy, CE and
entropy over epochs for training and pooled testing across all three conditions.
Uniform predictions have accuracy 1/3 on this balanced design and CE/entropy
ln(3). Inspect emotion accuracy too: a condition classifier at chance can also
occur when the shared representation has collapsed. Training diagnostics alone
are not an independent test of condition invariance.

DBM score-decomposition plots contain the sole DBM output term, not separate audio
and video contributions. Removing that only term leaves no classifier, so no
term-removal accuracy is reported. `--skip-epoch-tests` still suppresses epoch test
monitoring, while the final three-condition tests remain available.

## Verification and references

Preparation and analysis tests run without PyTorch. Optional PyTorch tests verify
DBM energy/mean-field updates, tied weights, gradient flow, gradient reversal signs
on features versus classifier weights, strict checkpoint loading and the trainer's
condition labels/diagnostic exports. Full training needs the real media, pretrained
checkpoint and PyTorch environment.

- [Deep Boltzmann Machines, Salakhutdinov & Hinton (2009)](https://proceedings.mlr.press/v5/salakhutdinov09a.html)
- [Domain-Adversarial Training of Neural Networks, Ganin et al. (2016)](https://jmlr.org/papers/v17/15-239.html)
