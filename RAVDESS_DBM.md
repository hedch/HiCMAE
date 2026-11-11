# DBM fusion with one audio-versus-video condition adversary

Run `run_ravdess_dbm.py` as a separate experiment. Keep the updated repository's
shared runner, analysis and explanation files alongside it. Existing interaction
and original-fusion runners remain available.

## Structure

1. Original pretrained HiCMAE video and audio encoders, without its fusion encoder.
2. Learned hierarchical pooling of each encoder's layer outputs.
3. Separate feature heads: Linear(D, visible_dim), LayerNorm, tanh.
4. A two-hidden-layer undirected Gaussian-visible/Bernoulli-hidden DBM.
5. Six-emotion linear classifier from the inferred top hidden probabilities.
6. One binary condition classifier from the same top representation, reached
   through gradient reversal.

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

## One audio-neutral versus video-noise adversary

New runs use exactly one binary classifier on the fused DBM representation:

| Input condition | Auxiliary target | Emotion CE |
| --- | --- | --- |
| Original audio + static neutral face (`audio_neutral`) | 0 | Included |
| Original video + white-noise audio (`video_noise`) | 1 | Included |
| Original audiovisual (`both`) | Excluded | Included |

The feature extractor and adversary have different gradient objectives:

```
encoders + feature heads + DBM: minimize L_emotion - lambda * L_audio_vs_video
emotion classifier:            minimize L_emotion
condition classifier:          minimize L_audio_vs_video
```

`--condition-lambda` controls the reversed feature gradient (default 0.1) and must
be finite/nonnegative. The previous `--audio-av-lambda` and `--video-av-lambda`
options are no longer used for new runs; replace them with `--condition-lambda`.
Setting it to zero removes the adversarial feature gradient, while the classifier
continues to learn as a diagnostic.

Emotion CE is averaged over all minibatch rows. Auxiliary CE is averaged only
over audio-neutral/video-noise rows, excluding audiovisual rows entirely. A batch
containing only audiovisual samples uses emotion CE alone. The backward scalar is
main CE plus eligible auxiliary CE; gradient reversal multiplies the auxiliary
feature gradient by `-lambda`, while classifier parameter gradients keep their
positive CE sign. Neither classifier predictions nor auxiliary loss enter the
emotion logits or the video heatmap target.

Each full epoch gives the adversary 1536 eligible presentations: 768 audio-neutral
and 768 video-noise. The 768 audiovisual presentations train emotion classification
only. Balanced binary chance accuracy is 50%. This objective encourages alignment
between the two control-condition representations; it does not explicitly align
either condition to audiovisual, nor guarantee equal modality reliance or AV>A=V.

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

For new runs, audio-neutral video inputs repeat frame 0 of the matched neutral
sample across all 16 frames. Config records `neutral_frame_policy=first`. The
shared loader applies this to training, testing and heatmaps. Old configurations
without the policy retain the previous midpoint-frame control when regenerating
explanations, so their original test input is preserved.

## Outputs, heatmaps and analysis

Existing epoch emotion metrics, trial predictions, final metrics, manifests,
`final.pth`, and configuration are retained. Config records `architecture=dbm`,
DBM settings, the adversarial coefficient, binary pair definition, condition-class order and training method.
Each prediction row includes the sole `dbm_logit_{emotion}` term plus the final
logits. There are no independently attributable additive audio/video terms.
The classifier archive contains the DBM emotion projection; the checkpoint contains
all encoder, DBM and classifier parameters.

Additional diagnostics:

- `pairwise_condition_metrics.csv`: epoch, split, adversary name, emotion CE,
  eligible binary CE/accuracy/correct/total, entropy, coefficient, effective
  feature objective and backward surrogate. The audiovisual test split is omitted
  from this auxiliary CSV because it has no eligible examples.
- `history.jsonl` and final test JSONs: `pairwise_condition_adversaries`, with each
  classifier's metrics, plus `adversarial_feature_objective`. A classifier with
  no eligible examples has total 0 and null loss/accuracy/entropy.
- Prediction CSVs: `audio_vs_video_target/prediction/correct/loss`. These fields
  are blank for audiovisual presentations.

The training backward surrogate is emotion CE plus eligible auxiliary CE;
gradient reversal gives the feature extractor the requested negative signs.
Existing `loss` fields remain emotion CE. Epoch component losses are averaged over
eligible presentations, while each optimized minibatch uses its own eligible-row
means. Therefore recombining epoch means is a descriptive feature objective and
need not equal the average optimized minibatch objective exactly.

```bash
python video_loss_explanations.py --output-dir runs/ravdess_dbm01 --device cuda
python analyze_ravdess_results.py --run-dir runs/ravdess_dbm01
```

The explanation script detects DBM checkpoints automatically and retains the same
video-attention capture, emotion-loss target, colorful overlays and three-condition
folders. Gradients propagate through DBM mean-field inference; condition CE is not
part of the explanation target. Average heatmaps and existing emotion accuracy
figures remain supported.

Analysis adds `condition_adversary_learning_audio_vs_video.png/.pdf` with binary
accuracy, CE and entropy for training and pooled testing of audio-neutral and
video-noise ONLY. Audiovisual is excluded from this auxiliary testing pool. Uniform
predictions have accuracy 1/2 and CE/entropy ln(2). Inspect emotion accuracy too:
chance condition accuracy can also occur when a representation collapses. These
diagnostics alone are not an independent test of condition invariance.

Older checkpoints with two AV/control adversaries or one three-way classifier
still load for explanations, and their original diagnostics remain analyzable.
New single-adversary runs require retraining in a new output folder; existing
runs are not converted.

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
