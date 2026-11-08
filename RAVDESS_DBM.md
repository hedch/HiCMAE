# DBM fusion with two pairwise condition adversaries

Run `run_ravdess_dbm.py` as a separate experiment. Keep the updated repository's
shared runner, analysis and explanation files alongside it. Existing interaction
and original-fusion runners remain available.

## Structure

1. Original pretrained HiCMAE video and audio encoders, without its fusion encoder.
2. Learned hierarchical pooling of each encoder's layer outputs.
3. Separate feature heads: Linear(D, visible_dim), LayerNorm, tanh.
4. A two-hidden-layer undirected Gaussian-visible/Bernoulli-hidden DBM.
5. Six-emotion linear classifier from the inferred top hidden probabilities.
6. Two separate binary condition classifiers from the same top representation,
   each reached through its own gradient reversal operation.

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

## Pairwise condition adversaries and objective

New runs replace the previous three-way classifier with two independent binary
classifiers operating on the fused DBM representation:

| Classifier | Label 0 | Label 1 | Excluded presentations |
| --- | --- | --- | --- |
| `audio_vs_av` | Original audiovisual | Original audio + static neutral face | Video + white-noise audio |
| `video_vs_av` | Original audiovisual | Original video + white-noise audio | Audio + static neutral face |

Original audiovisual rows contribute to both losses. Each control row contributes
only to its own pair's loss. Emotion CE still uses all three conditions. No labels
are inferred from filenames or emotion categories. Each binary classifier has its
own parameters and learns to minimize its eligible-pair CE; its reversed feature
gradient encourages that pair to become indistinguishable:

```
encoders + feature heads + DBM:
    minimize L_emotion - lambda_audio * L_audio_vs_av - lambda_video * L_video_vs_av
emotion classifier:
    minimize L_emotion
binary classifiers:
    minimize their respective condition cross-entropies
```

`--audio-av-lambda` controls audio-neutral vs audiovisual alignment.
`--video-av-lambda` controls video-noise vs audiovisual alignment. Both default to
`--condition-lambda` (0.1) unless explicitly supplied. Weights must be finite and
nonnegative. Setting a weight to 0 disables that adversary's feature gradient,
while its classifier continues to learn as a diagnostic.

Each minibatch uses main emotion CE averaged over all rows, plus each binary CE
averaged over that pair's eligible rows. A classifier with no eligible examples
in a minibatch contributes no loss for that batch. Gradient reversal applies
`-lambda_audio` or `-lambda_video` to the corresponding feature gradient; classifier
parameter gradients retain the positive CE sign. There are no added emotion
auxiliary heads. Neither condition classifier enters emotion prediction or the
video heatmap target.

The full epoch gives each binary classifier 1536 eligible training presentations:
768 audiovisual and 768 of its paired control. Binary chance accuracy is 50% on
this balanced design. Alignment does not by itself guarantee equal modality
reliance or AV>A=V performance; inspect emotion performance alongside confusion.

## Run

```bash
python run_ravdess_dbm.py --video-root /path/to/videos --audio-root /path/to/audio --checkpoint /path/to/checkpoint-99.pth --output-dir runs/ravdess_dbm01 --epochs 50 --batch-size 4 --audio-av-lambda 0.1 --video-av-lambda 0.1 --dbm-visible-dim 128 --dbm-hidden-dims 256 128 --dbm-steps 5
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
DBM settings, both adversarial coefficients, pair definitions, condition-class order and training method.
Each prediction row includes the sole `dbm_logit_{emotion}` term plus the final
logits. There are no independently attributable additive audio/video terms.
The classifier archive contains the DBM emotion projection; the checkpoint contains
all encoder, DBM and classifier parameters.

Additional diagnostics:

- `pairwise_condition_metrics.csv`: epoch, split, adversary name, emotion CE,
  eligible binary CE/accuracy/correct/total, entropy, coefficient, effective
  feature objective and backward surrogate. Ineligible test splits are omitted
  for that classifier.
- `history.jsonl` and final test JSONs: `pairwise_condition_adversaries`, with each
  classifier's metrics, plus `adversarial_feature_objective`. A classifier with
  no eligible examples has total 0 and null loss/accuracy/entropy.
- Prediction CSVs: `audio_vs_av_target/prediction/correct/loss` and
  `video_vs_av_target/prediction/correct/loss`. Fields are blank on unrelated rows.

The training backward surrogate is emotion CE plus both available binary CEs;
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

Analysis adds two sets of learning curves:

- `condition_adversary_learning_audio_vs_av.png/.pdf`
- `condition_adversary_learning_video_vs_av.png/.pdf`

Each plots binary accuracy, CE and entropy for training and pooled testing of
ONLY the relevant pair. Uniform predictions have accuracy 1/2 and CE/entropy
ln(2). Inspect emotion accuracy too: chance condition accuracy can also occur
when the shared representation collapses. These diagnostics alone are not an
independent test of condition invariance.

Older DBM checkpoints with the three-way classifier still load for explanations,
and their original three-way diagnostics remain analyzable. New pairwise runs
require retraining in a new output folder; existing runs are not converted.

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
