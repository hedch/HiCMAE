# Standalone indicator-concatenation MLP experiment

Run `run_ravdess_indicator_mlp.py`. It has its own data discovery, split generation,
media loader, model construction, training loop, optimizer and output saving. It
does not import or call any other experiment runner. It uses the original HiCMAE
encoder definitions in `modeling_finetune_av.py`, so keep the repository's upstream
model modules and existing Conda dependencies available.

## Model

The original video and audio encoders remain; their original cross-modal fusion
encoder is removed. Each encoder returns its patch/token features across layers.
We mean-pool tokens and use learned softmax layer weights to pool layers. Each
pooled feature then has its own feature head:

```
video head: LayerNorm(D) -> Linear(D, H) -> GELU -> hv
audio head: LayerNorm(D) -> Linear(D, H) -> GELU -> ha
MLP input: concatenate(hv, ha, on_v, on_a)
emotion logits: MLP(MLP input)
```

Here the heads are feature projections, not separate six-emotion classifiers.
Defaults: D=512, H=256, MLP input width=514, hidden layers 256 and 128, six output
logits. Hidden MLP layers use GELU and dropout 0.1; the final linear layer has no
softmax because training uses cross-entropy on logits.

Indicators are appended as two scalar input features. They do not multiply, zero
or gate the encoder features. Both encoders run for every input condition. There
is no DBM, interaction-product term, condition adversary or auxiliary emotion loss.

## Training and testing indicators

| Actual input condition | Training on_v/on_a | Testing on_v/on_a |
| --- | --- | --- |
| Original video + original audio | 1, 1 | 1, 1 |
| Original video + white-noise audio | 1, 0 | 1, 1 |
| Repeated neutral image + original audio | 0, 1 | 1, 1 |

All three conditions are mixed and shuffled during training. At testing, both
indicators are always 1, even when a real modality has been replaced by its control
input. This is an explicit protocol rule; the test flags are not inferred from the
condition. The model's eval-mode forward call also defaults to [1,1], so heatmaps
and standalone evaluation use the same rule. Training requires explicit flags.

The audio-only condition uses frame 0 of the actor/statement/repetition-matched
neutral sample, repeated across all 16 frames. The original emotional audio is
retained. Video-only uses fixed-RMS Gaussian white-noise audio. These placeholders
are still passed through the encoders; availability flags refer to the intended
emotion-bearing modality. The shared original audiovisual condition remains real
video/audio. Video and audio are cropped to a synchronized 2.56-second window.

Six emotions and order: happy, sad, angry, fearful, surprised, disgusted. The
mixed-actor split uses 128 training and 64 testing trials per emotion (768/384),
without validation. Each base training trial occurs under all three conditions,
creating 2304 shuffled presentations per epoch. Every condition therefore has 128
training presentations per emotion. Training noise changes reproducibly by epoch;
test noise is fixed. The held-out emotional trial identities remain disjoint.

The loss is ordinary six-class emotion cross-entropy over all presentations.
AdamW and cosine learning-rate decay train both encoders, pooling weights, feature
heads and MLP. The final epoch is saved; test accuracy does not select a checkpoint.

## Initialization and running

Default behavior is literal training from scratch: no pretrained encoder weights
are loaded when `--checkpoint` is omitted. The independent pipeline can optionally
initialize only the encoders/pooling from a checkpoint when requested. Its new
feature heads and MLP always start newly initialized.

```bash
python run_ravdess_indicator_mlp.py --video-root /path/to/videos --audio-root /path/to/audio --output-dir runs/indicator_mlp01 --epochs 30 --batch-size 4 --lr 0.0001 --device cuda --head-dim 256 --mlp-hidden-dims 256 128 --mlp-dropout 0.1
```

To use pretrained encoders, append:

```text
--checkpoint /path/to/checkpoint-99.pth
```

Match `--embed-dim` and `--depth` to that checkpoint. For scratch training they
configure the original encoder architecture (defaults: 512 and 10). Use a new
output folder for every experiment. `--weight-decay`, `--drop-path`, `--noise-rms`,
`--seed`, `--skip-epoch-tests`, `--skip-explanations` and `--explain-limit` are
available. No previous runner's head-selection or auxiliary-loss parameters apply.

Preparation runs using only Python's standard library and this script:

```bash
python run_ravdess_indicator_mlp.py --video-root /path/to/videos --audio-root /path/to/audio --output-dir runs/indicator_prepare --prepare-only
```

## Saved results and explanations

- `config.json`: architecture, initialization, MLP settings and explicit training
  and testing indicator policies.
- `train.csv`, `test.csv`, `split_report.json`: underlying trial manifests/counts.
- `train_expanded.csv`: condition-specific training rows with on_v/on_a.
- `epoch_metrics.csv`, `history.jsonl`, `epoch_predictions/`: online aggregate and
  per-condition training results, plus three-condition testing per epoch unless
  disabled. Accuracy, loss, correct/total, UAR, macro F1 and confusion matrices
  are saved. There is no validation split.
- Prediction CSVs: actual input availability (`available_v`, `available_a`) and
  flags actually supplied (`on_v`, `on_a`) are separate fields. Test on_v/on_a
  are always 1/1. Each row includes emotion logits, probabilities and correctness.
- `final.pth`: all trained parameters and run settings.
- `classifier_weights.npz`: feature-head/MLP tensors, not scalar modality weights.
- Final three-condition prediction CSVs and metric JSONs.

The updated shared explanation and analysis scripts support this architecture:

```bash
python video_loss_explanations.py --output-dir runs/indicator_mlp01 --device cuda
python analyze_ravdess_results.py --run-dir runs/indicator_mlp01
```

Explanation-only loading dispatches to this standalone runner's model and loader,
without importing the interaction runner for this architecture. It explains the
emotion CE through the MLP/video encoder with both indicators fixed to 1, saving
that flag setting in explanation metadata. Per-condition images, colorful
heatmaps and global average emotion heatmaps keep the existing output format.

Analysis retains emotion/condition accuracy curves and emotion bars. The model
has one joint nonlinear MLP output; it does not have independently additive visual
and audio logits. Its sole recorded term is `mlp_logit_{emotion}`, equal to the
final logit. Score-decomposition plots therefore describe the single joint output
and do not quantify separate modality contributions. Removing its only term has
no meaningful classifier accuracy.

## Verification

Tests cover independent import/preparation with no other runner files, balanced
and disjoint data splits, exact indicator policies, configuration/manifest exports,
and numerical single-output analysis. Optional PyTorch tests cover concatenation,
feature gradients, checkpoint loading, eval defaults and actual training/testing
flags saved by the training loop. Full GPU/media training requires the user's
PyTorch environment and dataset.
