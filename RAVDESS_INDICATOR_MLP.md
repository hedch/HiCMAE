# Standalone MLP with fixed first-layer gradient gates

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
MLP input: concatenate(hv, ha)
emotion logits: MLP(MLP input)
```

Here the heads are feature projections, not separate six-emotion classifiers.
Defaults: D=512, H=256, MLP input width=512, hidden layers 256 and 128, six output
logits. Hidden MLP layers use GELU and dropout 0.1; the final linear layer has no
softmax because training uses cross-entropy on logits.

Indicators are NOT concatenated to the MLP input in new runs. The first hidden
layer is permanently split into two equal neuron groups. The first half's incoming
weights/biases are controlled by on_v and the second half's by on_a; the assignments
never shuffle across epochs, batches or GPUs. The first hidden width must be even
and at least 2, but subsequent layers can have any positive widths.

## Training-only backward gating

| Actual input condition | First-half gradient gate | Second-half gradient gate | Testing indicators |
| --- | --- | --- | --- |
| Original video + original audio | 1 | 1 | Not supplied |
| Original video + white-noise audio | 1 | 0 | Not supplied |
| Audio-only: white image in training, neutral image in testing | 0 | 1 | Not supplied |

The gates act on gradients immediately after the first hidden activation, using:

```
g = concatenate(repeat(on_v, first_width/2), repeat(on_a, first_width/2))
h_gated = detach(h) + g * (h - detach(h))
```

Forward values are exactly h. Backward gradients into the first hidden layer are
multiplied by g for EACH example. This is backward-only gating, not neuron
activation dropout or forward masking. Subsequent MLP layers train normally using
all examples. Both encoders run for every condition. There is no DBM, adversarial
classifier or auxiliary emotion loss.

Both neuron groups receive the full concatenated visual/audio features; the gate
names label groups, not exclusive connections to one modality. Either active
group can therefore send gradients to both encoders. Only first-layer incoming
weights/biases are subject to the strict inactive-group freezing rule. Frozen
weights do not mean constant activations when the upstream encoders change.

The two first-layer groups have separate parameter tensors. After backward, if
NO example in the entire global minibatch enables a group, its parameter gradients
are set to None before AdamW.step(). AdamW then skips that group entirely, including
its weight decay, momentum and optimizer-state step. With mixed examples, an
active group updates once per optimizer step, using only its eligible examples'
task gradients. Global eligibility is checked after DataParallel reduces gradients,
so a GPU-local inactive group can still update from eligible examples on another
selected GPU.

## Indicator-free testing and explanations

All three conditions run without indicator inputs or gradient gates during testing.
Both neuron groups participate in every forward pass. The previous [1,1]/[1,0]
test-input policy no longer applies to newly trained gated models. Direct eval-mode
forward calls and heatmap regeneration require only video and audio tensors.

During audio-only training, the visual input is a solid white RGB image
(255,255,255), repeated across all 16 frames and processed with the same image
normalization. During audio-only testing and heatmap regeneration, the visual
input remains frame 0 of the actor/statement/repetition-matched neutral sample,
repeated across all 16 frames. The original emotional audio is retained.
Config records `training_audio_only_visual=white` and
`testing_audio_only_visual=neutral`. Video-only uses fixed-RMS Gaussian
white-noise audio. These placeholders
are still passed through the encoders; availability flags refer to the intended
emotion-bearing modality. The shared original audiovisual condition remains real
video/audio. Video and audio are cropped to a synchronized 2.56-second window.

Six emotions and order: happy, sad, angry, fearful, surprised, disgusted. The
split is actor-disjoint: 20 actors for training and 4 held-out actors for testing,
selected by shuffling the sorted 24 actor IDs using `--seed` (default 42). No actor
appears in both partitions, including the neutral references matched to that actor.
There is no validation set.

On the complete dataset each actor has 8 trials per target emotion, giving 160
training and 32 test trials per emotion: 960 base training trials and 192 test
trials. Every training trial appears in all three shuffled conditions, creating
2880 presentations per epoch (960 per condition; 480 per emotion across conditions).
The test set is evaluated separately under all three conditions.

Each partition is balanced separately across emotions by downsampling to its
smallest emotion count if files are incomplete; actors are never moved between
partitions to compensate. Missing an entire emotion in a partition raises an
error. The 20:4 split requires all 24 actors to have emotional trials. Config and
split report save the selected actor IDs, actual class counts and unused counts.
Training noise changes reproducibly by epoch; test noise is fixed. Use a new run
folder and retrain to adopt this split. Older run manifests remain unchanged.

The loss is ordinary six-class emotion cross-entropy over all presentations;
training gates control which samples contribute first-layer gradients.
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

## Select one or several GPUs (MLP runner only)

The optional `--gpu-ids` parameter selects CUDA devices. One ID runs on that GPU;
multiple IDs use PyTorch `DataParallel` in a single process, compatible with normal
PyCharm script execution. The first listed GPU is the primary device for the
model, optimizer and gathered logits.

Append these parameters to your usual MLP command:

```text
--device cuda --gpu-ids 0 1 2 3 --batch-size 16
```

Other examples:

```text
--device cuda --gpu-ids 2 --batch-size 4
--device cuda --gpu-ids 1 3 --batch-size 8
```

If `--gpu-ids` is omitted, the runner retains single-GPU behavior: `--device cuda`
selects GPU 0, while `--device cuda:2` selects GPU 2. When both are provided,
`--gpu-ids` determines the selection and its first ID determines the primary GPU.
CPU execution uses `--device cpu` without GPU IDs. Duplicates, negative IDs and
out-of-range IDs are rejected; runtime validates IDs against visible CUDA devices.
GPU IDs are relative to devices visible to PyTorch, including any
CUDA_VISIBLE_DEVICES filtering. Preparation validates syntax without requiring
CUDA/PyTorch to be installed on the preparation machine.

`--batch-size` is GLOBAL, not per GPU. Batch 16 across four GPUs normally gives
four presentations per GPU. A global batch of four gives only one per GPU and
parallel overhead can dominate; larger batches may improve utilization if memory
permits. Increasing global batch size changes optimizer steps per epoch; the
runner does not automatically change the learning rate. Media decoding remains
on the existing single-process loader, so actual speedup depends on input loading
and GPU communication as well as model compute. No fourfold speedup is promised.

Training and epoch/final testing use the selected devices. Single-trial heatmap
export deliberately uses the unwrapped model on the primary GPU, preserving
native attention hooks and indicator-free evaluation for new gated models. Checkpoints and
weight archives are saved without DataParallel's `module.` prefix and can be
loaded on one GPU for explanation regeneration. `parallelism.json` records GPU
IDs/names, primary device, mode and global batch size; final metrics also record
this configuration. Other experiment runners are unchanged.

Reference: [PyTorch DataParallel documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html).

## Saved results and explanations

- `config.json`: architecture, initialization, MLP settings,
  `mlp_input_mode=gradient_gated`, the fixed half assignments and training gates.
  `test_indicators` is null.
- `train.csv`, `test.csv`, `split_report.json`: underlying trial manifests/counts.
- `train_expanded.csv`: condition-specific training rows with on_v/on_a.
- `epoch_metrics.csv`, `history.jsonl`, `epoch_predictions/`: online aggregate and
  per-condition training results, plus three-condition testing per epoch unless
  disabled. Accuracy, loss, correct/total, UAR, macro F1 and confusion matrices
  are saved. There is no validation split.
- Prediction CSVs: actual input availability (`available_v`, `available_a`) and
  training gates (`on_v`, `on_a`) are separate fields. Test on_v/on_a
  are blank because no indicators are supplied. Each row includes emotion logits, probabilities and correctness.
- `final.pth`: all trained parameters and run settings.
- `classifier_weights.npz`: feature-head/MLP tensors, not scalar modality weights.
- Final three-condition prediction CSVs and metric JSONs.

The updated shared explanation and analysis scripts support this architecture:

```bash
python video_loss_explanations.py --output-dir runs/indicator_mlp01 --device cuda
python analyze_ravdess_results.py --run-dir runs/indicator_mlp01
```

Explanation-only loading dispatches to this standalone runner's model and loader,
without importing the interaction runner for this architecture. It explains emotion
CE through the full MLP/video encoder in eval mode, without indicators or backward
gates. Per-condition images, colorful heatmaps and global average emotion heatmaps
keep the existing format.

Older concatenated-indicator checkpoints still reconstruct their original model
and saved test-indicator policy. New gated checkpoints have a different first-layer
shape/parameter layout and must be retrained in a NEW output folder. They cannot
reuse an old MLP head directly. The optional --checkpoint encoder initializer can
still load matching encoder/pooling weights while initializing the new head.

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
backward-only sample masks, fixed group assignments, exact inactive-half AdamW
parameter/state freezing, checkpoint loading and indicator-free testing. Full GPU/media training requires the user's
PyTorch environment and dataset.
