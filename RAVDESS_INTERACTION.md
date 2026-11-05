# RAVDESS interaction experiment

Run from this repository's root using its existing requirements (Python 3.8,
PyTorch, torchaudio, timm 0.4.12, decord, einops, NumPy, scikit-learn).

## Model

No cross-modal fusion encoder is registered or executed. Each encoder returns
features from every layer. Tokens are mean-pooled, then layers are pooled using
learned softmax weights, as in HiCMAE. The resulting `ov` and `oa` are B x D.

```
logits = wv(ov) + wa(oa) + wf(ov * oa)
```

The multiplication is elementwise. Each of `wv`, `wa`, `wf` is an independent,
unconstrained, bias-free learned linear map from D to 8, implementing the requested
formula exactly. These are weight matrices, not three scalar mixture weights.
Cross entropy receives the raw logits. Both encoders and all new weights are
fine-tuned. No pretraining reconstruction or contrastive loss is used.

The defaults match the released base checkpoint: D=512, 10 layers in each encoder,
16 video frames at 160 x 160, audio filterbanks of 256 x 128. Use `--embed-dim 384`
and the appropriate `--depth` for other checkpoints. All encoder parameter names
and shapes must match; incompatible checkpoints fail instead of silently leaving
encoder weights uninitialized. Fusion, decoder and old classifier weights are ignored.

## Data

Download and extract the RAVDESS **speech** videos and speech WAVs:
https://zenodo.org/records/1188976

Pass a directory containing MP4s (original video-only `02` or full AV `01` modality)
and another containing audio-only WAVs (`03`). Subdirectories are searched recursively.
Example matching pair:

```
videos/Actor_01/02-01-05-01-01-01-01.mp4
audio/Actor_01/03-01-05-01-01-01-01.wav
```

Keep the original seven-field filenames. This script uses all eight speech emotions;
song files are excluded. Audio is resampled to 16 kHz internally. Videos and audio
must retain the same time origin. Face-cropped MP4s retaining the original frame
rate and full duration are preferable for transfer from the face-pretrained model.
Raw videos are accepted, but the center square crop is not a face detector.
Extracted image folders from the original HiCMAE loader are not supported here.

The script samples synchronized 2.56-second windows (random during training,
centered during evaluation), with 16 uniformly sampled video frames. Short clips
use their available interval and zero-padded audio. Video ImageNet normalization
and the upstream AudioSet filterbank normalization are used. Loading errors stop
the experiment instead of substituting a different labeled sample.

## Balancing and split

A seeded actor shuffle assigns 16 actors to training, 4 to validation and 4 to
testing. To choose your own held-out groups, pass BOTH `--val-actors` and
`--test-actors`; all remaining actors are used for training.

Within EACH split, every emotion is randomly undersampled without replacement to
the smallest class count. Therefore all eight emotions have exactly equal numbers
of unique trials, including validation and test. No duplicated oversampling is used.
For complete speech data, expected counts per emotion are 64 training, 16 validation,
and 16 test (768 selected trials total). This discards many non-neutral trials.
Validation/test metrics describe these balanced subsets rather than the entire
original distribution. Saved manifests and the split report record every selection.

Check paths and counts without loading any models:

```bash
python run_ravdess_interaction.py --video-root /path/to/videos --audio-root /path/to/audio --output-dir runs/ravdess_check --prepare-only
```

Fine-tune using the trusted upstream pretrained checkpoint:

```bash
python run_ravdess_interaction.py --video-root /path/to/videos --audio-root /path/to/audio --checkpoint /path/to/checkpoint-99.pth --output-dir runs/ravdess_interaction --batch-size 4 --epochs 50 --lr 1e-4
```

Use a fresh output directory for every invocation, including after preparation.
Default device is `cuda`; pass `--device cpu` for CPU execution. This is a single
device script with full precision and zero data-loader workers for Windows support.
Lower the batch size if GPU memory is insufficient. Multi-GPU/DDP, resuming and
automatic dataset download are not implemented.

The best checkpoint is selected using validation UAR only. Test data are evaluated
once, after training. Outputs include `best.pth`, `history.jsonl`, `config.json`,
the three manifests, `split_report.json`, `test_metrics.json`, and
`test_predictions.csv`. Metrics include accuracy, UAR, macro F1 and the confusion
matrix. Class order is neutral, calm, happy, sad, angry, fearful, disgust, surprised.

## Checks

```bash
python -m unittest discover -s tests -p 'test_ravdess_interaction.py'
```

Data preparation tests run without ML dependencies. The optional model check needs
the repository's PyTorch dependencies and verifies logits, gradients, and the absence
of a fusion module with tiny input shapes. No training accuracy is asserted.
