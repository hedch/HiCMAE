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
einops / decord / NumPy / scikit-learn environment.

Download RAVDESS speech videos and WAVs from https://zenodo.org/records/1188976.
Keep neutral speech videos available for the test control. Extract the archives
and pass the video and audio parent directories; subdirectories are searched.

Example paired filenames:

```
videos/Actor_01/02-01-05-01-01-01-01.mp4
audio/Actor_01/03-01-05-01-01-01-01.wav
```

Both video-only (02) and full-AV (01) MP4 modalities are accepted; video-only is
preferred if both exist. Audio-only (03) WAV files are paired by filename. Song
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

Best checkpoint selection uses paired-input validation UAR only. After training,
the best checkpoint is loaded once and tested in all three conditions. Outputs:

- `best.pth`, `history.jsonl`, `config.json`, and `split_report.json`.
- `train.csv`, `val.csv`, and `test.csv` with trial/control paths.
- `test_metrics.json` comparing all three conditions.
- `test_metrics_{condition}.json` and `test_predictions_{condition}.csv`.

Metrics include accuracy, UAR, macro F1, and the confusion matrix. Prediction CSVs
include probabilities for all six classes.

## Verification

```bash
python -m unittest discover -s tests -p 'test_ravdess_interaction.py'
```

Data preparation tests need only Python. Optional PyTorch checks verify the logit
formula, gradients, absence of fusion modules, and seeded fixed-RMS noise. Real
media decoding and full fine-tuning require local RAVDESS media and a compatible
pretrained checkpoint; they have not been executed in the editing environment.
