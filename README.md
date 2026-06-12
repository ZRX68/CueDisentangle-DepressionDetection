# Depression Multimodal Classification

This repository contains two PyTorch training pipelines for depression classification experiments on multimodal features.

## Project Structure

```text
src/
  dvlog_c/
    dataset.py
    model.py
    train.py
  lmvd_c/
    dataset.py
    model.py
    train.py
```

- `dvlog_c`: D-Vlog feature training code.
- `lmvd_c`: LMVD cue-level contrastive training code.

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install the project in editable mode:

```bash
pip install -e .
```

## Data Format

Both training scripts expect three directories:

- `video-dir`: video feature files, stored as `.npy`
- `audio-dir`: audio feature files, stored as `.npy`
- `label-dir`: label files named like `ID_Depression.csv`

The video and audio feature files should share the same file names, for example:

```text
video-dir/1.npy
audio-dir/1.npy
label-dir/1_Depression.csv
```

## Training

Train the DVLOG_C model:

```bash
python -m dvlog_c.train \
  --video-dir /path/to/video_features \
  --audio-dir /path/to/audio_features \
  --label-dir /path/to/labels
```

Train the LMVD_C model:

```bash
python -m lmvd_c.train \
  --video-dir /path/to/video_features \
  --audio-dir /path/to/audio_features \
  --label-dir /path/to/labels
```

Optional arguments:

```bash
--folds 10
--seed 2222
```

Training logs, prediction files, and model checkpoints are generated during training and are ignored by Git.

