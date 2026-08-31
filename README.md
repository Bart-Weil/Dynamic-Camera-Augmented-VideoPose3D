<div align="center">

# Dynamic Camera Augmented VideoPose3D

### 2D-to-3D human pose lifting that learns from camera motion

[Overview](#why-dynamic-cameras) · [Models](#models) · [Results](#project-results) · [Quick start](#quick-start) · [Documentation](#documentation)

</div>

## Why dynamic cameras?

Most 2D-to-3D pose-lifting benchmarks use fixed cameras or camera motion drawn from a narrow distribution. That is a poor match for footage from drones, panning phones, and augmented-reality devices, where the viewpoint changes throughout a sequence.

This project extends [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) with synthetic dynamic-camera training data and camera-aware model variants. It pairs sequences from the [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu/) with camera trajectories from [CMU Procedural Trajectories](https://github.com/Bart-Weil/CMU-Procedural-Trajectories). Each generated sequence contains aligned 2D poses, 3D poses, and camera parameters.

| Area | Included here |
|---|---|
| Dynamic-camera data | Procedural camera paths applied to real CMU motion-capture sequences |
| Camera-aware lifters | Coupled LSTM and Transformer architectures that consume pose and camera information |
| Model ensembling | A stacked pose lifter that combines Temporal FCN and Transformer predictions |
| Evaluation | Human3.6M, HumanEva-I, CMU Cameras, 3DPW, and custom-video workflows |
| Motion analysis | MPJPE/P-MPJPE reporting and correlations with camera and subject motion |

## Models

Choose a model with `--use-model`:

| Model | CLI value | Camera poses | Summary |
|---|---|:---:|---|
| Temporal FCN | `FCN` | No | VideoPose3D temporal-convolution baseline |
| Coupled LSTM | `LSTM-Coupled` | Yes | Jointly encodes temporal 2D poses and camera motion |
| Uncoupled LSTM | `LSTM-Uncoupled` | No | LSTM comparison without camera coupling |
| Camera Transformer | `Transformer` | Yes | Transformer encoder over pose and camera sequences |
| Stacked pose lifter | `StackedPoselifter` | Indirectly | Learns to combine pretrained FCN and Transformer outputs |

All architectures use [`run.py`](run.py) for training and evaluation. Command-line options control the receptive field, hidden dimensions, attention heads, dropout, and stacked-lifter weights.

## Project results

In the undergraduate project evaluation, dynamic-camera pretraining produced lower 3DPW errors than either no pretraining or stationary-camera pretraining:

| 3DPW evaluation | MPJPE ↓ | P-MPJPE ↓ |
|---|---:|---:|
| No fine-tuning | 410 mm | 202 mm |
| Stationary-camera pretraining + fine-tuning | 142 mm | 101 mm |
| Dynamic-camera pretraining + fine-tuning | 125 mm | 89.4 mm |

> [!NOTE]
> These values come from the undergraduate project evaluation, not a leaderboard submission. The reported MPJPE uses pose rooting, so it is not directly comparable with the standard 3DPW protocol. That choice does not affect P-MPJPE.

The generated CMU Cameras dataset can exceed 3 million frames without augmentation. In the final synthesis configuration, 97.9% of subject joints remained visible. This creates a controlled dataset for measuring how camera dynamics affect lifting error.

## Quick start

### 1. Create an environment

```bash
git clone https://github.com/Bart-Weil/Dynamic-Camera-Augmented-VideoPose3D.git
cd Dynamic-Camera-Augmented-VideoPose3D

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy scipy matplotlib tqdm h5py
```

Some workflows need extra dependencies: `cdflib` for Human3.6M CDF conversion, `ffmpeg` for video rendering, and Detectron2 for 2D keypoint inference on other videos.

### 2. Prepare data

- [Dataset setup](DATASETS.md) for Human3.6M, HumanEva-I, and custom 2D detections
- [`data/prepare_data_3dpw.py`](data/prepare_data_3dpw.py) for 3DPW conversion
- [CMU Procedural Trajectories](https://github.com/Bart-Weil/CMU-Procedural-Trajectories) for generating dynamic-camera CMU sequences

> [!IMPORTANT]
> This is a research snapshot. `run.py` and some preprocessing scripts contain local dataset archive paths. Change them for your machine before training.

### 3. Train a model

```bash
python run.py \
  --dataset CMU \
  --keypoints gt \
  --use-model Transformer \
  --subjects-train <train-subjects> \
  --subjects-validate <validation-subjects> \
  --subjects-test <test-subjects> \
  --epochs 60 \
  --checkpoint checkpoint
```

Use `FCN`, `LSTM-Coupled`, `LSTM-Uncoupled`, `Transformer`, or `StackedPoselifter` to switch architectures. Run `python run.py --help` for the complete option list.

### 4. Evaluate a checkpoint

```bash
python run.py \
  --dataset 3DPW \
  --keypoints detections \
  --use-model FCN \
  --subjects-test '*' \
  --checkpoint checkpoint \
  --evaluate model.bin
```

## Inference on your own video

To process your own video:

1. Detect 2D keypoints with Detectron2.
2. Convert detections to the repository's NumPy dataset format.
3. Lift the 2D sequence with a compatible checkpoint.
4. Render the 3D reconstruction or export joint coordinates.

See [Inference in the wild](INFERENCE.md) for the commands, assumptions, and limitations.

## Repository map

```text
.
├── common/
│   ├── datasets/        # Dataset loaders and skeleton definitions
│   ├── models/          # FCN, LSTM, Transformer, and stacked lifter
│   ├── generators.py    # Chunked and unchunked sequence batching
│   └── loss.py          # Pose and motion evaluation metrics
├── data/                # Dataset conversion scripts
├── inference/           # Detectron / Detectron2 video keypoint extraction
├── images/              # Diagrams and reconstruction demos
├── run.py               # Training, evaluation, and visualization entry point
└── gridsearch.json      # Model hyperparameter search spaces
```

## Documentation

- [Training, evaluation, and visualization reference](DOCUMENTATION.md)
- [Dataset setup](DATASETS.md)
- [Inference in the wild](INFERENCE.md)
- [Contributing](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)

## Research status

This repository accompanies the undergraduate project *2D-to-3D Human Pose Lifting Incorporating Known Camera Motions*. It is research code, not production inference software. The licenses and access conditions for 3DPW, Human3.6M, and the other datasets still apply.

## Acknowledgements

The code builds on [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) by Dario Pavllo, Christoph Feichtenhofer, David Grangier, and Michael Auli. It uses motion data from the [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu/).

## License

The code is distributed under the [Creative Commons Attribution-NonCommercial 4.0 International license](LICENSE). External datasets have their own terms, which may restrict redistribution of data or trained models.
