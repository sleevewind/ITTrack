# ITTrack Submission Code

This repository contains the submission code for **ITTrack**, a unified UAV-based multiple object tracking framework for weak small targets in complex scenes.

## Environment Setup

Create a conda environment and install the required dependencies:

```bash
conda create -n ittrack python=3.8
conda activate ittrack

# Example PyTorch installation for CUDA 11.0
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 \
  -f https://download.pytorch.org/whl/torch_stable.html

cd ${ITTRACK_ROOT}
pip install cython
pip install -r requirements.txt
```

This project follows FairMOT and uses DCNv2. Please build it before training or testing:

```bash
git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git
cd DCNv2
./make.sh
```

## Data Preparation

Please prepare the datasets under your local `data` directory.

- VisDrone-MOT: [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- UAVDT: [UAVDT Dataset](https://github.com/dataset-ninja/uavdt)

Update the dataset configuration files if your local directory structure is different.

## Training

Use the public training scripts in the `experiments` directory:

```bash
sh experiments/train_visdrone_ittrack.sh
sh experiments/train_uavdt_ittrack.sh
```

## Testing

Use the corresponding public test scripts:

```bash
sh experiments/test_visdrone_ittrack.sh
sh experiments/test_uavdt_ittrack.sh
```

The tracking results will be saved as `.txt` files.

## Evaluation

Please evaluate the generated tracking results with the official benchmark toolkits:

- VisDrone-MOT toolkit: [official toolkit](https://github.com/VisDrone/VisDrone2018-MOT-toolkit)
- UAVDT toolkit: please follow the official dataset evaluation protocol

## Notes

- The provided scripts use relative paths to make the code easier to release publicly.
- Please adjust GPU IDs, model checkpoints, and local dataset paths according to your environment.
