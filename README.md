# ML model for predicting elastic modulus of imperfect planar lattice structures

This repository predicts the elastic modulus of planar lattice structures using **Graph Neural Networks (GNNs)**.

## Repository Structure


```text
├── final_model_local.py          # Main local Python script for training and evaluation
├── gin_model.ipynb               # GIN notebook experiment
├── gin_model_6_features.ipynb    # GIN experiment with additional node features
├── saved_model/                  # Saved model checkpoints and generated figures
├── utils/
│   ├── architecture.py           # GIN and Graph Transformer model definitions
│   ├── data_processing.py        # Graph utilities, normalization, dataset split, duplicate checking
│   ├── train_model.py            # Training loop
│   └── evaluate_model.py         # Evaluation and prediction-vs-ground-truth visualization
├── requirements.txt              # Python package dependencies
├── Hexagonal/                    # Dataset folder, not included in repository
├── Kagome/                       # Dataset folder, not included in repository
├── Demi-A/                       # Dataset folder, not included in repository
    ├── Results.txt
    └── inp_files/
        ├── sample_1.inp
        ├── sample_2.inp
        └── ...
├── Demi-B/                       # Dataset folder, not included in repository
├── Demi-C/                       # Dataset folder, not included in repository
└── README.md
```

## Models

* **GIN (Graph Isomorphism Network)** with 3 convolution layers and global pooling
* **Graph Transformer** using 4 attention heads

Each node is represented by two features (x, y) coordinates. Graph connectivity encodes struts. The model predicts a single scalar output (elastic modulus).

## Dataset

* Data is generated from finite element (FE) simulations of planar lattice structures
* Each simulation corresponds to one `.inp` file and one elastic modulus E value
* The dataset is not included in this repository

## Reproducibility

The repository provides:

* Complete model architectures
* Training and evaluation pipelines

## Installation

Create or activate a Python environment, then install the required packages:

```bash
pip3 install -r requirements.txt
```

On macOS, use `python3` and `pip3` instead of `python` and `pip` if needed.

## Running the Code Locally

From the repository root, run:

```bash
python3 final_model_local.py
```

The default settings are:

- model: Graph Transformer
- batch size: 32
- epochs: 200
- learning rate: 0.001
- train/validation/test split: 80% / 10% / 10%

You can also change settings from the command line:

```bash
python3 final_model_local.py --model gin --batch_num 64 --epoch_num 100
```

If the dataset is stored in different folder, pass the path using:

```bash
python3 final_model_local.py --data_dir /path/to/dataset
```

The script automatically checks the available device:

1. Apple GPU through MPS,
2. CUDA GPU,
3. CPU.

## Outputs

Generated files are saved in `saved_model/`, including:

- trained model checkpoint, for example `epoch_200.pt`
- dataset distribution figure
- training loss figure
- prediction-vs-ground-truth evaluation plots

## License

This repository is intended for **academic and educational use**.

## Reference 
```
@article{chung2024prediction,
title={Prediction of effective elastic moduli of rocks using Graph Neural Networks},
author={Chung, Jaehong and Ahmad, Rasool and Sun, WaiChing and Cai, Wei and Mukerji, Tapan},
journal={Computer Methods in Applied Mechanics and Engineering},
volume={421},
pages={116780},
year={2024},
publisher={Elsevier}
}
```
