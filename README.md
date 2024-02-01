
This repo is developed for image classification task using CIFAR-10 dataset. The structure of the repo is as follows:

```
home/
│
├── src/                      
│   ├── models/                 
│   │   ├── __init__.py
│   │   └── model.py
│   │
│   │── results/                # Store results
│   │  
│   ├── utils/                  # Utility functions and classes
│   │   ├── __init__.py
│   │   ├── data_preparation.py
│   │   ├── model_wrapper.py
│   │   └── experiment_runner.py
│   │
│   ├── config/                 # Configuration files for models, optimizers, training
│   │   ├── __init__.py
│   │   ├── model_config.py
│   │   ├── optimizer_config.py
│   │   └── training_config.py
│   │
│   ├── scripts/                # Scripts for analysis and other tasks
│   │   └── analyze_results.py
│   │
│   └── run.py                 # Main script
│
├── data/                       # CIFAR-10 Dataset
│   
│   
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_inference_time.py
│   ├── test_model.py
│   ├── test_save_load_model.py
│   └── test_training_process.py
│
├── checkpoints/                # Checkpoints for trained model weights, architectures
│
├── requirements.txt            # Project dependencies
│
├── README.md                   # Overview of the project
│
└── .gitignore                  # Specifies intentionally untracked files to ignore.
```

Brief description of your project.

## Installation

To install the necessary dependencies, activate your environment and run the following command:

```bash
pip install -r requirements.txt
```
## Run Experiments

To execute the experiments, run the following command:
```bash
python src/run.py
```
## Analyze Results

After running experiments, you can analyze the results with:
```bash
python src/scripts/analyze_results.py
```
To validate setup and functionality, run:
```bash
python -m unittest discover tests
```

