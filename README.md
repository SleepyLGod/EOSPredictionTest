# \<EoS\> Prediction Test

@`SleepyLGod` for current observation 📂 /cur

## Project Overview

This project implements and evaluates an end-of-sequence (EoS) prediction system for large language models (LLMs). The system trains neural networks to predict how many tokens remain in a generation sequence at any given step, enabling more efficient text generation and resource planning.

## Key Components

1. **Length Predictor Module** (`/cur/predictors/`)
   - Neural network training (`ebdModelGenFinal.py`)
   - Model evaluation framework (`resultTest.py`)
   - Results analysis and visualization (`graphsModified.py`)

2. **Signal Observation** (`/cur/signalObs/`)
   - Attention map analysis for understanding LLM behavior
   - Parameter impact studies

<!-- 3. **Prompt Engineering Tests** (`/cur/PETest.py`)
   - Experiments with different prompting strategies for length prediction -->

## Developing Rules

- Build one's own virtual environment and install the dependencies in `requirements.txt`
- Python 3.9.7 is recommended (see `.python-version`)
- Follow the code organization structure in existing modules

## Getting Started

### Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .env
source .env/bin/activate

# Install dependencies
pip install -r cur/requirements.txt

# Install appropriate PyTorch version for your system
# Visit https://pytorch.org/get-started/locally/ for instructions
```

### Running the Core Components

#### 1. Training a Length Predictor

```bash
cd cur/predictors
python ebdModelGenFinal.py
```

#### 2. Evaluating a Model

```bash
cd cur/predictors
python resultTest.py
```

#### 3. Analyzing Results

```bash
cd cur/predictors
python graphsModified.py
```

## Project Structure

```
/cur
├── predictors/               # Length prediction module
│   ├── ebdModelGenFinal.py   # Neural network training
│   ├── resultTest.py         # Model evaluation
│   ├── graphsModified.py     # Results visualization
│   ├── saved_models/         # Trained model checkpoints
│   ├── results/              # Analysis results
│   └── README.md             # Module documentation
│
├── signalObs/                # Attention map observation
│   ├── AttentionMapTest.ipynb
│   └── README.md             # Observation documentation
│
├── IdeaList.md               # Project ideas and concepts
├── PETest.py                 # Prompt engineering tests
├── datasetInit.ipynb         # Dataset initialization
└── requirements.txt          # Project dependencies
```

## Key Concepts

The project explores three main approaches to length prediction:

1. **Vector DB Approach**: Using vector databases to predict response length before prefilling
2. **Prompt Engineering**: Using specialized prompts to elicit length predictions from LLMs
3. **Neural Network Prediction**: Training dedicated models on LLM hidden states to predict remaining tokens

## Datasets

The project uses several datasets for training and evaluation:
- Alpaca dataset (cleaned version)
- Databricks Dolly dataset
- LMSYS Chat-1M

## Models Supported

- Meta-Llama-3 series (8B, 70B)
- DeepSeek-R1 series
- GPT-J, GPT-NeoX
- And others (see `signalObs/mapTest.py` for full list)
