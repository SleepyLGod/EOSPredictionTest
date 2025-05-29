# Length Predictor Module

This module implements an end-of-sequence (EOS) prediction system for large language models (LLMs). It trains a neural network to predict how many tokens remain in a generation sequence at any given step, enabling more efficient text generation and resource planning.

## Overview

The system consists of three main components:
1. **Model Training** (`ebdModelGenFinal.py`) - Trains the length prediction neural network
2. **Model Evaluation** (`resultTest.py`) - Evaluates trained models on various datasets
3. **Results Analysis** (`graphsModified.py`) - Generates comprehensive visualizations and statistics

## Core Files

### 🧠 `ebdModelGenFinal.py` - Neural Network Training
**Purpose**: Trains an Enhanced MLP (Multi-Layer Perceptron) to predict remaining sequence length using LLM embeddings and generation parameters.

**Key Features**:
- **Enhanced MLP Architecture**: 8192-dimensional input with residual connections, batch normalization, and dropout
- **Multi-GPU Training**: Distributed Data Parallel (DDP) support with automatic mixed precision (AMP)
- **Robust Data Loading**: Handles compressed `.npz` feature files with error recovery
- **Advanced Optimization**: AdamW optimizer with cosine annealing and gradient clipping

**Input Features**:
- LLM hidden state embeddings (8192-dim)
- Decoding parameters (temperature, top_k, repetition_penalty, max_tokens)
- Current sequence position
- Target: Remaining tokens until EOS

**Training Configuration**:
```python
HIDDEN_SIZE = 8192
BATCH_SIZE_PER_GPU = 32
LEARNING_RATE = 1e-3
EPOCHS = 50
```

### 🔬 `resultTest.py` - Model Evaluation Framework
**Purpose**: Comprehensive evaluation of trained length predictors on real text generation tasks using various datasets and decoding parameters.

**Key Features**:
- **Multi-Dataset Support**: Alpaca, Alpaca-Eval, Databricks Dolly datasets
- **Real-Time Prediction**: Step-by-step length prediction during actual LLM generation
- **Parameter Sweep**: Tests multiple temperature, top_k, repetition_penalty combinations
- **Detailed Logging**: Per-step predictions, latencies, and errors saved to JSONL format

**Evaluation Process**:
1. Load prompts from selected dataset (with training data exclusion)
2. For each prompt and parameter combination:
   - Generate text step-by-step with the LLM
   - At each step, predict remaining length using the trained model
   - Record actual vs predicted lengths
3. Calculate comprehensive error metrics (MAE, RMSE, bias)

**Output**: Detailed JSONL files with per-step predictions for downstream analysis

### 📊 `graphsModified.py` - Results Visualization & Analysis
**Purpose**: Generates comprehensive visualizations and statistical analysis from evaluation results.

**Key Features**:
- **Performance Metrics**: MAE, RMSE, R², bias calculations overall and per parameter group
- **Advanced Visualizations**:
  - Error distribution histograms
  - Prediction consistency heatmaps (TRAIL-style)
  - Scatter plots (predicted vs actual)
  - Per-prompt evolution plots
  - Latency analysis (violin plots)
- **Tail Analysis**: Special handling for sequences near completion (≤5 tokens remaining)
- **Parameter Group Analysis**: Separate analysis for each decoding parameter combination

**Generated Outputs**:
- Overall performance plots and metrics
- Per-parameter-group detailed analysis
- Per-prompt evolution tracking
- Aggregated metadata JSON with all statistics

## Folder Structure

```
cur/predictors/
├── README.md                          # This file
├── ebdModelGenFinal.py                # Neural network training script
├── resultTest.py                      # Model evaluation framework
├── graphsModified.py                  # Results analysis and visualization
├── idChecking.py                      # Utility for ID validation
├── used_prompt_ids.txt                # Training data prompt IDs (for exclusion)
│
├── saved_models/                      # Trained model checkpoints
│   └── 20250509_003641/              # Training run timestamp
│       ├── enhanced_mlp_best.pth     # Best model checkpoint
│       └── checkpoint_epoch_*.pth    # Per-epoch checkpoints
│
├── logs/                             # Training logs
│   ├── 20250509_003641/             # Training run logs
│   └── train_*.log                  # Historical training logs
│
├── results/                         # Analysis results
│   ├── clean/                       # Clean dataset results
│   ├── databricks/                  # Databricks dataset results
│   └── eval/                        # Evaluation dataset results
│       ├── overall/                 # Overall performance plots
│       ├── per_param_group/         # Parameter-specific analysis
│       └── per_prompt_evolution/    # Individual prompt tracking
│
├── eval_output/                     # Evaluation output directory
├── length_predictor_eval_results_*.jsonl  # Detailed evaluation results
└── Phase*.ipynb                     # Development notebooks
```

## Quick Start

### 1. Training a Model
```bash
# Ensure training data is available in ../training_data/ebd/features/llama3_70b/
python ebdModelGenFinal.py
```

### 2. Evaluating a Model
```bash
# Update LENGTH_PREDICTOR_PATH in resultTest.py to point to your trained model
python resultTest.py
```

### 3. Analyzing Results
```bash
# Update RESULTS_JSONL_FILE in graphsModified.py to point to your evaluation results
python graphsModified.py
```

## Dependencies

- **PyTorch**: Neural network training and inference
- **Transformers**: LLM loading and tokenization
- **Datasets**: HuggingFace dataset loading
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **scikit-learn**: Metrics calculation
- **tqdm**: Progress bars

## Key Metrics

The system tracks several important metrics:
- **MAE (Mean Absolute Error)**: Average prediction error in tokens
- **RMSE (Root Mean Square Error)**: Penalizes larger errors more heavily
- **R² Score**: Correlation between predicted and actual lengths
- **Bias**: Systematic over/under-prediction tendency
- **Tail Error Ratio**: Special metric for near-completion sequences
- **Latency**: Prediction inference time

## Notes

- Training requires pre-computed LLM embeddings and features (see `../training_data/`)
- Evaluation uses real LLM generation, requiring significant GPU memory
- Results analysis generates extensive visualizations for thorough performance assessment
- The system is designed for Meta-Llama-3-70B but can be adapted for other models
