# Length Predictor Module

This module implements an end-of-sequence (EOS) prediction system for large language models (LLMs). It trains a neural network to predict how many tokens remain in a generation sequence at any given step, enabling more efficient text generation and resource planning.

## Overview

The system consists of six main components:
1. **Model Training** (`ebdModelGenFinal.py`) - Trains the length prediction neural network
2. **Model Evaluation** (`resultTest.py`) - Evaluates trained models on various datasets
3. **Results Analysis** (`graphsModified.py`) - Generates comprehensive visualizations and statistics
4. **Error Profile Analysis** (`errorProfileGen.py`) - Analyzes step-0 prediction errors and generates error profiles
5. **Error Evolution Analysis** (`errorRatioEvol.py`) - Tracks error ratio evolution throughout generation sequences
6. **Parameter Optimization** (`sysParaRank.py`) - Ranks and optimizes decoding parameter combinations

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

### 📈 `errorProfileGen.py` - Error Profile Analysis
**Purpose**: Analyzes initial prediction errors (step-0) and generates comprehensive error profiles for different parameter combinations.

**Key Features**:
- **Step-0 Error Analysis**: Focuses on the critical first prediction made after prompt prefilling
- **Error Ratio Calculations**: Computes prediction ratios and error percentages
- **Parameter Group Profiling**: Generates separate error profiles for each decoding parameter combination
- **Statistical Analysis**: Comprehensive statistics including mean, median, std dev, and percentiles
- **Distribution Plotting**: Creates error ratio distribution plots with multiple ranges and density visualization

**Analysis Types**:
- Prefill-based error profiles (step_index=0 predictions)
- Combined error profiles across all parameters
- Error ratio distributions (overall and per-parameter group)
- Focused and wide-range distribution plots

### 🔄 `errorRatioEvol.py` - Error Evolution Analysis
**Purpose**: Tracks and visualizes how prediction errors evolve throughout the generation sequence for individual prompts.

**Key Features**:
- **Error Ratio Evolution**: Plots error ratio changes across decoding steps
- **Prediction Length Tracking**: Shows predicted remaining length annotations
- **Curve Fitting**: Applies linear and quadratic trend fitting with tail data exclusion
- **Volatility Analysis**: Calculates error volatility metrics for different sequence segments
- **Enhanced Visualizations**: Three-panel layouts with error ratios, prediction lengths, and tail data tables

**Analysis Capabilities**:
- Single-prompt error evolution plots
- Error tokens vs. decoding step analysis
- Tail behavior analysis (last 5-20% of tokens)
- Statistical curve fitting and trend analysis
- Volatility metrics for different sequence segments

### 🏆 `sysParaRank.py` - Parameter Optimization
**Purpose**: Ranks decoding parameter combinations based on composite performance scores to identify optimal configurations.

**Key Features**:
- **Composite Scoring**: Combines multiple metrics (MAE, RMSE, bias, error ratios) into unified scores
- **Parameter Ranking**: Ranks parameter groups from best to worst performing
- **Statistical Aggregation**: Processes raw evaluation logs to extract key performance metrics
- **Top-N Selection**: Identifies and reports the best-performing parameter combinations

**Ranking Metrics**:
- Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)
- Absolute bias and standard deviation of errors
- Mean and standard deviation of error ratios
- Prediction ratio deviations from ideal (1.0)
- Composite scores based on rank aggregation

## Folder Structure

```
cur/predictors/
├── README.md                          # This file
├── ebdModelGenFinal.py                # Neural network training script
├── resultTest.py                      # Model evaluation framework
├── graphsModified.py                  # Results analysis and visualization
├── errorProfileGen.py                 # Error profile analysis and step-0 error analysis
├── errorRatioEvol.py                  # Error ratio evolution analysis and plotting
├── sysParaRank.py                     # System parameter ranking and optimization
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
├── results/                         # Analysis results from graphsModified.py
│   ├── clean/                       # Clean dataset results
│   ├── databricks/                  # Databricks dataset results
│   └── eval/                        # Evaluation dataset results
│       ├── overall/                 # Overall performance plots
│       ├── per_param_group/         # Parameter-specific analysis
│       └── per_prompt_evolution/    # Individual prompt tracking
│
├── eval_output/                     # Advanced analysis outputs
│   ├── clean/                       # Clean dataset analysis
│   │   ├── group_step0_error_ratio_plots/  # Per-parameter error distributions
│   │   ├── per_param_group/         # Parameter group analysis
│   │   └── sum/                     # Summary plots
│   ├── databricks_evol/             # Databricks evolution analysis
│   ├── eval_evol/                   # Evaluation evolution analysis
│   ├── prefill_err_profiles_*.jsonl # Error profile data files
│   └── rank_param_*.txt             # Parameter ranking results
│
├── length_predictor_eval_results_*.jsonl  # Detailed evaluation results
├── pro_length_predictor_eval_results_*.jsonl  # Professional evaluation results
└── __pycache__/                     # Python cache files
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

### 3. Basic Results Analysis
```bash
# Update RESULTS_JSONL_FILE in graphsModified.py to point to your evaluation results
python graphsModified.py
```

### 4. Advanced Error Analysis
```bash
# Generate error profiles and step-0 analysis
python errorProfileGen.py

# Analyze error evolution for individual prompts
python errorRatioEvol.py

# Rank parameter combinations by performance
python sysParaRank.py
```

## Complete Analysis Workflow

For a comprehensive analysis of your length predictor, follow this workflow:

1. **Train the model** using `ebdModelGenFinal.py`
2. **Evaluate the model** using `resultTest.py` to generate detailed JSONL results
3. **Generate basic visualizations** using `graphsModified.py`
4. **Analyze error profiles** using `errorProfileGen.py` for step-0 error distributions
5. **Track error evolution** using `errorRatioEvol.py` for individual prompt analysis
6. **Optimize parameters** using `sysParaRank.py` to identify best-performing configurations

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

### Primary Performance Metrics
- **MAE (Mean Absolute Error)**: Average prediction error in tokens
- **RMSE (Root Mean Square Error)**: Penalizes larger errors more heavily
- **R² Score**: Correlation between predicted and actual lengths
- **Bias**: Systematic over/under-prediction tendency
- **Latency**: Prediction inference time

### Advanced Error Analysis Metrics
- **Error Ratio**: `(actual_rest_len - predicted_rest_len) / actual_rest_len`
- **Prediction Ratio**: `(prompt_len + predicted_rest) / (prompt_len + actual_generated)`
- **Step-0 Error**: Initial prediction error immediately after prompt prefilling
- **Error Evolution**: How prediction errors change throughout generation
- **Volatility Metrics**: Error stability across different sequence segments

### Parameter Optimization Metrics
- **Composite Score**: Weighted combination of multiple performance metrics
- **Parameter Ranking**: Relative performance of different decoding configurations
- **Error Profile Statistics**: Mean, median, standard deviation, and percentiles of errors

## Output Files

The system generates various types of output files:

### Evaluation Results
- `length_predictor_eval_results_*.jsonl`: Detailed step-by-step evaluation results
- `pro_length_predictor_eval_results_*.jsonl`: Professional evaluation results

### Error Analysis
- `prefill_err_profiles_*.jsonl`: Error profile data for different datasets
- `rank_param_*.txt`: Parameter ranking results with performance scores

### Visualizations
- Error ratio distribution plots (overall and per-parameter group)
- Error evolution plots for individual prompts
- Parameter performance comparison charts
- Statistical analysis summaries

## Notes

- Training requires pre-computed LLM embeddings and features (see `../training_data/`)
- Evaluation uses real LLM generation, requiring significant GPU memory
- The advanced analysis pipeline generates extensive visualizations and statistical reports
- Error profile analysis focuses on step-0 predictions which are critical for early stopping decisions
- Parameter optimization helps identify the best decoding configurations for your specific use case
- The system is designed for Meta-Llama-3-70B but can be adapted for other models
- All analysis scripts can be configured by modifying the file paths at the top of each script
