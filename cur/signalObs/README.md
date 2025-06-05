# Attention Map Observation

This module provides comprehensive tools for analyzing attention patterns in large language models (LLMs) during text generation. It captures, visualizes, and analyzes attention maps across different layers, heads, and generation steps to understand model behavior and identify potential signals for end-of-sequence prediction.

✍️ @SleepyLGod

## Overview

The system consists of several analysis tools:

1. **Interactive Analysis** (`AttentionMapTest.ipynb`, `attentionObs.ipynb`) - Jupyter notebooks for exploratory attention analysis
2. **Attention Map Generation** (`mapTest.py`) - Generates attention heatmaps for specific prompts and models
3. **Parameter Impact Analysis** (`parameterRangeTest.py`) - Studies how decoding parameters affect generation length
4. **Attention Signal Detection** (`valueTest.py`) - Detects attention patterns that might signal end-of-sequence
5. **Visualization Tools** (`graphsCount.py`) - Combines multiple attention heatmaps into comprehensive visualizations
6. **Automated Testing** (`autoTest.sh`, `longRun.sh`) - Scripts for batch processing and systematic analysis
7. **Dataset Processing** (`attached/datasetTest.py`) - Processes datasets for attention analysis

## Core Files

### 📊 `mapTest.py` - Attention Map Generation
**Purpose**: Generates detailed attention heatmaps for specific prompts across different models and datasets.

**Key Features**:
- **Multi-Model Support**: Llama-3.2-1B, Meta-Llama-3-8B, Meta-Llama-3-70B, Meta-Llama-3-8B-Instruct, GPT-J, Llama-2, GPT-NeoX
- **Multi-Dataset Support**: Alpaca, Simplified Alpaca, LMSYS Chat 1M
- **Layer-Head Analysis**: Generates heatmaps for all layer-head combinations
- **Generation Tracking**: Records attention patterns throughout the generation process
- **Output Management**: Saves heatmaps, generation info, and metadata to organized directories

### 🔬 `valueTest.py` - Attention Signal Detection
**Purpose**: Analyzes attention patterns to detect potential signals that indicate approaching end-of-sequence.

**Key Features**:
- **Tail Attention Analysis**: Monitors attention scores in the last few tokens of sequences
- **Alert System**: Detects when attention concentrates on recent tokens (potential EOS signal)
- **Multi-Model Testing**: Supports the same model range as mapTest.py
- **Detailed Logging**: Records attention alerts with specific scores and positions

### 📈 `parameterRangeTest.py` - Parameter Impact Analysis
**Purpose**: Systematically tests how different decoding parameters affect generation length and behavior.

**Key Features**:
- **Parameter Sweeps**: Tests ranges of repetition_penalty, temperature, and top_k values
- **Length Tracking**: Records output lengths for each parameter combination
- **Visualization**: Generates plots showing parameter vs. output length relationships
- **Comprehensive Testing**: Three separate experiments for each parameter type

### 🎨 `graphsCount.py` - Visualization Combination
**Purpose**: Combines individual attention heatmaps into comprehensive grid visualizations.

**Key Features**:
- **Grid Layout**: Arranges heatmaps by layer (rows) and head (columns)
- **Missing Data Handling**: Creates blank spaces for missing heatmaps
- **Large-Scale Visualization**: Supports up to 15 layers × 31 heads grids

## Running Config

```bash
# python >= 3.9.7
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
# install your own torch version
```

## Quick Start

### 1. Single Attention Map Analysis
```bash
python mapTest.py
# Follow prompts to select model, dataset, and prompt
```

### 2. Parameter Impact Testing
```bash
python parameterRangeTest.py
# Tests how different parameters affect generation length
```

### 3. Attention Signal Detection
```bash
python valueTest.py
# Analyzes attention patterns for EOS signals
```

### 4. Automated Batch Testing
```bash
# For systematic testing across multiple configurations
./autoTest.sh

# For large-scale parallel processing
./longRun.sh
```

### 5. Visualization Combination
```bash
python graphsCount.py
# Combines individual heatmaps into grid layouts
```

## Folder Structure

```
cur/signalObs/
├── README.md                          # This file
├── mapTest.py                         # Attention map generation for specific prompts
├── valueTest.py                       # Attention signal detection and EOS analysis
├── parameterRangeTest.py              # Parameter impact analysis on generation length
├── graphsCount.py                     # Visualization combination tool
├── AttentionMapTest.ipynb             # Interactive attention analysis notebook
├── attentionObs.ipynb                 # Attention observation experiments
├── autoTest.sh                        # Automated testing script
├── longRun.sh                         # Large-scale parallel processing script
└── attached/                          # Additional analysis tools
    ├── datasetTest.py                 # Dataset processing for attention analysis
    └── datasetTestMac.py              # Mac-specific dataset processing
```

## Supported Models

The system supports analysis of the following models:
1. **Llama-3.2-1B** - Lightweight model for quick testing (16 layers, 32 heads)
2. **Meta-Llama-3-8B** - Standard Llama-3 model
3. **Meta-Llama-3-70B** - Large-scale Llama-3 model
4. **Meta-Llama-3-8B-Instruct** - Instruction-tuned variant (used in TRAIL experiments)
5. **GPT-J-6B** - EleutherAI's GPT-J model
6. **Llama-2-13B-Chat** - Fine-tuned Llama-2 model
7. **GPT-NeoX-20B** - EleutherAI's GPT-NeoX model

## Supported Datasets

1. **Alpaca Dataset** - Standard instruction-following dataset
2. **Simplified Alpaca** - Reduced version for faster testing
3. **LMSYS Chat 1M** - Large-scale conversation dataset

## Analysis Types

### 1. Layer-wise Analysis
- Tracks attention patterns across different transformer layers
- Identifies how attention evolves from shallow to deep layers
- Generates heatmaps showing layer-specific attention distributions

### 2. Head-wise Analysis
- Analyzes attention patterns across different attention heads
- Identifies head specialization and attention focus patterns
- Creates visualizations for head-specific behaviors

### 3. Token-wise Analysis
- Tracks attention evolution as new tokens are generated
- Monitors attention shifts throughout the generation process
- Identifies patterns that might indicate approaching EOS

### 4. Parameter Impact Analysis
- Studies how decoding parameters affect attention patterns
- Tests ranges of temperature, top_k, and repetition_penalty values
- Correlates parameter settings with generation length and quality

## Observations

### Basic Introduction

**Attention Map Structure**:
- Attention maps are matrices of size $(num_{layers}, num_{heads}, num_{tokens}, num_{tokens})$
- Each element $(l, h, i, j)$ represents the attention score of layer $l$, head $h$, from token $i$ to token $j$
- Attention scores are calculated by: $score = softmax(Q \cdot K^T/\sqrt{d})$

**Key Considerations**:
- During decoding, each token has specific attention distributions to all previous tokens
- Attention patterns can reveal model behavior and potential EOS signals
- Raw attention data provides insights into model decision-making processes

### Heads and Layers

Heads and layers are the two **main hyperparameters** of the attention mechanism.

> **Heads**: The number of heads determines the number of parallel attention mechanisms.

- Various mechanisms: $MHA,\ MQA,\ GQA$, etc.
- Diverse focus: Each head, for instace, in MHA, focuses on different subspaces (parts) of the input sequence. For example, one head might focus on syntactic dependencies (like subject-verb agreement), while another might focus on semantic relationships (such as noun-adjective associations).
- Separate Projections: Each head has its own query, key, and value matrices, enabling it to project inputs into different subspaces and compute attention independently.
- Combination of Outputs: After processing, the outputs of all heads are concatenated and passed through a linear layer to form the final attention output for that layer.

> **Layers**: The number of layers determines the depth of the model.

- Hierarchical Abstraction: Each layer builds upon the previous one, progressively capturing more complex features. Early layers often handle simpler relationships (like word-level interactions), while deeper layers integrate this information to understand higher-level abstractions (such as sentence meaning or context).
- Sequential Processing: The output of one layer serves as the input to the next, allowing the model to refine and enhance representations across layers.

*Owing to various head mechanism, we'll mainly discuss the layer things.*

### Key Findings

> **Experimental Settings**:
> - Multiple models: Llama-3.2-1B to GPT-NeoX-20B
> - Multiple datasets: Alpaca, Simplified Alpaca, LMSYS Chat 1M
> - Systematic parameter testing across temperature, top_k, and repetition_penalty

> **Current Capabilities**:
> - Comprehensive attention map generation and visualization
> - Automated batch processing for large-scale analysis
> - Parameter impact analysis on generation behavior
> - Attention signal detection for potential EOS prediction

**Attention Pattern Observations**:

1. **Local Attention Dominance**: For most layer-head combinations, attention scores are highest for nearby tokens. The pattern $Q_i K_{j \rightarrow i}^T$ shows peak values when $j$ is close to $i$, indicating strong local dependencies.

2. **Layer-Specific Behaviors**:
   - Early layers focus on local syntactic relationships
   - Middle layers capture semantic dependencies
   - Later layers show reduced attention magnitudes (consistent with layer-skip research)

3. **Head Specialization**: Different attention heads within the same layer exhibit distinct attention patterns:
   - Some heads focus on recent context (recency bias)
   - Others maintain broader attention distributions
   - Certain heads show potential EOS-predictive behaviors

4. **Generation Phase Patterns**:
   - Attention patterns evolve throughout generation
   - Tail attention analysis reveals potential EOS signals
   - Parameter settings significantly impact attention distributions

5. **Parameter Impact**:
   - **Temperature**: Higher values increase attention diversity
   - **Top-k**: Affects the breadth of attention distributions
   - **Repetition Penalty**: Influences attention to previously generated tokens

## Automation Scripts

### `autoTest.sh` - Systematic Testing
**Purpose**: Automated testing across multiple parameter combinations for systematic analysis.

**Features**:
- Tests combinations of models, datasets, and prompts
- Configurable parameter ranges
- Error handling and status reporting
- Sequential execution with timing information

**Usage**:
```bash
./autoTest.sh
# Tests predefined combinations automatically
```

### `longRun.sh` - Large-Scale Parallel Processing
**Purpose**: High-throughput analysis using GNU Parallel for extensive experiments.

**Features**:
- Parallel execution with configurable job limits
- Timeout handling for long-running tasks
- Progress tracking and ETA estimation
- Resume capability for failed jobs
- Comprehensive logging

**Usage**:
```bash
./longRun.sh
# Runs large-scale parallel analysis
```

## Output Structure

The analysis generates organized outputs in the following structure:

```
../images/maps_final/[model]_[dataset]_prompt_[id]/
├── generation_info.txt                # Prompt, output, and metadata
├── heatmap_layer_[L]_head_[H].png    # Individual attention heatmaps
└── [combined_visualizations].png     # Grid layouts of all heatmaps

../scores/[model]/
├── ds_[dataset]_p_[prompt]_gen_info.txt  # Generation info with attention alerts
└── [analysis_results].txt               # Detailed attention analysis

../.output/prompt_[id]/
├── exp1_params_[value].txt           # Parameter experiment results
├── exp2_params_[value].txt           # Temperature experiment results
├── exp3_params_[value].txt           # Top-k experiment results
└── [experiment]_graph.png            # Parameter vs. length plots
```

## Decoding Parameters Reference

### 1. **Temperature** $T$
Controls randomness in token selection:
- $P_i = \frac{e^{z_i/T}}{\sum_{j}e^{z_j/T}}$ where $z_i$ is the logit for token $i$
- **High $T$ (> 1.0)**: More creative, diverse, less deterministic output
- **Low $T$ (< 1.0)**: More conservative, predictable output
- **$T = 0$**: Greedy sampling (always most probable token)

### 2. **Top-k Sampling**
Limits token selection to top-k most probable tokens:
- **$k = 1$**: Greedy sampling
- **$k > 1$**: Random sampling from top-k candidates
- **Effect**: Controls diversity vs. quality trade-off

### 3. **Repetition Penalty**
Penalizes repeated tokens to encourage diversity:
- **Penalty > 1.0**: Reduces repetition, increases diversity
- **Penalty = 1.0**: No penalty applied
- **Effect**: Balances coherence and repetition avoidance

### 4. **Max New Tokens**
Controls maximum generation length:
- **Longer limits**: More detailed responses, potential coherence loss
- **Shorter limits**: Concise responses, potential information cutoff

## Dependencies

- **PyTorch**: Neural network operations and model loading
- **Transformers**: HuggingFace model and tokenizer support
- **Datasets**: Dataset loading and processing
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Visualization and plotting
- **PIL (Pillow)**: Image processing for heatmap combination
- **GNU Parallel**: Large-scale parallel processing (for longRun.sh)
