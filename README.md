# \<EoS\> Prediction Test

@`SleepyLGod` for current observation 📂 /cur

## Project Overview

This project implements and evaluates a comprehensive end-of-sequence (EoS) prediction system for large language models (LLMs). The system explores multiple approaches to predict how many tokens remain in a generation sequence at any given step, enabling more efficient text generation, resource planning, and scheduling optimization.

The project encompasses three main methodologies:
1. **Vector Database Approach**: Pre-generation length prediction using prompt embeddings
2. **Neural Network Prediction**: Training specialized models on LLM hidden states and attention patterns
3. **Prompt Engineering**: Using specialized prompts to elicit length predictions from LLMs directly

The system provides comprehensive tools for data generation, model training, evaluation, and analysis across multiple model architectures and datasets.

## Key Components

### 1. **Data Generation Pipeline**
- **`dsGenEbd.py`** - Generates embedding-based training data from LLM hidden states
- **`dsGenLogitsAS.py`** - Generates logits-based training data with attention sampling
- **`dsGenLogitsBS.py`** - Generates logits-based training data with beam search
- **`dsGenLogitsNC.py`** - Generates logits-based training data with nucleus sampling
- **`dsGenDraft.py`** - Generates draft attention-based training data
- **`datasetInit.ipynb`** - Dataset initialization and preprocessing notebook

### 2. **Length Predictor Module** (`/cur/predictors/`)
- **Neural network training** (`ebdModelGenFinal.py`) - Enhanced MLP training on embeddings
- **Model evaluation framework** (`resultTest.py`) - Comprehensive evaluation pipeline
- **Results analysis and visualization** (`graphsModified.py`) - Statistical analysis and plotting
- **Error profile analysis** (`errorProfileGen.py`) - Step-0 error analysis and profiling
- **Error evolution tracking** (`errorRatioEvol.py`) - Error ratio evolution throughout generation
- **Parameter optimization** (`sysParaRank.py`) - Decoding parameter ranking and optimization

### 3. **Signal Observation** (`/cur/signalObs/`)
- **Attention map analysis** for understanding LLM behavior patterns
- **Parameter impact studies** on generation length and quality
- **EOS signal detection** through attention pattern analysis
- **Automated testing frameworks** for large-scale analysis

### 4. **Prompt Engineering Tests** (`/cur/PETest.py`)
- Experiments with different prompting strategies for length prediction
- Multi-model evaluation of prompt-based length prediction accuracy

### 5. **Model Response Analysis** (`/cur/preview/`)
- Pre-generated responses from various models for analysis
- Prompt engineering response collections
- Model comparison datasets

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

### Complete Workflow

#### 1. Data Generation
```bash
cd cur

# Generate embedding-based training data
python dsGenEbd.py

# Generate logits-based training data (choose one approach)
python dsGenLogitsAS.py  # Attention sampling
python dsGenLogitsBS.py  # Beam search
python dsGenLogitsNC.py  # Nucleus sampling

# Generate attention-based training data
python dsGenDraft.py
```

#### 2. Model Training and Evaluation
```bash
cd cur/predictors

# Train the length predictor
python ebdModelGenFinal.py

# Evaluate the trained model
python resultTest.py

# Generate comprehensive analysis
python graphsModified.py
python errorProfileGen.py
python errorRatioEvol.py
python sysParaRank.py
```

#### 3. Attention Analysis
```bash
cd cur/signalObs

# Single attention map analysis
python mapTest.py

# Parameter impact testing
python parameterRangeTest.py

# Attention signal detection
python valueTest.py

# Automated batch testing
./autoTest.sh
./longRun.sh
```

#### 4. Prompt Engineering Testing
```bash
cd cur

# Test prompt-based length prediction
python PETest.py
```

## Project Structure

```
/cur
├── predictors/                        # Length prediction module
│   ├── ebdModelGenFinal.py           # Neural network training
│   ├── resultTest.py                 # Model evaluation framework
│   ├── graphsModified.py             # Results visualization and analysis
│   ├── errorProfileGen.py            # Error profile analysis
│   ├── errorRatioEvol.py             # Error evolution tracking
│   ├── sysParaRank.py                # Parameter optimization
│   ├── idChecking.py                 # Utility for ID validation
│   ├── used_prompt_ids.txt           # Training data exclusion list
│   ├── saved_models/                 # Trained model checkpoints
│   ├── results/                      # Analysis results
│   ├── eval_output/                  # Advanced analysis outputs
│   ├── logs/                         # Training logs
│   └── README.md                     # Module documentation
│
├── signalObs/                        # Attention map observation
│   ├── mapTest.py                    # Attention map generation
│   ├── valueTest.py                  # Attention signal detection
│   ├── parameterRangeTest.py         # Parameter impact analysis
│   ├── graphsCount.py                # Visualization combination
│   ├── AttentionMapTest.ipynb        # Interactive analysis notebook
│   ├── attentionObs.ipynb            # Attention observation experiments
│   ├── autoTest.sh                   # Automated testing script
│   ├── longRun.sh                    # Large-scale parallel processing
│   ├── attached/                     # Additional analysis tools
│   └── README.md                     # Observation documentation
│
├── data/                             # Dataset files
│   ├── dataset_alpaca.json           # Alpaca dataset
│   ├── datasetSimplified_alpaca.json # Simplified Alpaca dataset
│   └── dataset_lmsys-chat-1m.json    # LMSYS Chat 1M dataset
│
├── training_data/                    # Generated training data
│   ├── ebd/                          # Embedding-based training data
│   │   ├── features/                 # Feature files (.npz)
│   │   └── metadata/                 # Metadata files
│   └── metadata/                     # Model-specific metadata
│
├── preview/                          # Model response analysis
│   ├── [model]_responses.json        # Pre-generated model responses
│   ├── dsTest/                       # Dataset testing responses
│   └── pe/                           # Prompt engineering responses
│
├── dsGenEbd.py                       # Embedding-based data generation
├── dsGenLogitsAS.py                  # Logits data generation (attention sampling)
├── dsGenLogitsBS.py                  # Logits data generation (beam search)
├── dsGenLogitsNC.py                  # Logits data generation (nucleus sampling)
├── dsGenDraft.py                     # Draft attention data generation
├── PETest.py                         # Prompt engineering tests
├── labelChecking.py                  # Label validation utility
├── datasetInit.ipynb                 # Dataset initialization notebook
├── IdeaList.md                       # Project ideas and concepts
└── requirements.txt                  # Project dependencies
```

## Methodologies and Approaches

### 1. **Neural Network Prediction** (Primary Approach)
**Training Data Generation**:
- **Embedding-based** (`dsGenEbd.py`): Extracts LLM hidden states during generation
- **Logits-based** (`dsGenLogitsAS/BS/NC.py`): Captures output distributions with different sampling strategies
- **Attention-based** (`dsGenDraft.py`): Records attention patterns across layers and heads

**Model Architecture**:
- Enhanced MLP with residual connections, batch normalization, and dropout
- 8192-dimensional input features from LLM embeddings
- Multi-GPU training with automatic mixed precision
- Advanced optimization with cosine annealing and gradient clipping

**Evaluation Pipeline**:
- Real-time prediction during actual LLM generation
- Comprehensive error analysis and parameter optimization
- Statistical analysis with error evolution tracking

### 2. **Attention Pattern Analysis**
**Signal Detection**:
- Monitors attention patterns that might indicate approaching EOS
- Analyzes layer-wise and head-wise attention behaviors
- Detects tail attention concentration as potential EOS signals

**Systematic Analysis**:
- Multi-model attention map generation and visualization
- Parameter impact studies on attention distributions
- Automated batch processing for large-scale analysis

### 3. **Prompt Engineering Approach**
**Direct Length Prediction**:
- Uses specialized prompts to elicit length predictions from LLMs
- Tests across multiple model architectures and sizes
- Evaluates accuracy and impact on generation quality

**Findings**:
- Accuracy typically below 0.5 across most models
- Larger models show improved accuracy but still limited
- Can negatively impact generation quality, especially for smaller models

## Supported Datasets

### Training and Evaluation Datasets
- **Alpaca Dataset** (`yahma/alpaca-cleaned`) - Standard instruction-following dataset
- **Databricks Dolly** - High-quality instruction dataset
- **LMSYS Chat-1M** - Large-scale conversation dataset
- **Simplified Alpaca** - Reduced version for faster testing and development

### Dataset Processing
- Automatic prompt formatting with instruction and input fields
- Response length labeling for training data generation
- Train/test splits with prompt ID exclusion to prevent data leakage

## Supported Models

### Primary Models (Extensively Tested)
- **Meta-Llama-3-8B** - Standard 8B parameter model
- **Meta-Llama-3-70B** - Large-scale 70B parameter model
- **Meta-Llama-3-8B-Instruct** - Instruction-tuned variant
- **Llama-3.2-1B** - Lightweight model for quick testing

### Additional Models (Attention Analysis)
- **Llama-2-13B-Chat** - Fine-tuned Llama-2 model
- **GPT-J-6B** - EleutherAI's GPT-J model
- **GPT-NeoX-20B** - EleutherAI's GPT-NeoX model
- **DeepSeek-R1 Series** - DeepSeek-R1-Distill models (1.5B, 8B, 14B, 32B, 70B)
- **GPT-3 Finnish Models** - Finnish language variants

### Model Architecture Support
- Supports models with different attention mechanisms (MHA, MQA, GQA)
- Configurable for various layer counts and attention head configurations
- Automatic model architecture detection and adaptation

## Key Features

### Data Generation
- **Multi-Strategy Sampling**: Attention sampling, beam search, nucleus sampling
- **Comprehensive Feature Extraction**: Hidden states, attention patterns, logits
- **Efficient Storage**: Compressed .npz format with batch processing
- **Parameter Sweep**: Systematic testing across temperature, top_k, repetition_penalty

### Model Training
- **Enhanced MLP Architecture**: Residual connections, batch normalization, dropout
- **Multi-GPU Support**: Distributed training with automatic mixed precision
- **Robust Data Loading**: Error recovery and memory-efficient batch processing
- **Advanced Optimization**: AdamW with cosine annealing and gradient clipping

### Analysis and Evaluation
- **Real-Time Prediction**: Step-by-step length prediction during generation
- **Comprehensive Metrics**: MAE, RMSE, R², bias, error ratios
- **Error Evolution Tracking**: How prediction errors change throughout generation
- **Parameter Optimization**: Ranking of decoding parameter combinations
- **Attention Signal Detection**: Identification of EOS-predictive attention patterns

## Research Findings

### Neural Network Approach
- **Embedding-based prediction** shows promising results for remaining token prediction
- **Step-0 predictions** (immediately after prompt prefilling) are critical for early stopping decisions
- **Parameter combinations** significantly impact prediction accuracy and generation quality
- **Error evolution** patterns provide insights into model confidence throughout generation

### Attention Pattern Analysis
- **Local attention dominance**: Most attention focuses on nearby tokens
- **Layer-specific behaviors**: Early layers handle syntax, deeper layers capture semantics
- **Head specialization**: Different heads exhibit distinct attention patterns
- **Tail attention concentration**: Potential signal for approaching EOS

### Prompt Engineering Limitations
- **Low accuracy**: Typically below 50% across most models
- **Generation quality impact**: Can negatively affect output quality, especially for smaller models
- **Model size dependency**: Larger models show better accuracy but still limited effectiveness

## Applications and Use Cases

### Resource Planning and Scheduling
- **Early length estimation** for compute resource allocation
- **Dynamic scheduling** based on predicted generation length
- **Memory management** optimization for batch processing

### Generation Optimization
- **Early stopping** based on length predictions
- **Parameter tuning** for desired output lengths
- **Quality vs. efficiency** trade-offs

### Model Analysis and Understanding
- **Attention pattern visualization** for model interpretability
- **Generation behavior analysis** across different parameter settings
- **Model comparison** through attention and prediction patterns

## Future Directions

### Methodology Improvements
- **Uncertainty quantification** for prediction confidence estimation
- **Multi-stage prediction** combining vector DB, neural networks, and attention analysis
- **Adaptive sampling** based on prediction confidence

### Technical Enhancements
- **Real-time inference optimization** for production deployment
- **Model architecture exploration** for better prediction accuracy
- **Cross-model generalization** for universal length predictors

### Research Extensions
- **Multi-language support** beyond English datasets
- **Domain-specific adaptation** for specialized use cases
- **Integration with existing LLM serving frameworks**
