import torch
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, utils

utils.logging.set_verbosity_error()  # Suppress standard warnings
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')
model.eval()  # Disable dropout (optional)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

# Step 1: Load and Tokenize the Dataset
def tokenize_dataset(dataset, tokenizer, max_length=512):
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples['instruction'],
                                    truncation=True,
                                    padding='max_length',
                                    max_length=max_length),
        batched=True
    )
    return tokenized_dataset

# Load Alpaca dataset
ds = load_dataset("tatsu-lab/alpaca")
small_dataset = ds['train'].select(range(50))  # Adjust range as needed
tokenized_dataset = tokenize_dataset(small_dataset, tokenizer)

device = 'cpu'

def get_attention_scores(model, tokenized_dataset, device='cpu'):
    model.eval()
    model.to(device)
    attention_scores = []
    batch_size = 2  # Adjust based on CPU memory
    for i in range(0, len(tokenized_dataset), batch_size):
        batch = tokenized_dataset[i:i+batch_size]
        input_ids = torch.tensor(batch['input_ids']).to(device)
        attention_mask = torch.tensor(batch['attention_mask']).to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            batch_attentions = outputs.attentions  # List of layers, each is (batch, heads, seq, seq)
            attention_scores.append(batch_attentions)
    return attention_scores

attention_scores = get_attention_scores(model, tokenized_dataset, device)

# Step 4: Calculate Metrics
def calculate_metrics(attention_scores):
    # Flatten the list of attention scores
    flattened_scores = [score for batch in attention_scores for score in batch]
    
    # Find the maximum sequence length
    max_seq_len = max(score.shape[-1] for score in flattened_scores)
    
    # Pad all attention scores to have the same sequence length
    padded_scores = []
    for score in flattened_scores:
        padding = ((0, 0), (0, 0), (0, 0), (0, max_seq_len - score.shape[-1]))
        padded_score = np.pad(score, padding, mode='constant')
        padded_scores.append(padded_score)
    
    # Convert list of padded scores to numpy array
    attention_scores_np = np.array(padded_scores)
    
    # Calculate average attention score per head across all layers and positions
    avg_per_head = np.mean(attention_scores_np, axis=(0, 2, 3))
    
    # Calculate variance of attention scores per layer across all heads and positions
    var_per_layer = np.var(attention_scores_np, axis=(1, 3, 4))
    
    # Calculate entropy of attention distributions per head and layer
    epsilon = 1e-10
    entropy_per_head_layer = -np.sum(attention_scores_np * np.log(attention_scores_np + epsilon), axis=-1)
    
    # Calculate correlation between attention scores of different heads within the same layer
    correlation_matrix = np.zeros((model.config.num_layers, model.config.num_attention_heads, model.config.num_attention_heads))
    for layer in range(model.config.num_layers):
        for head1 in range(model.config.num_attention_heads):
            for head2 in range(model.config.num_attention_heads):
                flat_head1 = attention_scores_np[:, layer, head1].flatten()
                flat_head2 = attention_scores_np[:, layer, head2].flatten()
                corr = np.corrcoef(flat_head1, flat_head2)[0, 1]
                correlation_matrix[layer, head1, head2] = corr
    
    return {
        'avg_per_head': avg_per_head,
        'var_per_layer': var_per_layer,
        'entropy_per_head_layer': entropy_per_head_layer,
        'correlation_matrix': correlation_matrix
    }

metrics = calculate_metrics(attention_scores)

# Step 5: Visualize Metrics
def plot_metrics(metrics):
    plt.use('TkAgg')  # Set backend for macOS
    import matplotlib.pyplot as plt
    
    # Average attention score per head
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['avg_per_head'])
    plt.title('Average Attention Score per Head')
    plt.xlabel('Head Number')
    plt.ylabel('Average Attention Score')
    plt.savefig('../../images/avg_attention_per_head.png')
    plt.show()
    
    # Variance of attention scores per layer
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['var_per_layer'])
    plt.title('Variance of Attention Scores per Layer')
    plt.xlabel('Layer Number')
    plt.ylabel('Variance')
    plt.savefig('../../images/variance_per_layer.png')
    plt.show()
    
    # Entropy of attention distributions per head and layer
    plt.figure(figsize=(12, 8))
    for layer in range(model.config.num_layers):
        plt.subplot(4, 8, layer + 1)
        plt.imshow(metrics['entropy_per_head_layer'][layer])
        plt.title(f'Entropy - Layer {layer}')
        plt.colorbar()
    plt.tight_layout()
    plt.savefig('../../images/entropy_per_head_layer.png')
    plt.show()
    
    # Correlation between attention scores of different heads within the same layer
    plt.figure(figsize=(12, 8))
    for layer in range(model.config.num_layers):
        plt.subplot(4, 8, layer + 1)
        plt.imshow(metrics['correlation_matrix'][layer])
        plt.title(f'Correlation - Layer {layer}')
        plt.colorbar()
    plt.tight_layout()
    plt.savefig('../../images/correlation_per_head.png')
    plt.show()

plot_metrics(metrics)