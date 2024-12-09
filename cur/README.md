# AttentionMap Obeservation

✍️ @SleepyLGod

## Running Config

```bash
# python >= 3.9.7
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Obersevations

### Basic Intro

- Basically, the attention map is a matrix of size $(num_{heads}, num_{tokens}, num_{tokens})$ where each element $(i, j, k)$ represents the attention score of the i-th head of the k-th token to the j-th token.
- Attention scores are calculated by the dot product of the query and key vectors of the attention mechanism, i.e. $score = softmax(Q \cdot K^T/\sqrt{d})$. And thus, the map is not such **accurate**, in the reason that in the decoding phase, each token has a specific scores (distribution but not the exact value) to all the tokens in the input sequence, so we have to get the raw data by fine-tuning the model.

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

### Rough Observations

> **Experimental settings**:
>
> - Llama-3.2-8b Model
> - Alpaca Dataset (QA tasks chosen)
>
> **Current defects**:
>
> - The model has not been fine-tuned, so it will generate a short answer and repeat it, and will not generate an *end_of_sequence* token.
> - After improvements such as adding *top-k sampling*, *temperature mechanism*, and *repetition penalty*, the generation length has been limited, but the coherence and accuracy of the answer are not high, and fine-tuning is still needed to obtain a more accurate end_of_sequence token.

1. No matter what the number of layer/head is, for one particular $Q_i{K_{1\rightarrow{i}}}^T$ vector, the hightest score is often from $Q_i{K_{j\rightarrow{i}}}^T$ chunks, where $j$ is close to $i$. Of course, there are some exceptions that: *the attention scores are distributed to all the tokens in the input sequence*, **or** *one token has a high score to all the tokens in the input sequence*. But the former is more common.
2. The same observation as the meta's layer-skip paper, the attention scores of the last few layers is really low enough to be ignored.
3. From the start of the decoding phase to the generation of the 'end_of_sequence' token: seems no such trend?
