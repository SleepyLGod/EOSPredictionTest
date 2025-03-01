# README

2 Phases of prediction:
- [0]: use outside model to predict the length (the most stupid BERT-based model idea).
  - OR: disaggregating the [tokens->embeddings, in the model] process and the prefill process, and thus we can predict by using rough embeddings to do the initial prediction.
- 1st: After Prefilling stage, where the token list of the prompt was converted into embeddings and before Decoding, we pass the final word's embedding through a vector db to get the stored (rough) history length prediction.
- 2nd: During Decoding, we pass the logits/final-layer embeddings through a lightweight model to get the prediction of the length again. (this one is like TRAIL ???) <= EOS signal sending machine.

Side ideas:
- under the PD disaggregation structure, so the 1st and 2nd phase predictions are good tools to do the resource allocation and scheduling.
- Can also use attention_weights to train the model, but all the attention weights ideas are basically based on the traditional transformer-based models, not MoE structures.
- **<EOS> signals ideas? need exploration.**

------
Progress:
- 1st phase: db selection
- 2nd&3rd phase: dataset preparation and model selection
  - dataset problems: 
    - toooo huge, and thus I compressed the data and randomly choose the (L,H) pairs.
    - keep the data that cannot meet the <EOS> token?
  - model problems: not trained yet.
- others:
  - is the IDC server ready?

**IMPORTANT**:
During inference:
- prompt->tokens (tokenization): in CPU; output: CPU tensors in RAM; no semantical information.
- tokens->embeddings (embedding layer): including mapping the token_ids to high-dimentional vectors (1st, by look-up table) and positional encoding (2nd); in GPU by default; output: GPU tensors in GPU memory; semantical information.
- attention layers...: in GPU

=> an no extra-cost way: as the prompt goes through the embedding layer and generate the embeddings (no attentions), we directly send it to the vector db to get the prediction result for D (at the same time, the model is doing P)

---------------------

Seems like the GPU-GPU connections are very fast (like NVLink)
For CPU-GPU connection, the speed can be improved by using CXL(u mentioned before), RDMA, or using DPU/shared memory to cache, but is still quite slower than the GPU-GPU things.

But I think the biggest point is that: the embeddings of prompt are huge, say for GPT-3, the size is seq_len\times 12288, and GPT-3 is small compared to current big models, especially MoE ones.

Of course, in our experiments, the model is not such big and the embedding size can somehow not be the bottleneck, but I personally think it is not a general solution (we r not arch people)

**we can also consider the embedding lookup part be  disaggregated on to another gpu**