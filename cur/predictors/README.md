# README

3 Phases of prediction:
- 1st: After the token list was converted into embeddings and before they are fed into the model, we pass the embeddings through a vector db to get the stored history length(s) prediction.
- 2nd: After Prefilling stage, we pass the attention weights through a lightweight model to get the prediction of the length again.
- 3rd (can be ignored): During Decoding stage, we iteratively pass the attention weights through the model to get the prediction of the length.

Side ideas:
- under the PD disaggregation structure, so the 1st and 2nd phase predictions are good tools to do the resource allocation and scheduling.
- 3rd phase is not necessary, but it can be used to check the ending of the decoding stage.
- all the attention weights ideas are basically based on the traditional transformer-based models, not MoE structures.
- <EOS> signals?

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

=> so what about eliminate the 1st phase? just use the last token in the P phase to predict the length?