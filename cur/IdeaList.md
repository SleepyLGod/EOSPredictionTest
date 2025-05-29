# Idea List about Length Prediction

## Our Basic Idea

**3 stages of predictions**:
- 1st stage: using the vector db to predict the length of the response right before Prefilling.
  [store the initial embeddings the the prompts and the related response lengths(numbers/intervals)]
- 2nd and 3rd stages: 
  - under the bg of PD disaggregation
  - right after P, predict the draft length
  - during D, iteratively predict the length of the rest of the response and refine the prediction
  - prediction method: train a small model by using attention weights to predict

## Prompt Engineering Way

> Paper link: [Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline](https://openreview.net/forum?id=eW233GDOpm&noteId=oRk1g62H6s)
> From the paper, the accuracy is not such high, maybe about 0.5.

### Our Obeservations

**Settings:** 
- Using diffferent series of models (gpt-2,3,4 series; deepseek-r1 series; llama-2,3 series, with different sizes) and the alpaca dataset
- Changing the system parameters

**Observations:**
- The accuracy is really low, less than 0.5, no matter what model is used.
- Of course, the bigger the model is, the higher the accuracy is.
- Just following the paper's intructions of setting the adding prompt, 2 things happens:
  - keep on generating non-sense predictions: huge interval, non-sense sentenses, or even "".
  - impact the output generation as well, for many 8b models and smaller ones, the output is really bad, even "".

## Other discusssions by Yan
- Owing to the auto-regressive nature of the language model, the prediction cannot be such accurate.
- What about we admit the low accuracy utilizing the uncertainty of the prediction? (e.g. just like the filter in db?)
  say, we can also give a region of the prediction uncertainty:
    - the region with high uncertainty is directly sent to the scheduler and allocate somehow more resources to it.
    - the region with low uncertainty can utilize the traditional scheduling strategies like chunked prefill, etc.
    - The uncertainty can be measured by the entropy of the prediction, the higher the entropy, the higher the uncertainty.
- Of course, the 1st stage prediction (using vector db) can also do in the uncertainty way, like choosing the top-k predictions and then do the scheduling.