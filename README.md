<img src="./sophia.png" width="700px"></img>

# Sophia - JAX

JAX implementation of the [Sophia optimizer](https://arxiv.org/abs/2305.14342) for LLM pre-training.

Official PyTorch implementation is here: https://github.com/Liuhong99/Sophia

In the paper, Sophia is reported to be 2x faster than Adam on GPT-2. It's recently been battle-tested on large-scale runs and this speed-up was observed as well.


## TODO
- [ ] Reproduce pretraining results with GPT models
- [ ] Comparisons to AdamW, LION, etc.