# JAX-based Implementation of Stupid Backoff

This repo documents a Jax-based implementation of the [Stupid Backoff algorithm](https://aclanthology.org/D07-1090.pdf) that was introduced as an alternative to Kneser-Ner Smoothing.

This is a execise I took to learn about how to use JAX. I thought it would be interesting to try.

## Implementation Notes

### Non-Recursive Calculation Method

While the original paper described the method is done recursively, this method doesn't work well with `jit`, `vmap`, and `pmap`. Instead, I opted to calculate all possible values as a grid of `k` by `seq_length` for each input sample. Having a finite limit of `k` (In this case `k=5` as in 5-grams) allows calculation of all backoff scores.

### Use Seqio to handle the tokens.

I use seqio as the library to help cache the text datasets. I also use a default T5 Tokenizer. The point here is to transform strings of text to tensors so that it can be processed with Jax.

## Limitations

I hit a wall in terms of efficiently calculating the score of a sentence due to the nature of the ngrams being stored in a nested dictionary. It works but it's not very efficient.

## How to play with it

1. Build the ngram table
2. Calculate the score