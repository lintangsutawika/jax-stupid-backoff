import jax
import jax.numpy as jnp

import numpy as np

from functools import partial

from perplexity_sampling import util

@partial(jax.jit, static_argnames=['alpha', 'seq_length', 'k'])
def get_alpha_matrix(alpha, k, seq_length):

    exponent_matrix = jnp.expand_dims(jnp.arange(k),1).repeat(seq_length, 1)

    for i in jnp.arange(k):
        exponent_matrix = exponent_matrix.at[:, i].set(jnp.roll(exponent_matrix[:, i], -(i+1)))

    base_matrix = jnp.ones((k, seq_length)) * alpha

    return jnp.power(base_matrix, exponent_matrix)


class StupidBackoff:

    def __init__(
        self, matrix, k, N, alpha=0.4
        ):

        self.matrix = matrix
        self.k = k
        self.N = N
        self.alpha = alpha

        self.get_ngram_count = partial(util.get_ngram_count, matrix=matrix)

        fn_get_k_ngrams = partial(util.get_k_ngrams, k=k)
        self.pmap_pad_ngram_table = jax.pmap(
            fn_get_k_ngrams,
            in_axes=(0)
            )

        self.num_device = jax.device_count()


    def score(self, seq):

        k = self.k
        N = self.N
        alpha = self.alpha
        matrix = self.matrix
        
        bs, seq_length = seq.shape

        _seq = seq.reshape(self.num_device, -1, seq_length)
        padded_seq_ngrams = self.pmap_pad_ngram_table(_seq).reshape(-1, k)

        score_table = jnp.asarray([self.get_ngram_count(i) for i in np.asarray(padded_seq_ngrams)]).reshape(bs, k, -1)

        base = (seq != 0).sum(1).astype(jnp.int32)

        mask = jnp.asarray([jnp.concatenate([jnp.ones((k, sl)), jnp.zeros((k, seq_length-sl))],1) for sl in base.tolist()], dtype=jnp.int32)
        a = jnp.rot90(jnp.tril(jnp.ones((k,k)), k=0))
        b = jnp.ones((k, seq_length-k))
        ramp_mask = jnp.repeat(jnp.concatenate((a,b), 1).reshape(1, k, seq_length), bs, 0).astype(jnp.int32)
        mask = jnp.multiply(mask, ramp_mask)
        score_table = jnp.multiply(score_table, mask)

        denominator = jnp.roll(score_table, -1, 1)
        denominator = denominator.at[:,-1,:].set(N)
        denominator = jnp.multiply(denominator, mask)

        alpha_matrix = get_alpha_matrix(alpha, k, seq_length)
        alpha_matrix = jnp.multiply(alpha_matrix, mask)

        score_table = jnp.multiply(score_table, alpha_matrix)
        score_table = jnp.divide(score_table, denominator)

        rows = k - (jnp.isfinite(score_table) & (score_table > 0)).sum(1)
        cols = jnp.arange(seq_length).reshape(1,-1).repeat(bs,0)
        batch = jnp.arange(bs).reshape(-1,1).repeat(seq_length,1)

        score = score_table[batch, rows, cols]
        score = jnp.nan_to_num(score, nan=1.0, posinf=1.0, neginf=1.0)
        score = jnp.log10(score)

        return score.sum(1)

