import os
import seqio
import argparse

import jax
import jax.numpy as jnp

import numpy as np

from perplexity_sampling import task
from perplexity_sampling import util
from perplexity_sampling import model

from tqdm import tqdm
from functools import partial
from jax_smi import initialise_tracking
initialise_tracking()

import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '0.95'

if __name__ == '__main__':

    import sentencepiece as spm

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--build_matrix", type=bool)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--seq_length", default=512, type=int)
    parser.add_argument("--save_interval", default=500, type=int)
    parser.add_argument("--checkpoint_prefix", default="checkpoint_", type=str)
    parser.add_argument("--vocab_path", type=str)
    parser.add_argument("--matrix_path", default=None, type=str)
    parser.add_argument("--cache_path", default=None, type=str)
    parser.add_argument("--ngram", default=5, type=int)
    args = parser.parse_args()


    if args.cache_dirs is not None:
        seqio.add_global_cache_dirs([args.cache_path])
        use_cached = True
    else:
        use_cached = False

    sequence_length = {"text": args.seq_length}
    dataset = seqio.get_mixture_or_task(args.task).get_dataset(
        sequence_length=sequence_length,
        split="train",
        shuffle=False,
        num_epochs=1,
        shard_info=seqio.ShardInfo(index=0, num_shards=10),
        use_cached=use_cached,
        seed=42
    )

    dataset = seqio.utils.trim_and_pad_dataset(dataset, sequence_length)
    dataset = dataset.batch(args.batch_size, drop_remainder=True)

    if args.vocab_path.startswith("gs://"):
        tokenizer = seqio.SentencePieceVocabulary(args.vocab_path).tokenizer
    else:
        tokenizer = spm.SentencePieceProcessor(model_file=args.vocab_path)

    k = args.ngram
    vocab_size = tokenizer.vocab_size()
    
    if args.build_matrix:

        matrix = util.init_matrix(vocab_size, k)
        increment_fn = jax.vmap(util.increment_at_coordinate, in_axes=[None, 0, None])

        pmap_get_k_ngram = jax.pmap(
            partial(util.get_k_ngrams, k=k),
            in_axes=(0)
        )

        total_tokens = 0
        for idx, ex in tqdm(enumerate(dataset.as_numpy_iterator())):

            seq = ex['text']
            _, seq_length = seq.shape

            num_devices = len(jax.devices())
            seq = seq.reshape(num_devices, -1, seq_length)

            total_tokens += seq[jnp.where(seq != 0)].shape[0]
            
            all_x = pmap_get_k_ngram(seq).reshape(-1, k)

            matrix = matrix + increment_fn(matrix, all_x, k).sum(0)

            if (idx > 0) and (idx%args.save_interval == 0):
                jnp.save("{}_{}.npy".format(args.checkpoint_prefix, idx), matrix)

        jnp.save("{}_{}_finished.npy".format(args.checkpoint_prefix, idx), matrix)

        tree = {}
        for count, indices in tqdm(zip(matrix.data, matrix.indices), total=len(matrix.data)):

            indices = indices.tolist()
            count = count.astype(int)

            _i = 0
            _tree = tree
            for idx in indices:
                # idx_name = str(idx)
                idx_name = int(idx)
                if idx_name not in _tree:
                    _tree[idx_name] = {}
                # else:
                #     _tree[idx_name] = {**_tree[idx_name], **{idx_name: {}}}
                _tree = _tree[idx_name]

                if (_i == len(indices)-1) or (indices[_i+1] == 0):
                    # tree[idx] = {**tree[idx], **{idx: count}}
                    if -1 in _tree:
                        _tree[-1] += int(count)
                    else:
                        _tree[-1] = int(count)
                    break
                _i += 1

        tree[0] = total_tokens
        jnp.save("{}_ngram_tree.npy".format(args.checkpoint_prefix), tree)

    else:

        import seaborn as sns
        from matplotlib import pyplot as plt

        matrix = jnp.load(args.matrix_path, allow_pickle=True).tolist()
        model = model.StupidBackoff(
            matrix=matrix,
            k=args.ngram,
            N=matrix[0],
            )

        all_seq_score = []
        for idx, ex in tqdm(enumerate(dataset.as_numpy_iterator()), total=len(list(dataset))):
            seq = ex['text']
            base = (seq != 0).sum(1).astype(jnp.float32)
            seq_score = jnp.power(10, -model.score(seq)/base).tolist()
            all_seq_score.extend(seq_score)

        sns.displot(all_seq_score, kind='kde', aspect=2)

        Q1, Q2, Q3 = np.quantile(all_seq_score, [0.25, 0.5, 0.75])
        plt.axvline(Q1, color='r')
        plt.axvline(Q2, color='r')
        plt.axvline(Q3, color='r')

        plt.savefig('score_distribution.png')

