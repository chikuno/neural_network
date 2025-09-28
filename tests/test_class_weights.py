import torch
from train.class_weights import compute_class_weights

class DummySeqs:
    def __init__(self, seqs):
        self.sequences = seqs


def test_class_weights_length_matches_vocab():
    train = [[0,1,2,2,3],[3,3,4]]
    val = [[1,2,4]]
    vocab_size = 6
    w = compute_class_weights(train, val, vocab_size, power=0.5)
    assert w.shape[0] == vocab_size


def test_class_weights_nonzero():
    train = [[0,0,0,1]]
    val = []
    vocab_size = 5
    w = compute_class_weights(train, val, vocab_size, power=0.5)
    assert torch.all(w > 0)


def test_class_weights_power_effect():
    train = [[0,0,0,1,2,2,3,4]]
    val = []
    vocab_size = 5
    w_sqrt = compute_class_weights(train, val, vocab_size, power=0.5)
    w_linear = compute_class_weights(train, val, vocab_size, power=1.0)
    # Higher power should exaggerate differences: variance larger or equal
    assert w_linear.var() - w_sqrt.var() >= -1e-6


def test_empty_sequences():
    train = []
    val = []
    vocab_size = 4
    w = compute_class_weights(train, val, vocab_size)
    assert w.shape[0] == vocab_size
    # All equal when no data
    assert torch.isclose(w.std(), torch.tensor(0.), atol=1e-6)
