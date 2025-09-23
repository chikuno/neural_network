from data import data


def test_preprocess_and_vocab():
    text = "Hello world! Hello AI."
    tokens = data.preprocess_data(text)
    assert 'hello' in tokens
    w2i, i2w = data.build_vocabulary(tokens, min_freq=1)
    assert '<PAD>' in w2i and '<UNK>' in w2i
    numerical = data.tokenize_and_numericalize(tokens, w2i)
    assert all(isinstance(n, int) for n in numerical)
