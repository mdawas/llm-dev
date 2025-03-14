from read_text import read_text_file
from simple_tokenizer import SimpleTokenizerV1
from pre_process_text import create_vocab


def test_tokenize_training_text():
    full_text = read_text_file("./the-verdict.txt")
    vocab = create_vocab(full_text)
    tokenizer = SimpleTokenizerV1(vocab)
    sample_text = """
    "It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride.
    """
    ids = tokenizer.encode(sample_text)
    print(sample_text)
    print(ids)
    decoded = tokenizer.decode(ids)
    assert (
        decoded
        == '" It\' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
    )
