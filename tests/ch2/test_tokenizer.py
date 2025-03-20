from llm_dev.ch2.read_text import read_text_file
from llm_dev.ch2.simple_tokenizer import SimpleTokenizerV1, SimpleTokenizerV2
from llm_dev.ch2.pre_process_text import create_vocab, create_vocab_v2


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


def test_tokenize_training_text_v2():
    full_text = read_text_file("./the-verdict.txt")
    vocab = create_vocab_v2(full_text)
    sample_text = "Hello, do you like tea?"
    sample_text = " <|endoftext|> ".join(
        [sample_text, "In the sunlit terraces of the palace."]
    )
    tokenizer_v2 = SimpleTokenizerV2(vocab)
    ids_v2 = tokenizer_v2.encode(sample_text)
    decoded_v2 = tokenizer_v2.decode(ids_v2)
    assert (
        decoded_v2
        == "<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>."
    )
