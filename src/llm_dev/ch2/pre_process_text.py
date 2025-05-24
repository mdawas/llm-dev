import re


def split_text_to_tokens(text: str) -> str:
    pattern = r'([,.:;?_!"()\']|--|\s)'
    preprocessed = re.split(pattern, text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print("Number of tokens:", len(preprocessed))
    return preprocessed


def create_vocab(text):
    all_words = sorted(set(split_text_to_tokens(text)))
    print("The generated vocabulary size:", len(all_words))
    vocab = {word: i for i, word in enumerate(all_words)}
    return vocab


def create_vocab_v2(text):
    all_words = sorted(list(set(split_text_to_tokens(text))))
    # Add <|unk|> and <|endoftext|> tokens to the vocabulary
    all_words.extend(["<|unk|>", "<|endoftext|>"])
    print("The generated vocabulary size:", len(all_words))
    vocab = {word: i for i, word in enumerate(all_words)}
    return vocab
