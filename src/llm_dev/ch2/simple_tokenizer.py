from llm_dev.ch2.pre_process_text import split_text_to_tokens
import re


# listing 2.3 A simple text tokenizer.
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = split_text_to_tokens(text)
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)
        return text


# listing 2.4 A simple text tokenizer that handles unknown words.
class SimpleTokenizerV2(SimpleTokenizerV1):
    def __init__(self, vocab):
        super().__init__(vocab)

    def encode(self, text):
        preprocessed = split_text_to_tokens(text)
        preprocessed = [s if s in self.str_to_int else "<|unk|>" for s in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
