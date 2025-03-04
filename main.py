from read_text import read_text_file
from hand_tokenizer import tokenize_text


def main():
    print("Hello from llm-dev!")
    full_text = read_text_file("./the-verdict.txt")
    tokens = tokenize_text(full_text)
    print(tokens[:30])


if __name__ == "__main__":
    main()
