import re


def encode(text: str) -> str:
    pattern = r'([,.:;?_!"()\']|--|\s)'
    preprocessed = re.split(pattern, text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print("Number of tokens:", len(preprocessed))
    return preprocessed
