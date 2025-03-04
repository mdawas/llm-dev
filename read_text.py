from pathlib import Path


def read_text_file(file_path: str) -> str:
    file_path = Path(file_path)
    if not file_path.is_file:
        raise FileNotFoundError
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total Number of character:", len(raw_text))
    print(raw_text[:99])
    return raw_text


read_text_file("./the-verdict.txt")
