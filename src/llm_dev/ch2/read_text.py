from pathlib import Path


def read_text_file(file_path: str) -> str:
    file_path = Path(file_path)
    if not file_path.is_file:
        raise FileNotFoundError
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    return raw_text
