import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return text.replace("đ", "d")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize(text))
