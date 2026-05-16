import re


def clean_text(text: str) -> str:
    text = str(text).encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"^\s*RT\s+", "", text)
    text = re.sub(r"http\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()
