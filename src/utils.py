import regex as re

def remove_emojis(data):
    return re.sub(r'\p{Emoji}', ' ', data)

def clean_data(data):
    if not isinstance(data, str):
        return ""

    data = remove_emojis(data)
    data = data.lower()

    data = re.sub(r"<.*?>", " ", data)
    data = re.sub(r"\s?@\w+", " ", data)
    data = re.sub(r"https?://[^\s|]+|www\.[^\s|]+", " ", data)  # 👈 улучшено!
    data = re.sub(r"\|\|+|[-_=~]{2,}", " ", data)
    data = re.sub(r"[^\w\s]", " ", data)
    data = " ".join(data.split())

    return data