import re

def remove_emojis(data):
    return re.sub(r'\p{Emoji}', ' ', data)

def clean_data(data):
    data = remove_emojis(data)
    data = data.lower()

    data = re.sub(r"<.*?>", " ", data)              # Remove HTML tags
    data = re.sub(r"\s?@\w+", " ", data)            # Remove @mentions
    data = re.sub(r"https?://\S+|www\.\S+", " ", data)  # Remove URLs
    data = re.sub(r"[^\w\s]", " ", data)            # Remove punctuation/special chars
    data = " ".join(data.split())                   # Remove extra whitespace
    
    return data