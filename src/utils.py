import regex as re
import html

# no need in proper cleaning for bert

def clean_for_bert(text):
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = re.sub(r'https?://\S+', '', text)       
    text = re.sub(r'<.*?>', '', text)              
    text = re.sub(r'\s+', ' ', text).strip()

    return text


word_re = re.compile(r"\b[a-zA-Z0-9]{2,}\b", re.IGNORECASE)

def extract_clean_words(text):
    return word_re.findall(text)