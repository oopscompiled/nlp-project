import re
import html
from bs4 import BeautifulSoup

def clean_for_bert(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    
    text = html.unescape(text)
    
    text = BeautifulSoup(text, "html.parser").get_text()
    
    text = re.sub(r'https?://\S+', '', text)
    
    text = re.sub(r'\b(http|href)\b', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'&\w+;', '', text)
    text = re.sub(r'[@#]\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

word_re = re.compile(r"\b[a-zA-Z]{2,}\b", re.IGNORECASE)

def extract_clean_words(text):
    return word_re.findall(text)