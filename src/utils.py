# import regex as re

# def clean_data(data):
#     if not isinstance(data, str):
#         return ""

#     data = re.sub(r'https?://\S+', '', data)            
#     data = re.sub(r'@\w+', '', data)                    
#     data = re.sub(r'\|\|+|[-_=~]{2,}', ' ', data)      
#     data = re.sub(r'<.*?>', '', data)                    
#     data = re.sub(r'\p{Emoji}', '', data)               
#     data = re.sub(r'[^\w\s]', '', data)                 
#     data = " ".join(data.split())                     

#     return data

import regex as re
import html

def clean_data(data):
    if not isinstance(data, str):
        return ""

    data = data.str.lower()
    # decode html entities
    data = html.unescape(data)

    data = re.sub(r'https?://\S+', '', data)

    data = re.sub(r'@\w+', '', data)

    data = re.sub(r'\|\|+|[-_=~]{2,}', ' ', data)

    data = re.sub(r'<.*?>', '', data)

    data = re.sub(r'\p{Emoji}', '', data)

    data = re.sub(r'[^\w\s]', '', data)

    data = " ".join(data.split())

    return data

word_re = re.compile(r"\b[a-zA-Z]{3,}\b", re.IGNORECASE)

def extract_clean_words(text):
    return word_re.findall(text)