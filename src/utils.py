# import regex as re

# def clean_data(data):
#     if not isinstance(data, str):
#         return ""

#     data = re.sub(r'https?://\S+', '', data)             # Удалить ссылки
#     data = re.sub(r'@\w+', '', data)                     # Удалить @юзернеймы
#     data = re.sub(r'\|\|+|[-_=~]{2,}', ' ', data)        # Удалить спец. символы
#     data = re.sub(r'<.*?>', '', data)                    # Удалить HTML-теги
#     data = re.sub(r'\p{Emoji}', '', data)                # Удалить эмодзи
#     data = re.sub(r'[^\w\s]', '', data)                  # Удалить всё лишнее кроме слов
#     data = " ".join(data.split())                        # Привести пробелы в порядок

#     return data

import regex as re
import html  # Для декодирования HTML сущностей

def clean_data(data):
    if not isinstance(data, str):
        return ""

    # Декодируем HTML-сущности (заменяет &gt;, &amp;, &lt; и другие на символы)
    data = html.unescape(data)

    # Удалить ссылки
    data = re.sub(r'https?://\S+', '', data)

    # Удалить @юзернеймы
    data = re.sub(r'@\w+', '', data)

    # Удалить спец. символы
    data = re.sub(r'\|\|+|[-_=~]{2,}', ' ', data)

    # Удалить HTML-теги
    data = re.sub(r'<.*?>', '', data)

    # Удалить эмодзи (если это необходимо)
    data = re.sub(r'\p{Emoji}', '', data)

    # Удалить всё лишнее кроме слов
    data = re.sub(r'[^\w\s]', '', data)

    # Привести пробелы в порядок (удалить лишние пробелы)
    data = " ".join(data.split())

    return data