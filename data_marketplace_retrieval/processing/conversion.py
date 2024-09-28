import re
from urllib.parse import urlparse


def is_absolute(url):
    return bool(urlparse(url).netloc)


def get_first_content_by_type(jsarr, t):
    for block in jsarr:
        if block is not None and block['type'] == t:
            return block['content']
    return None


def get_all_content_by_type(jsarr, t, field='content'):
    strings = [c[field] for c in jsarr if c is not None and c['type'] == t and field in c and c[field] is not None]
    return ' '.join(strings) if strings else None


def get_category(url):
    if not url:
        return "uncategorized"
    if is_absolute(url):
        return url.split('/')[3]
    else:
        return url.split('/')[1]


def get_raw_text(jsarr, type: str = "all"):
    if type == "first":
        text = get_first_content_by_type(jsarr, 'sanitized_html')
    else:
        text = get_all_content_by_type(jsarr, 'sanitized_html')
    if text:
        text = re.sub('<.*?>', ' ', text)
    return text or ''


def get_doc_dict(js):
    return {
        "title": get_all_content_by_type(js['contents'], 'title'),
        "date": get_first_content_by_type(js['contents'], 'date'),
        "kicker": get_first_content_by_type(js['contents'], 'kicker'),
        "author": js['author'],
        "raw_text": get_raw_text(js['contents'], type="all"),
        "first_paragraph": get_raw_text(js['contents'], type="first"),
        "url": js['article_url'],
        "category": get_category(js['article_url']),
        "id": js['id']
    }
