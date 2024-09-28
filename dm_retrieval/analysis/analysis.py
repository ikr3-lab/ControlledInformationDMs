import json
from urllib.parse import urlparse
import pandas as pd
import pprint


def is_absolute(url):
    return bool(urlparse(url).netloc)


def count_lines(path: str):
    lines = 0
    with open(path) as f:
        for line in f:
            lines += 1
    return lines


def get_categories(path: str):
    categories = dict()
    with open(path) as f:
        for i, line in enumerate(f):
            j_content = json.loads(line)
            if j_content['article_url']:
                if is_absolute(j_content['article_url']):
                    cat = j_content['article_url'].split('/')[3]
                else:
                    cat = j_content['article_url'].split('/')[1]
            else:
                cat = "uncategorized"
            categories[cat] = categories[cat] + 1 if cat in categories else 1

    categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
    categories = pd.DataFrame.from_dict(categories)
    categories.columns = ['Category', 'Count']
    return categories


def pprint_file(path: str, l: int = 1):
    with open(path) as f:
        for i, line in enumerate(f):
            if i == l:
                j_content = json.loads(line)
                pprint.pprint(j_content)
                break
