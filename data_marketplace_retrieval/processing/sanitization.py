from typing import Union
import numpy as np
import coreferee, spacy
import re


nlp = spacy.load('en_core_web_sm')
nlp_cor = spacy.load('en_core_web_sm')
nlp_cor.add_pipe('coreferee')


def sanitize_ent(sanit_type: str = "remove", ent=None):
    if sanit_type == "remove":
        return ""
    elif sanit_type == "mask":
        return "<mask>"
    elif sanit_type == "mask_alt":
        if "." not in ent["text"]:
            return f'<mask{ent["text"]}>'
        else:
            return ent["text"]
    elif sanit_type == "ent":
        return f'({ent.text}-{ent.label_})'


def get_random_ents(ents: list[dict] = None, seed: int = 42) -> list:
    np.random.seed(seed)
    ents_label = []
    for ent in ents:
        if ent['label'] not in ents_label:
            ents_label.append(ent['label'])
    if len(ents_label) > 0:
        num = np.random.randint(0, len(ents_label))
        return list(set(np.random.choice(list(ents_label), num)))
    else:
        return []


def ner_sanitize(ents_to_remove: list, text: str, sanit_type='remove'):
    ents = get_ents(text)
    ents = [ent for ent in ents if ent['label'] in ents_to_remove]

    pos = 0
    new_cont = ""
    for ent in ents:
        new_cont += text[pos:ent['start']]
        new_cont += sanitize_ent(sanit_type=sanit_type, ent=ent)
        pos = ent['end']
    new_cont += text[pos:]
    return new_cont, ents


def unmask(masked_text: str, masked_ents: list[dict], unmasker):
    inferred_count = 0
    mask_count = masked_text.count('<mask>')
    if mask_count > 1:
        preds_ents = unmasker(masked_text)
        for preds_ent, masked_ent in zip(preds_ents, masked_ents):
            pred_list = [pred_ent['token_str'].strip().lower() for pred_ent in preds_ent if pred_ent['score'] > 0.05]
            if masked_ent['text'].strip().lower() in pred_list:
                inferred_count += 1
    elif mask_count == 1:
        preds_ent = unmasker(masked_text)
        pred_list = [pred_ent['token_str'].strip() for pred_ent in preds_ent if pred_ent['score'] > 0.05]
        if masked_ents[0]['text'].strip() in pred_list:
            inferred_count += 1
    return inferred_count, mask_count

def unmask_alt(masked_text: str, unmasker):
    inferred_count = 0
    mask_count = masked_text.count('<mask')
    regex_find = r"<mask(.*?)>"
    masked_ents = re.findall(regex_find, masked_text)

    regex_sub = r'<mask[^>]+>'
    masked_text = re.sub(regex_sub, r'<mask>', masked_text)
    if mask_count > 1:
        preds_ents = unmasker(masked_text)
        for preds_ent, masked_ent in zip(preds_ents, masked_ents):
            pred_list = [pred_ent['token_str'].strip() for pred_ent in preds_ent if pred_ent['score'] > 0.05]
            if masked_ent.strip() in pred_list:
                inferred_count += 1
    elif mask_count == 1:
        preds_ent = unmasker(masked_text)
        pred_list = [pred_ent['token_str'].strip().lower() for pred_ent in preds_ent if pred_ent['score'] > 0.05]
        if masked_ents[0].strip().lower() in pred_list:
            inferred_count += 1
    return inferred_count, mask_count


def get_ents(text: str):
    ents = []
    for ent in nlp(text).ents:
        ents.append({
            'start': ent.start_char,
            'end': ent.end_char,
            'text': ent.text,
            'label': ent.label_
        })
    return ents


def coreferee(raw_text) -> str:
    docnlp = nlp_cor(raw_text)
    new_text = ""
    for token in docnlp:
        solved = docnlp._.coref_chains.resolve(token)
        if not solved:
            new_text += f' {token}'
        elif len(solved) == 1:
            new_text += f' {solved[0].text}'
        elif len(solved) > 1:
            new_text += f' {" and ".join([t.text for t in solved])}'

    return new_text.strip()