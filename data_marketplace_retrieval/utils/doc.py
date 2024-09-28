import pandas as pd
from typing import Union
from data_marketplace_retrieval.processing import sanitization
from transformers import pipeline
import re


class DocList(list):
    def to_df(self):
        return pd.DataFrame.from_records([doc.to_dict() for doc in self])


class Doc(object):

    def __init__(self,
                 id: str,
                 title: str,
                 date: str,
                 kicker: str,
                 author: str,
                 raw_text: str,
                 first_paragraph: str,
                 url: str,
                 category: str,
                 ents: list[dict] = None,
                 processed_text: str = None,
                 indexable_text: str = None):
        self.id: str = id
        self.title: str = title
        self.date: str = date
        self.kicker: str = kicker
        self.author: str = author
        self.raw_text: str = raw_text
        self.text: str = ""
        self.masked_text: str = ""
        self.first_paragraph: str = first_paragraph
        self.url: str = url
        self.category: str = category
        self.ents_to_remove: list = None
        self.ents: list[dict] = ents
        self.masked_ents: list[dict] = None
        self.inferred_count = 0
        self.mask_count = 0
        self.processed_text: str = processed_text
        self.indexable_text: str = indexable_text

    def sanitize_text(self, sanit_type='mask'):
        if self.ents_to_remove and sanit_type != "skip":
            self.masked_text, self.masked_ents = sanitization.ner_sanitize(ents_to_remove=self.ents_to_remove,
                                                                           text=self.text,
                                                                           sanit_type=sanit_type)
        else:
            self.masked_text = self.text
            self.masked_ents = []

    def process_text(self, type="text", summarizer=None, coreferee=False, penalize_mask=False, k=1):
        if coreferee:
            self.text = sanitization.coreferee(self.raw_text)
        elif penalize_mask:
            self.text = self.masked_text
        else:
            self.text = self.raw_text

        if type == "summarize":
            self.text = summarizer.summarize(self.text, penalize_mask=penalize_mask, k=k)
        elif type == "title":
            self.text = self.title
        elif type == "paragraph":
            self.text = self.first_paragraph

    def set_indexable_text(self, add_title=False):
        self.indexable_text = self.masked_text.replace("<mask>", "")
        # self.indexable_text = self.masked_text
        if add_title:
            self.indexable_text = self.title + " " + self.indexable_text

    def get_ents(self):
        return sanitization.get_ents(self.raw_text)

    def set_ents_to_remove(self):
        ents = sanitization.get_ents(self.raw_text)
        self.ents_to_remove = sanitization.get_random_ents(ents)

    def to_index_doc(self):
        return {
            'docno': self.id,
            'text': self.indexable_text
        }

    def to_dict(self):
        return self.__dict__

    def unmask(self, unmasker, penalize_mask=False):
        if not penalize_mask:
            self.inferred_count, self.mask_count = sanitization.unmask(self.masked_text, self.masked_ents, unmasker)
        else:
            self.inferred_count, self.mask_count = sanitization.unmask_alt(self.text, unmasker)
            regex_sub = r'<mask[^>]+>'
            self.masked_text = re.sub(regex_sub, r'<mask>', self.text)
