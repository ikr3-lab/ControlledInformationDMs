from dm_retrieval.summarization.summarization import Summarizer
import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
import nltk
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer

from sumy.utils import get_stop_words
from enum import Enum


class SumyModel(Enum):
    LexRankSummarizer = LexRankSummarizer
    LuhnSummarizer = LuhnSummarizer
    LsaSummarizer = LsaSummarizer
    TextRankSummarizer = TextRankSummarizer
    SumBasicSummarizer = SumBasicSummarizer
    KLSummarizer = KLSummarizer


class SumySummarizer(Summarizer):

    def __init__(self, len: float = 0.1, model=SumyModel.LexRankSummarizer, stopwords: bool = True,
                 stemmer: bool = True):
        if stemmer:
            stemmer = Stemmer("english")
        self.len: float = len
        self.model = model.value(stemmer) if stemmer else model.value()
        if stopwords:
            self.model.stop_words = nltk.corpus.stopwords.words('english')

    def summarize(self, processed_text: str, penalize_mask: bool = False, k=1):
        n_sentences = len(self._count_sentences(processed_text))
        parser = PlaintextParser.from_string(processed_text, Tokenizer("english"))

        penalty_list = []
        if penalize_mask:
            for col, sentence in enumerate(parser.document.sentences):
                if len(sentence.words) > 0:
                    penalty_list.append(sentence._text.count("<mask") / len(sentence.words))
                else:
                    penalty_list.append(1)
            summary = self.model(parser.document, int(round(n_sentences * self.len)), penalty_list=penalty_list, k=k)
        else:
            summary = self.model(parser.document, int(round(n_sentences * self.len)))

        return " ".join([str(sentence) for sentence in summary])

    @staticmethod
    def _count_sentences(text: str) -> list:
        return nltk.sent_tokenize(text)
