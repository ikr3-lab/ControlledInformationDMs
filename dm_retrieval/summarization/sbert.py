from dm_retrieval.summarization.summarization import Summarizer
import nltk
from summarizer.sbert import SBertSummarizer


class SBERTSummarizer(Summarizer):

    def __init__(self, len: float = 0.1, model="paraphrase-MiniLM-L6-v2"):
        self.len: float = len
        self.model = SBertSummarizer(model)

    def summarize(self, processed_text: str, penalize_mask: bool = False, k=1):
        n_sentences = len(self._count_sentences(processed_text))
        return self.model(processed_text, num_sentences=int(round((n_sentences * self.len))))

    @staticmethod
    def _count_sentences(text: str) -> list:
        return nltk.sent_tokenize(text)
