from data_marketplace_retrieval.summarization.summarization import Summarizer
import nltk
import heapq


class NLTKSummarizer(Summarizer):

    def __init__(self, len: float = 0.1):
        self.__stopwords = nltk.corpus.stopwords.words('english')
        self.len: float = len

    def summarize(self, processed_text: str, penalize_mask: bool = False, k=1):
        sentence_list: list = self._nltk_summarize(processed_text)
        word_frequencies: dict = self._get_word_frequencies(formatted_text=processed_text)
        word_frequencies = self._normalize_frequencies(word_frequencies)
        sentence_scores: dict = self._get_sentence_score(sentence_list=sentence_list, word_frequencies=word_frequencies)
        return self._get_summary(sentence_scores=sentence_scores, n_sentences=int(round(len(sentence_list) * self.len)))

    def _get_word_frequencies(self, formatted_text: str) -> dict:

        word_frequencies = {}
        for word in nltk.word_tokenize(formatted_text):
            if word not in self.__stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1
        if not word_frequencies:
            print(formatted_text)
        return word_frequencies

    @staticmethod
    def _nltk_summarize(text: str) -> list:
        return nltk.sent_tokenize(text)

    @staticmethod
    def _normalize_frequencies(word_frequencies: dict) -> dict:
        maximum_frequency = max(word_frequencies.values())

        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word] / maximum_frequency)
        return word_frequencies

    @staticmethod
    def _get_sentence_score(sentence_list: list, word_frequencies: dict) -> dict:
        sentence_scores = {}
        for sent in sentence_list:
            for word in nltk.word_tokenize(sent.lower()):
                if word in word_frequencies.keys():
                    if len(sent.split(' ')) < 30:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]
        return sentence_scores

    @staticmethod
    def _get_summary(sentence_scores: dict, n_sentences: int) -> str:
        summary_sentences = heapq.nlargest(n_sentences, sentence_scores, key=sentence_scores.get)
        summary = ' '.join(summary_sentences)
        return summary
