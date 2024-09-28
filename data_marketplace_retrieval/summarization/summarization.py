import re
import nltk
import heapq
from abc import ABC, abstractmethod


class Summarizer(ABC):

    @abstractmethod
    def summarize(self, processed_text: str):
        pass
