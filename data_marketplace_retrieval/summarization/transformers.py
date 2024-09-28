from transformers import pipeline
from data_marketplace_retrieval.summarization.summarization import Summarizer
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer


class TransformerSummarizer(Summarizer):
    def __init__(self, len: float = 0.1, interval: float = 0.05,
                 model: str = "Alred/t5-small-finetuned-summarization-cnn", tokenizer: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if tokenizer else AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.max_length: float = len + interval
        self.min_length: float = len - interval

    def summarize(self, processed_text: str) -> str:
        length = len(processed_text)
        inputs = self.tokenizer(processed_text, return_tensors="pt").input_ids
        outputs = self.model.generate(inputs, max_length=self._get_abs_max_length(length),
                                      min_length=self._get_abs_min_length(length), do_sample=False)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _get_abs_min_length(self, length: int) -> int:
        return int(round(length * self.min_length))

    def _get_abs_max_length(self, length: int) -> int:
        return int(round(length * self.max_length))
