from tokenizers import Tokenizer


class MyTokenizer:
    def __init__(self, tokenizer_model):
        self.tokenizer = Tokenizer.from_file(tokenizer_model)
    
    def encode(self, text, add_begin=False):
        text = text.strip()
        if add_begin:
            text = "<|endoftext|> " + text
        return self.tokenizer.encode(text).ids
    
    def decode(self, ids):
        return self.tokenizer.decode(ids)