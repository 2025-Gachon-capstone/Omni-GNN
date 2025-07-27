from collections import Counter

def convert_by_vocab(vocab, tokens):
    return [vocab[token] for token in tokens]

class FreqVocab:
    def __init__(self):
        self.counter = Counter()
        self.frequency = []

    def update(self, user2seq):
        for user_id, seq in user2seq.items():
            self.counter[f"[USR_{user_id}]"] = len(seq)
            self.counter.update([t["product_id"] for t in seq])

    def generate_vocab(self):
        self.token_count = len(self.counter)
        self.special_tokens = ["[MASK]", "[pad]", "[NO_USE]"]
        self.token_to_ids = {}

        # assign special tokens first
        for token in self.special_tokens:
            self.token_to_ids[token] = len(self.token_to_ids) + 1

        # assign remaining tokens
        for token, _ in self.counter.most_common():
            if token not in self.token_to_ids:
                self.token_to_ids[token] = len(self.token_to_ids) + 1

        # ensure special tokens have zero count
        for token in self.special_tokens:
            self.counter[token] = 0

        self.id_to_tokens = {v: k for k, v in self.token_to_ids.items()}
        self.vocab_words = list(self.token_to_ids.keys())

        id_list = sorted(self.id_to_tokens.keys())
        self.frequency = [self.counter[self.id_to_tokens[i]] for i in id_list]

        # BERT4ETH-style token ID attributes
        self.mask_token_id = self.token_to_ids["[MASK]"]
        self.pad_token_id = self.token_to_ids["[pad]"]
        self.no_use_token_id = self.token_to_ids["[NO_USE]"]

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.token_to_ids, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.id_to_tokens, ids)
