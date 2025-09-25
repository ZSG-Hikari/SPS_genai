import random, re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

class BigramModel:
    def __init__(self, corpus: List[str]):
        self.vocab = set()
        self.bigrams: Dict[str, Counter] = defaultdict(Counter)
        self.cdf: Dict[str, List[Tuple[str, float]]] = {}

        for line in corpus:
            tokens = re.findall(r"[a-zA-Z0-9']+", line.lower())
            if not tokens: 
                continue
            self.vocab.update(tokens)
            tokens = ["<s>"] + tokens + ["</s>"]
            for a, b in zip(tokens, tokens[1:]):
                self.bigrams[a][b] += 1

        for prev, counter in self.bigrams.items():
            total = sum(counter.values())
            running = 0.0
            lst = []
            for word, cnt in sorted(counter.items()):
                running += cnt / total
                lst.append((word, running))
            self.cdf[prev] = lst

    def _sample_next(self, prev: str) -> str:
        if prev not in self.cdf:
            return random.choice(list(self.vocab)) if self.vocab else "</s>"
        r = random.random()
        for word, cutoff in self.cdf[prev]:
            if r <= cutoff:
                return word
        return self.cdf[prev][-1][0]

    def generate_text(self, start_word: str, length: int = 15) -> str:
        start = start_word.lower()
        prev = start if start in self.bigrams else "<s>"
        out = [] if prev == "<s>" else [start_word]
        for _ in range(max(1, length)):
            nxt = self._sample_next(prev)
            if nxt == "</s>": break
            out.append(nxt); prev = nxt
        return " ".join(out)
