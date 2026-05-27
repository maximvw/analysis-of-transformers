"""Tokenizer for the string edit sequence experiment."""

from __future__ import annotations


SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<START>": 1,
    "<ARROW>": 2,
    "<SEP>": 3,
    "<END>": 4,
}
PAD_ID = SPECIAL_TOKENS["<PAD>"]
END_ID = SPECIAL_TOKENS["<END>"]


class StringEditTokenizer:
    def __init__(self, alphabet: list[str] | tuple[str, ...]):
        self.alphabet = list(alphabet)
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self):
        self.token2id = dict(SPECIAL_TOKENS)
        idx = len(self.token2id)

        for char in self.alphabet:
            self.token2id[char] = idx
            idx += 1

        for op in ("COPY", "DEL", "INS", "SUB"):
            for char in self.alphabet:
                self.token2id[f"{op}_{char}"] = idx
                idx += 1

        self.id2token = {idx: token for token, idx in self.token2id.items()}
        self.vocab_size = len(self.token2id)

    def encode_prompt(self, source: str, target: str) -> list[int]:
        ids = [self.token2id["<START>"]]
        ids.extend(self.token2id[char] for char in source)
        ids.append(self.token2id["<ARROW>"])
        ids.extend(self.token2id[char] for char in target)
        ids.append(self.token2id["<SEP>"])
        return ids

    def encode_example(self, source: str, target: str, script: list[str]) -> list[int]:
        ids = self.encode_prompt(source, target)
        ids.extend(self.token2id[token] for token in script)
        ids.append(self.token2id["<END>"])
        return ids

    def decode_ids(self, ids: list[int]) -> list[str]:
        return [self.id2token[token_id] for token_id in ids]

    def decode_script(self, ids: list[int]) -> list[str]:
        script = []
        for token_id in ids:
            token = self.id2token[token_id]
            if token == "<END>":
                break
            script.append(token)
        return script
