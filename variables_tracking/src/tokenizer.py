"""Compound tokenizer for the variable tracking task.

Vocabulary:
  - Special: <PAD>=0, <START>=1, <SEP>=2, <ANS>=3, <END>=4
  - Commands: SET / COPY / INC / DEC / ADD / ADDC
  - Queries: Q(x), Q(y), Q(z), Q(w)
  - Answers: 0..max_value-1
"""

from __future__ import annotations


SPECIAL_TOKENS = {"<PAD>": 0, "<START>": 1, "<SEP>": 2, "<ANS>": 3, "<END>": 4}
PAD_ID = 0


class VariableTrackingTokenizer:
    def __init__(self, variables: list[str], max_value: int = 32):
        self.variables = list(variables)
        self.max_value = max_value
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self):
        self.token2id = dict(SPECIAL_TOKENS)
        idx = len(SPECIAL_TOKENS)

        for value in range(self.max_value):
            self.token2id[str(value)] = idx
            idx += 1

        for v in self.variables:
            for value in range(self.max_value):
                self.token2id[f"SET({v},{value})"] = idx
                idx += 1

        for dst in self.variables:
            for src in self.variables:
                self.token2id[f"COPY({dst},{src})"] = idx
                idx += 1

        for v in self.variables:
            self.token2id[f"INC({v})"] = idx
            idx += 1
            self.token2id[f"DEC({v})"] = idx
            idx += 1

        for dst in self.variables:
            for a in self.variables:
                for b in self.variables:
                    self.token2id[f"ADD({dst},{a},{b})"] = idx
                    idx += 1

        for dst in self.variables:
            for src in self.variables:
                for value in range(self.max_value):
                    self.token2id[f"ADDC({dst},{src},{value})"] = idx
                    idx += 1

        for v in self.variables:
            self.token2id[f"Q({v})"] = idx
            idx += 1

        self.id2token = {v: k for k, v in self.token2id.items()}
        self.vocab_size = len(self.token2id)

    def encode_program(self, program: list[str], query_var: str, answer: int) -> list[int]:
        ids = [self.token2id["<START>"]]
        ids.extend(self.token2id[token] for token in program)
        ids.append(self.token2id["<SEP>"])
        ids.append(self.token2id[f"Q({query_var})"])
        ids.append(self.token2id["<ANS>"])
        ids.append(self.token2id[str(answer)])
        ids.append(self.token2id["<END>"])
        return ids

    def get_answer_position(self, program_length: int) -> int:
        return program_length + 3

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.id2token.get(i, f"[UNK:{i}]") for i in ids)

