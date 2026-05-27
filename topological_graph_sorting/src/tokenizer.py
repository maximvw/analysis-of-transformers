"""Tokenizer for lexicographic topological sorting.

Vocabulary:
  - Special: <PAD>, <START>, <SEP>, ####, <END>
  - Graph size tokens: N(k)
  - Edge tokens: E(u,v)
  - Answer tokens: V(v)
"""

from __future__ import annotations


SPECIAL_TOKENS = {"<PAD>": 0, "<START>": 1, "<SEP>": 2, "####": 3, "<END>": 4}
PAD_ID = SPECIAL_TOKENS["<PAD>"]
END_ID = SPECIAL_TOKENS["<END>"]


class TopologicalSortTokenizer:
    def __init__(self, max_n: int):
        self.max_n = max_n
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self):
        self.token2id = dict(SPECIAL_TOKENS)
        idx = len(SPECIAL_TOKENS)

        for n in range(1, self.max_n + 1):
            self.token2id[f"N({n})"] = idx
            idx += 1

        for u in range(self.max_n):
            for v in range(self.max_n):
                if u == v:
                    continue
                self.token2id[f"E({u},{v})"] = idx
                idx += 1

        for v in range(self.max_n):
            self.token2id[f"V({v})"] = idx
            idx += 1

        self.id2token = {idx: token for token, idx in self.token2id.items()}
        self.vocab_size = len(self.token2id)

    def encode_prompt(self, n: int, edges: list[list[int]] | list[tuple[int, int]]) -> list[int]:
        ids = [self.token2id["<START>"], self.token2id[f"N({n})"]]
        ids.extend(self.token2id[f"E({u},{v})"] for u, v in edges)
        ids.append(self.token2id["<SEP>"])
        ids.append(self.token2id["####"])
        return ids

    def encode_example(
        self,
        n: int,
        edges: list[list[int]] | list[tuple[int, int]],
        order: list[int],
    ) -> list[int]:
        ids = self.encode_prompt(n, edges)
        ids.extend(self.token2id[f"V({v})"] for v in order)
        ids.append(self.token2id["<END>"])
        return ids

    def vertex_id(self, vertex: int) -> int:
        return self.token2id[f"V({vertex})"]

    def decode_vertices(self, ids: list[int]) -> list[int]:
        vertices = []
        for token_id in ids:
            token = self.id2token.get(token_id)
            if token == "<END>":
                break
            if token is None or not token.startswith("V("):
                break
            vertices.append(int(token[2:-1]))
        return vertices
