"""Tokenizer for fixed-size single-source shortest paths.

Vocabulary:
  - Special: <PAD>, <START>, <SEP>, ####, <END>
  - Source tokens: SRC(v)
  - Matrix cell tokens: W(weight) and INF
  - Answer tokens: D(distance) and UNR

The graph is serialized as a full N x N weighted adjacency matrix. This keeps
the input length fixed for a fixed N, unlike edge-list graph formats.
"""

from __future__ import annotations


SPECIAL_TOKENS = {"<PAD>": 0, "<START>": 1, "<SEP>": 2, "####": 3, "<END>": 4}
PAD_ID = SPECIAL_TOKENS["<PAD>"]
END_ID = SPECIAL_TOKENS["<END>"]


class ShortestPathTokenizer:
    def __init__(self, n: int, max_weight: int, max_distance: int):
        self.n = n
        self.max_weight = max_weight
        self.max_distance = max_distance
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self):
        self.token2id = dict(SPECIAL_TOKENS)
        idx = len(SPECIAL_TOKENS)

        for source in range(self.n):
            self.token2id[f"SRC({source})"] = idx
            idx += 1

        self.token2id["INF"] = idx
        idx += 1
        for weight in range(self.max_weight + 1):
            self.token2id[f"W({weight})"] = idx
            idx += 1

        self.token2id["UNR"] = idx
        idx += 1
        for distance in range(self.max_distance + 1):
            self.token2id[f"D({distance})"] = idx
            idx += 1

        self.id2token = {idx: token for token, idx in self.token2id.items()}
        self.vocab_size = len(self.token2id)

    def encode_prompt(self, source: int, matrix: list[list[int | None]]) -> list[int]:
        ids = [self.token2id["<START>"], self.token2id[f"SRC({source})"]]
        for row in matrix:
            for weight in row:
                if weight is None:
                    ids.append(self.token2id["INF"])
                else:
                    ids.append(self.token2id[f"W({weight})"])
        ids.append(self.token2id["<SEP>"])
        ids.append(self.token2id["####"])
        return ids

    def encode_example(
        self,
        source: int,
        matrix: list[list[int | None]],
        distances: list[int],
    ) -> list[int]:
        ids = self.encode_prompt(source, matrix)
        ids.extend(self.distance_id(distance) for distance in distances)
        ids.append(self.token2id["<END>"])
        return ids

    def distance_id(self, distance: int) -> int:
        if distance > self.max_distance:
            return self.token2id["UNR"]
        return self.token2id[f"D({distance})"]

    def decode_distances(self, ids: list[int]) -> list[int | None]:
        distances: list[int | None] = []
        for token_id in ids:
            token = self.id2token.get(token_id)
            if token == "<END>":
                break
            if token == "UNR":
                distances.append(None)
            elif token is not None and token.startswith("D("):
                distances.append(int(token[2:-1]))
            else:
                break
        return distances

