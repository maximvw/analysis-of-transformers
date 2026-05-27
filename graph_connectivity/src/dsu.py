"""Disjoint Set Union with union-by-min for deterministic component IDs."""

from __future__ import annotations


class DSU:
    """DSU with configurable union strategies (min, size, rank)."""

    def __init__(self, n: int, mode: str = "min"):
        self.parent = list(range(n))
        self.mode = mode
        if self.mode == "size":
            self.size = [1] * n
        elif self.mode == "rank":
            self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, u: int, v: int) -> bool:
        ru, rv = self.find(u), self.find(v)
        if ru == rv:
            return False

        if self.mode == "min":
            # union by min: smaller root becomes the root
            if ru > rv:
                ru, rv = rv, ru
            self.parent[rv] = ru
        elif self.mode == "size":
            # union by size: root of larger tree becomes the root
            if self.size[ru] < self.size[rv]:
                ru, rv = rv, ru
            self.parent[rv] = ru
            self.size[ru] += self.size[rv]
        elif self.mode == "rank":
            # union by rank: root of taller tree becomes the root
            if self.rank[ru] < self.rank[rv]:
                ru, rv = rv, ru
            self.parent[rv] = ru
            if self.rank[ru] == self.rank[rv]:
                self.rank[ru] += 1
        else:
            raise ValueError(f"Unknown DSU mode: {self.mode}")

        return True

    def comp(self) -> list[int]:
        """Return comp[i] = Find(i) for all vertices."""
        return [self.find(i) for i in range(len(self.parent))]


def compute_dsu_states(edges: list[tuple[int, int]], n: int, mode: str = "min") -> list[list[int]]:
    """Compute DSU comp[] state after each edge.

    Returns list of length len(edges), where states[t] is comp[] after processing
    edges[0..t].
    """
    dsu = DSU(n, mode=mode)
    states = []
    for u, v in edges:
        dsu.union(u, v)
        states.append(dsu.comp())
    return states
