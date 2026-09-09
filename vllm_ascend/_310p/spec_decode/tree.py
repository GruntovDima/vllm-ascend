# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Backend-independent tree topology and greedy target verification.

Node zero is the already-sampled input token, not a draft candidate. Flat
indices identify scratch-cache slots; depths identify relative RoPE positions.
These coordinates deliberately differ for siblings. This module does not run
models or mutate caches: its accepted input indices describe what to commit.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class TokenTree:
    """Immutable root plus candidates in parent-before-child order.

    ``parents[0]`` must be -1. Token IDs may repeat on different branches,
    but children of the same parent must have distinct IDs so verification
    has an unambiguous path. Input sequences are copied to immutable tuples.
    """

    token_ids: tuple[int, ...]
    parents: tuple[int, ...]
    depths: tuple[int, ...] = field(init=False)
    children: tuple[tuple[int, ...], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        token_ids = tuple(_integer(token, "token ID") for token in self.token_ids)
        parents = tuple(_integer(parent, "parent", minimum=-1) for parent in self.parents)
        if not token_ids:
            raise ValueError("A token tree must contain its input root")
        if len(token_ids) != len(parents):
            raise ValueError("token_ids and parents must have the same node count")
        if parents[0] != -1:
            raise ValueError("The root must have parent -1")

        depths = [0]
        children: list[list[int]] = [[] for _ in token_ids]
        child_tokens: list[set[int]] = [set() for _ in token_ids]
        for node in range(1, len(token_ids)):
            parent = parents[node]
            if not 0 <= parent < node:
                raise ValueError(f"Node {node} must have an earlier parent, got {parent}")
            if token_ids[node] in child_tokens[parent]:
                raise ValueError(f"Duplicate child token {token_ids[node]} under parent {parent}")
            child_tokens[parent].add(token_ids[node])
            children[parent].append(node)
            depths.append(depths[parent] + 1)

        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "depths", tuple(depths))
        object.__setattr__(self, "children", tuple(tuple(nodes) for nodes in children))

    @property
    def num_nodes(self) -> int:
        return len(self.token_ids)

    @property
    def num_candidates(self) -> int:
        return self.num_nodes - 1

    def validate_candidate_count(self, expected: int) -> None:
        """Check the scheduler's candidate count, which excludes the root."""
        expected = _integer(expected, "expected candidate count")
        if self.num_candidates != expected:
            raise ValueError(f"Expected {expected} candidates, got {self.num_candidates}")

    def truncate(self, num_candidates: int) -> "TokenTree":
        """Retain a topological prefix when fewer draft slots are scheduled."""
        num_candidates = _integer(num_candidates, "candidate count")
        if num_candidates > self.num_candidates:
            raise ValueError("Cannot truncate a tree to more candidates than it contains")
        num_nodes = num_candidates + 1
        return TokenTree(self.token_ids[:num_nodes], self.parents[:num_nodes])

    def ancestor_indices(self, node: int) -> tuple[int, ...]:
        """Return root-to-node indices, including self and excluding siblings."""
        node = _integer(node, "node index")
        if node >= self.num_nodes:
            raise ValueError(f"Node index {node} is outside the tree")
        ancestors = []
        while node != -1:
            ancestors.append(node)
            node = self.parents[node]
        return tuple(reversed(ancestors))


def build_comb_tree(
    root_token_id: int,
    candidates_per_depth: Sequence[Sequence[int]],
    *,
    max_candidates: int | None = None,
) -> TokenTree:
    """Attach ranked siblings to each step of the primary MTP backbone.

    Each row contains top-k next-token IDs for the preceding primary node;
    its first token is the primary node for the following row. Other siblings
    remain leaves. ``max_candidates`` excludes the root and can cut a row.
    Empty input rows are invalid; no rows at all produces a root-only tree.
    The budget is an upper bound, not a requirement to manufacture padding.
    """
    root_token_id = _integer(root_token_id, "root token ID")
    if max_candidates is not None:
        max_candidates = _integer(max_candidates, "maximum candidate count")
    rows = []
    for row in candidates_per_depth:
        tokens = tuple(_integer(token, "candidate token ID") for token in row)
        if not tokens:
            raise ValueError("Each candidate depth must contain at least its primary token")
        if len(set(tokens)) != len(tokens):
            raise ValueError("Duplicate child token IDs in a candidate row")
        rows.append(tokens)

    token_ids = [root_token_id]
    parents = [-1]
    parent = 0
    for row in rows:
        remaining = len(row) if max_candidates is None else max_candidates - (len(token_ids) - 1)
        if remaining == 0:
            break
        primary = len(token_ids)
        selected = row[:remaining]
        token_ids.extend(selected)
        parents.extend([parent] * len(selected))
        parent = primary
    return TokenTree(tuple(token_ids), tuple(parents))


@dataclass(frozen=True)
class TreeVerification:
    """Chronological outputs and the corresponding input nodes to commit.

    The final emitted token is not committed: it is the next step's input.
    Even when a budget ends on a matching child, that child is not included
    in the committed path. Consequently both tuples always have equal length.
    A zero output budget returns two empty tuples and commits nothing.
    """

    emitted_token_ids: tuple[int, ...]
    accepted_input_indices: tuple[int, ...]


def greedy_verify(
    tree: TokenTree,
    target_token_ids: Sequence[int],
    *,
    max_output_tokens: int | None = None,
) -> TreeVerification:
    """Follow target predictions only through direct children of each node.

    ``target_token_ids[i]`` is the full target model's greedy next token at
    input node i. Predictions must have been computed with ancestor-only
    attention and parent-derived recurrent state; this routine cannot verify
    those model-side preconditions. It performs no probabilistic rejection.
    """
    predictions = tuple(_integer(token, "target token ID") for token in target_token_ids)
    if len(predictions) != tree.num_nodes:
        raise ValueError(f"Expected {tree.num_nodes} target predictions, got {len(predictions)}")
    budget = tree.num_nodes if max_output_tokens is None else _integer(max_output_tokens, "output token budget")
    emitted = []
    accepted_input_indices = []
    node = 0
    while len(emitted) < budget:
        accepted_input_indices.append(node)
        predicted = predictions[node]
        emitted.append(predicted)
        if len(emitted) == budget:
            break
        child = next((child for child in tree.children[node] if tree.token_ids[child] == predicted), None)
        if child is None:
            break
        node = child
    return TreeVerification(tuple(emitted), tuple(accepted_input_indices))
