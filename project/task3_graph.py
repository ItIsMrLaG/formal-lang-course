import functools
import itertools
import operator
from typing import Iterable

import numpy as np
from networkx import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from pyformlang.finite_automaton import State
from pyformlang.finite_automaton import Symbol
import scipy as sp

from project.task1_graph_utils import graph_to_nfa

__all__ = [
    "AdjacencyMatrixFA",
    "tensor_based_rpq",
    "intersect_automata",
    "get_edges_from_fa",
]

from project.task2_regex_utils import regex_to_dfa


def get_edges_from_fa(
    fa: NondeterministicFiniteAutomaton,
) -> set[tuple[State, Symbol, State]]:
    edges = set()
    for start_state, links in fa.to_dict().items():
        for label, end_states in links.items():
            if not isinstance(end_states, Iterable):
                edges.add((start_state, label, end_states))
                continue

            for end_state in end_states:
                edges.add((start_state, label, end_state))

    return edges


class AdjacencyMatrixFA:
    alphabet: set[Symbol]
    states: set[State]
    start_states: set[State]
    final_states: set[State]
    adj_bool_decompress: dict[Symbol, sp.sparse.csc_matrix]
    state_to_idx: dict[State, int]
    idx_to_state: dict[int, State]

    def get_states_number(self) -> int:
        return len(self.states)

    def get_state_to_idx(self, states: list[State]) -> list[int]:
        return list(map(lambda x: self.state_to_idx[x], states))

    def get_idx_to_state(self, ids: list[id]) -> list[State]:
        return list(map(lambda x: self.idx_to_state[x], ids))

    def _get_bool_mask(self, ids: list[int]) -> np.ndarray:
        mask = np.zeros(self.get_states_number(), dtype=bool)
        mask[list(ids)] = True
        return mask

    def _get_start_final_masks(self) -> tuple[np.ndarray, np.ndarray]:
        start_ids = self.get_state_to_idx(list(self.start_states))
        final_ids = self.get_state_to_idx(list(self.final_states))
        return self._get_bool_mask(start_ids), self._get_bool_mask(final_ids)

    def transitive_closure(self) -> sp.sparse.csc_matrix:
        n = self.get_states_number()
        matrices = list(self.adj_bool_decompress.values())
        common_matrix = sp.sparse.csc_matrix(
            (np.ones(n, dtype=bool), (range(n), range(n))), shape=(n, n)
        )

        return (
            functools.reduce(operator.add, matrices, common_matrix)
            ** self.get_states_number()
        )

    def __init__(self, fa: NondeterministicFiniteAutomaton):
        self.states = fa.states
        self.start_states = fa.start_states
        self.final_states = fa.final_states
        self.alphabet = fa.symbols

        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.idx_to_state = {idx: state for state, idx in self.state_to_idx.items()}

        edges = tuple(zip(*get_edges_from_fa(fa)))
        column_states, symbols, row_states = ([], [], []) if not edges else edges

        n = len(self.states)

        self.adj_bool_decompress = {}
        for symbol in self.alphabet:
            mask = np.equal(symbols, symbol).astype(bool)
            self.adj_bool_decompress[symbol] = sp.sparse.csc_matrix(
                (
                    mask,
                    (
                        self.get_state_to_idx(column_states),
                        self.get_state_to_idx(row_states),
                    ),
                ),
                shape=(n, n),
            )

    @classmethod
    def construct_from_adj_bool_decompress(
        cls,
        adj_bool_decompress: dict[Symbol, sp.sparse.csc_matrix],
        states: set[State],
        start_states: set[State],
        final_states: set[State],
        state_to_idx: dict[State, int],
    ):
        self = cls.__new__(cls)
        self.adj_bool_decompress = adj_bool_decompress
        self.alphabet = set(adj_bool_decompress.keys())
        self.states = states
        self.start_states = start_states
        self.final_states = final_states
        self.state_to_idx = state_to_idx
        self.idx_to_state = {idx: state for state, idx in self.state_to_idx.items()}

        return self

    def accepts(self, word: Iterable[Symbol]) -> bool:
        configuration, final = self._get_start_final_masks()

        try:
            for symbol in word:
                bool_matrix = self.adj_bool_decompress[symbol]
                configuration = configuration @ bool_matrix
        except KeyError:
            return False

        return np.any(final & configuration)

    def get_start_final(self) -> list[tuple[State, State]]:
        transition_matrix = self.transitive_closure()
        start_ids = self.get_state_to_idx(list(self.start_states))
        final_ids = self.get_state_to_idx(list(self.final_states))

        return [
            (self.idx_to_state[start], self.idx_to_state[final])
            for start in start_ids
            for final in final_ids
            if transition_matrix[start, final]
        ]

    def is_empty(self) -> bool:
        return not self.get_start_final()

    def update_bool_decompress(self, delta: dict[Symbol, sp.sparse.csc_matrix]):
        for var, matrix in delta.items():
            if var in self.adj_bool_decompress:
                self.adj_bool_decompress[var] += matrix
            else:
                self.alphabet.add(var)
                self.adj_bool_decompress[var] = matrix


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    n, m = automaton1.get_states_number(), automaton2.get_states_number()
    _new_states = [
        State(
            (automaton1.idx_to_state[idx1].value, automaton2.idx_to_state[idx2].value)
        )
        for idx1 in range(n)
        for idx2 in range(m)
    ]

    new_alphabet = automaton1.alphabet & automaton2.alphabet
    new_state_to_idx = {state: idx for idx, state in enumerate(_new_states)}
    new_states = set(_new_states)
    new_start_states = set(
        itertools.product(
            map(lambda x: x.value, automaton1.start_states),
            map(lambda x: x.value, automaton2.start_states),
        )
    )
    new_final_states = set(
        itertools.product(
            map(lambda x: x.value, automaton1.final_states),
            map(lambda x: x.value, automaton2.final_states),
        )
    )

    new_adj_bool_decompress = {}
    for symbol in new_alphabet:
        new_adj_bool_decompress[symbol] = sp.sparse.kron(
            automaton1.adj_bool_decompress[symbol],
            automaton2.adj_bool_decompress[symbol],
        ).tocsc()

    return AdjacencyMatrixFA.construct_from_adj_bool_decompress(
        new_adj_bool_decompress,
        new_states,
        start_states=new_start_states,
        final_states=new_final_states,
        state_to_idx=new_state_to_idx,
    )


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    regex_dfa = regex_to_dfa(regex)

    adj_graph_nfa_representation = AdjacencyMatrixFA(graph_nfa)
    adj_regex_dfa_representation = AdjacencyMatrixFA(regex_dfa)

    common_fa = intersect_automata(
        adj_graph_nfa_representation, adj_regex_dfa_representation
    )

    rpq_answer = tuple(zip(*common_fa.get_start_final()))
    rpq_start_states, rpq_final_states = ([], []) if not rpq_answer else rpq_answer

    rpq_start_state_values = list(map(lambda x: x.value[0], rpq_start_states))
    rpq_final_state_values = list(map(lambda x: x.value[0], rpq_final_states))

    return set(zip(rpq_start_state_values, rpq_final_state_values))
