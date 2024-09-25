import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable

import numpy as np
from networkx import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, DeterministicFiniteAutomaton
from pyformlang.finite_automaton import Symbol
import scipy as sp

import scripts.shared
from project.task1_graph_utils import graph_to_nfa, read_graph_from_dot

__all__ = [
    "AdjacencyMatrixFA",
    "tensor_based_rpq",
    "intersect_automata",
]

from project.task2_regex_utils import regex_to_dfa


class AdjacencyMatrixFA:
    alphabet: set[Any]
    adjacencyMatrixBoolDecomposition: dict[Any, sp.sparse.csc_matrix]
    n: int
    start_ids: set[int]
    final_ids: set[int]

    def __get_bool_mask(self, ids: Iterable):
        mask = np.zeros(self.n, dtype=bool)
        mask[list(ids)] = True
        return mask

    def __construct_from_fa(self, fa: NondeterministicFiniteAutomaton | DeterministicFiniteAutomaton):
        row_nodes, col_nodes, labels = zip(*filter(lambda x: x[2] is not None, fa.to_networkx().edges(data="label")))
        self.alphabet = set(labels)
        self.n = len(fa.states)

        node_to_idx = dict(map(lambda x: (x[1], x[0]), enumerate(set(row_nodes + col_nodes))))
        row_ids = list(map(lambda node: node_to_idx[str(node)], row_nodes))
        col_ids = list(map(lambda node: node_to_idx[str(node)], col_nodes))

        self.start_ids = set(map(lambda node: node_to_idx[str(node)], fa.start_states))  # TODO: test correctness
        self.final_ids = set(map(lambda node: node_to_idx[str(node)], fa.final_states))

        self.adjacencyMatrixBoolDecomposition = {}
        for label in self.alphabet:
            mask = np.equal(labels, label).astype(bool)
            self.adjacencyMatrixBoolDecomposition[label] = sp.sparse.csc_matrix((mask, (row_ids, col_ids)),
                                                                                shape=(self.n, self.n))

    def __construct_from_adjacencyMatrixBoolDecomposition(self, args: tuple[
        set[Any], dict[Any, sp.sparse.csc_matrix], int, set[int], set[int]]):
        self.alphabet = args[0]
        self.adjacencyMatrixBoolDecomposition = args[1]
        self.n = args[2]
        self.start_ids = args[3]
        self.final_ids = args[4]

    def __init__(self, args: NondeterministicFiniteAutomaton | DeterministicFiniteAutomaton | tuple[
        set[Any], dict[Any, sp.sparse.csc_matrix], int, set[int], set[int]]):
        if type(args) is NondeterministicFiniteAutomaton or type(args) is DeterministicFiniteAutomaton:
            self.__construct_from_fa(args)
        else:
            self.__construct_from_adjacencyMatrixBoolDecomposition(args)

    def accepts(self, word: Iterable[Symbol]) -> bool:

        configuration = self.__get_bool_mask(self.start_ids)
        final = self.__get_bool_mask(self.final_ids)

        try:
            for symbol in word:
                bool_matrix = self.adjacencyMatrixBoolDecomposition[symbol]
                configuration = configuration @ bool_matrix
        except KeyError:
            return False

        if np.any(final & configuration):
            return True
        return False

    def is_empty(self) -> bool:
        matrices = list(self.adjacencyMatrixBoolDecomposition.values())
        common_matrix = sp.sparse.csc_matrix((np.ones(self.n, dtype=bool), (range(self.n), range(self.n))),
                                             shape=(self.n, self.n))

        for matrix in matrices:
            common_matrix = (matrix + common_matrix).astype(bool)

        transition_matrix = common_matrix ** self.n

        configuration = self.__get_bool_mask(self.start_ids)
        final = self.__get_bool_mask(self.final_ids)

        configuration = configuration @ transition_matrix.toarray()

        if np.any(final & configuration):
            return False
        return True


def tenser_multiplication(matrix1: sp.sparse.csc_matrix, matrix2: sp.sparse.csc_matrix) -> sp.sparse.csc_matrix:
    return sp.sparse.kron(matrix1, matrix2, "csc")


@dataclass
class KronIds:
    n: int
    m: int

    def get_compressed_state(self, n_idx: int, m_idx: int):
        return n_idx * self.m + m_idx


def intersect_automata(automaton1: AdjacencyMatrixFA,
                       automaton2: AdjacencyMatrixFA) -> AdjacencyMatrixFA:
    st = KronIds(automaton1.n, automaton2.n)

    new_n = st.n * st.m
    new_start_states = set(map(lambda x: st.get_compressed_state(x[0], x[1]),
                               itertools.product(automaton1.start_ids, automaton2.start_ids)))
    new_final_states = set(map(lambda x: st.get_compressed_state(x[0], x[1]),
                               itertools.product(automaton1.final_ids, automaton2.final_ids)))
    new_alphabet = automaton1.alphabet & automaton2.alphabet

    new_adjacencyMatrixBoolDecomposition = {}
    for symbol in new_alphabet:
        new_adjacencyMatrixBoolDecomposition[symbol] = tenser_multiplication(
            automaton1.adjacencyMatrixBoolDecomposition[symbol], automaton2.adjacencyMatrixBoolDecomposition[symbol])

    return AdjacencyMatrixFA(
        (new_alphabet, new_adjacencyMatrixBoolDecomposition, new_n, new_start_states, new_final_states))


def tensor_based_rpq(regex: str, graph: MultiDiGraph, start_nodes: set[int],
                     final_nodes: set[int]) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)

    adj_regex_dfa_representation = AdjacencyMatrixFA(regex_dfa)
    adj_graph_nfa_representation = AdjacencyMatrixFA(graph_nfa)

    common_fa_adj_representation = intersect_automata(adj_graph_nfa_representation, adj_regex_dfa_representation)

    pass

# if __name__ == '__main__':
#     graph = read_graph_from_dot(
#         scripts.shared.TESTS / Path("resources/res_task1_graph_utils") / "test_graph.dot")
# nfa = graph_to_nfa(graph, {1}, {4})
#
# testAdj = AdjacencyMatrixFA(nfa)
# print(testAdj.accepts([
#     Symbol("a"),
#     Symbol("b"),
#     # Symbol("c"),
#     Symbol("d"),
#     Symbol("e"),
#     # Symbol("f"),
# ]))
# print(testAdj.is_empty())
