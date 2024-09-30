import functools
import itertools
import operator

import scipy as sp
from networkx import MultiDiGraph

from project.task1_graph_utils import graph_to_nfa
from project.task2_regex_utils import regex_to_dfa
from project.task3_graph import AdjacencyMatrixFA


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)

    dfa = AdjacencyMatrixFA(regex_dfa)
    nfa = AdjacencyMatrixFA(graph_nfa)

    dfa_m, nfa_n = dfa.get_states_number(), nfa.get_states_number()
    new_alphabet = dfa.alphabet & nfa.alphabet

    start_states = list(
        itertools.product(
            dfa.get_state_to_idx(list(dfa.start_states)),
            nfa.get_state_to_idx(list(nfa.start_states)),
        )
    )

    matrices = [
        sp.sparse.csc_matrix((dfa_m, nfa_n), dtype=bool)
        for _ in range(len(start_states))
    ]
    for (dfa_idx, nfa_idx), matrix in zip(start_states, matrices):
        matrix[dfa_idx, nfa_idx] = True

    front = sp.sparse.vstack(matrices, "csc", dtype=bool)
    visited = front
    permutation_matrices = {
        symbol: dfa.adj_bool_decompress[symbol].transpose() for symbol in new_alphabet
    }

    while front.count_nonzero() != 0:
        symbol_fronts = []

        for symbol in new_alphabet:
            _symbol_front = front @ nfa.adj_bool_decompress[symbol]
            symbol_fronts.append(
                sp.sparse.vstack(
                    [
                        permutation_matrices[symbol]
                        @ _symbol_front[dfa_m * idx : dfa_m * (idx + 1)]
                        for idx in range(len(start_states))
                    ]
                )
            )

        front = functools.reduce(operator.add, symbol_fronts, front) > visited
        visited = visited + front

    answer = set()
    for idx, (_, nfa_start_idx) in enumerate(start_states):
        _states_fix_start = visited[dfa_m * idx : dfa_m * (idx + 1)]
        row, col = _states_fix_start.nonzero()
        dfa_states, nfa_states = dfa.get_idx_to_state(row), nfa.get_idx_to_state(col)

        for dfa_state, nfa_state in zip(dfa_states, nfa_states):
            if dfa_state in dfa.final_states and nfa_state in nfa.final_states:
                answer.add((nfa.idx_to_state[nfa_start_idx].value, nfa_state.value))

    return answer
