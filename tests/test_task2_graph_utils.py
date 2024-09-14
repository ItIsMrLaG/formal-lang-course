from pathlib import Path

from pyformlang.regular_expression import MisformedRegexError
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton

import cfpq_data as cd
import pytest

from project.task1_graph_utils import read_graph_from_dot
from project.task1_graph_utils import save_labeled_two_cycles_graph_to_dot
from project.task1_graph_utils import get_graph_meta_from_graph
from project.task1_graph_utils import graph_to_nfa
from project.task2_regex_utils import regex_to_dfa


def test_regex_to_dfa_with_misformed_regex_input():
    with pytest.raises(MisformedRegexError):
        regex_to_dfa("*")


def get_states_as_int_set(
    nfa: NondeterministicFiniteAutomaton,
) -> tuple[set[int], set[int], set[int]]:
    st_states = set(int(st.value) for st in nfa.start_states)
    fin_states = set(int(st.value) for st in nfa.final_states)
    states = set(int(st.value) for st in nfa.states)
    return st_states, fin_states, states


@pytest.mark.parametrize(
    "graph_name,start,final",
    [
        pytest.param("wc", {1}, {2}, id="wc_not_all_states_start&final"),
        pytest.param("wc", set(), set(), id="wc"),
        pytest.param("atom", {1}, {2}, id="atom_not_all_states_start&final"),
        pytest.param("atom", set(), set(), id="atom"),
        pytest.param("core", {1}, {2}, id="core_not_all_states_start&final"),
        pytest.param("core", set(), set(), id="core"),
    ],
)
def test_graph_to_nfa_with_graph_from_dataset_all_nodes_final_and_start(
    graph_name: str,
    start: set[int],
    final: set[int],
):
    path = cd.download(graph_name)
    graph = cd.graph_from_csv(path)
    graph_meta = get_graph_meta_from_graph(graph)

    test_nfa = graph_to_nfa(graph, start, final)
    start_states, final_states, all_states = get_states_as_int_set(test_nfa)

    if len(start) == len(final) == 0:
        assert start_states == final_states == all_states
    else:
        assert start_states == start
        assert final_states == final

    assert all_states == set(int(st) for st in graph.nodes)
    assert test_nfa.symbols == graph_meta.labels


def test_graph_to_nfa_with_saved_graph(tmp_path: Path):
    test_path = tmp_path / "tmp_file.dot"

    cycle_n = (2, 3)
    labels = ("fst", "snd")
    save_labeled_two_cycles_graph_to_dot(
        cycle_sizes=cycle_n, labels=labels, path=test_path
    )

    graph = read_graph_from_dot(test_path)
    graph_meta = get_graph_meta_from_graph(graph)

    test_nfa = graph_to_nfa(graph, set(), set())
    start_states, final_states, all_states = get_states_as_int_set(test_nfa)

    assert start_states == final_states == all_states
    assert all_states == set(int(st) for st in graph.nodes)
    assert test_nfa.symbols == graph_meta.labels
