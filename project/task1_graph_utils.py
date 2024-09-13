from dataclasses import dataclass
from pathlib import Path
from typing import Set
from typing import Tuple

from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from pyformlang.finite_automaton import State
import json
import cfpq_data as cd
import networkx as nx

__all__ = [
    "GraphMeta",
    "get_graph_meta_by_name",
    "save_labeled_two_cycles_graph_to_dot",
    "get_graph_meta_from_json",
    "read_graph_from_dot",
    "get_graph_meta_from_graph",
    "graph_to_nfa",
]


@dataclass
class GraphMeta:
    nodes_n: int
    edges_n: int
    labels: Set[str]


def read_graph_from_dot(path: Path) -> nx.MultiDiGraph:
    return nx.drawing.nx_pydot.read_dot(path)


def get_graph_meta_from_json(path: Path) -> GraphMeta:
    with open(path) as json_fl:
        data = json.load(json_fl)
        return GraphMeta(
            nodes_n=data["nodes_n"], edges_n=data["edges_n"], labels=set(data["labels"])
        )


def get_graph_meta_from_graph(graph: nx.Graph) -> GraphMeta:
    return GraphMeta(
        nodes_n=graph.number_of_nodes(),
        edges_n=graph.number_of_edges(),
        labels=set(str(attr["label"]) for (_, _, attr) in graph.edges.data()),
    )


def _save_graph_to_dot(path: Path, graph: nx.Graph):
    return nx.drawing.nx_pydot.write_dot(graph, path)


def get_graph_meta_by_name(name: str) -> GraphMeta:
    path = cd.download(name)
    graph = cd.graph_from_csv(path)
    return get_graph_meta_from_graph(graph)


def save_labeled_two_cycles_graph_to_dot(
    cycle_sizes: Tuple[int, int], labels: Tuple[str, str], path: Path
):
    graph = cd.labeled_two_cycles_graph(cycle_sizes[0], cycle_sizes[1], labels=labels)

    _save_graph_to_dot(path, graph)


def graph_to_nfa(
    graph: nx.MultiDiGraph, start_states: Set[int], final_states: Set[int]
) -> NondeterministicFiniteAutomaton:
    nfa: NondeterministicFiniteAutomaton = (
        NondeterministicFiniteAutomaton.from_networkx(
            graph
        ).remove_epsilon_transitions()
    )

    nodes_set = set([int(node) for node in graph.nodes])

    start_nodes = start_states if start_states else nodes_set
    final_nodes = final_states if final_states else nodes_set

    for node in start_nodes:
        nfa.add_start_state(State(node))

    for node in final_nodes:
        nfa.add_final_state(State(node))

    return nfa
