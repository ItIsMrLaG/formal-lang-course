import itertools
from typing import Any, Iterable
from pyformlang.finite_automaton import Symbol
from pyformlang.finite_automaton import State

import networkx as nx
import pyformlang

__all__ = [
    "RsmState",
    "GssNode",
    "Configuration",
    "gll_based_cfpq",
]


class RsmState:
    label: Symbol
    state: State

    def __str__(self):
        return f"(label={self.label}, state={self.state})"

    def __init__(self, label: Symbol, state: State):
        self.state = state
        self.label = label

    def __eq__(self, other):
        if not isinstance(other, RsmState):
            raise TypeError("Compare RsmState with something else")
        return (self.label, self.state).__eq__((other.label, other.state))

    def __hash__(self) -> int:
        return hash((self.label, self.state))


class GssNode:
    rsm_state: RsmState
    graph_node: int

    def __str__(self):
        return f"(rsm_state={self.rsm_state}, graph_node={self.graph_node})"

    def __init__(self, rsm_state: RsmState, graph_node: int):
        self.rsm_state = rsm_state
        self.graph_node = graph_node

    def __eq__(self, other):
        if not isinstance(other, GssNode):
            raise TypeError("Compare GssNode with something else")
        return (self.rsm_state, self.graph_node).__eq__(
            (other.rsm_state, other.graph_node)
        )

    def __hash__(self) -> int:
        return hash((self.rsm_state, self.graph_node))


class Configuration:
    rsm_state: RsmState
    graph_node: int
    gss_node: GssNode

    def __str__(self):
        return f"(rsm_state={self.rsm_state}, graph_node={self.graph_node}, gss_node={self.gss_node})"

    def __init__(self, rsm_state: RsmState, graph_node: int, gss_node: GssNode):
        self.rsm_state = rsm_state
        self.graph_node = graph_node
        self.gss_node = gss_node

    def __eq__(self, other):
        if not isinstance(other, Configuration):
            raise TypeError("Compare Configuration with something else")
        return (
            self.rsm_state,
            self.graph_node,
            self.gss_node,
        ).__eq__((other.rsm_state, other.graph_node, other.gss_node))

    def __hash__(self) -> int:
        return hash((self.rsm_state, self.graph_node, self.gss_node))


def get_graph_from_node_edges(g: nx.Graph, from_nd: Any) -> dict[Symbol, set[Any]]:
    edges = {}
    for from_nd, to_nd, lbl in g.edges(from_nd, data="label"):
        symbl = lbl if isinstance(lbl, Symbol) else Symbol(lbl)
        edges.setdefault(symbl, set()).add(to_nd)
    return edges


def get_rsm_from_state_edges(
        rsm: pyformlang.rsa.RecursiveAutomaton, from_st: RsmState
) -> dict[Symbol, set[RsmState]]:
    box_label = from_st.label
    dfa = rsm.get_box(box_label).dfa
    edges = {}
    if from_st.state not in dfa.to_dict():
        return {}

    for label, to_states in (dfa.to_dict()[from_st.state]).items():
        if not isinstance(to_states, Iterable):
            edges.setdefault(label, set()).add(RsmState(box_label, to_states))
            continue

        for to_state in to_states:
            edges.setdefault(label, set()).add(RsmState(box_label, to_state))

    return edges


def _gss_eval_rule(
        cfg: Configuration,
        rsm: pyformlang.rsa.RecursiveAutomaton,
        graph: nx.DiGraph,
) -> set[Configuration]:
    rsm_to_states: dict[Symbol, set[RsmState]] = get_rsm_from_state_edges(
        rsm, cfg.rsm_state
    )
    g_to_nodes: dict[Symbol, set[int]] = get_graph_from_node_edges(
        graph, cfg.graph_node
    )

    common_terms = set(rsm_to_states.keys()) & set(g_to_nodes.keys())
    return {
        Configuration(rsm_state, g_node, cfg.gss_node)
        for lbl in common_terms
        for rsm_state in rsm_to_states[lbl]
        for g_node in g_to_nodes[lbl]
    }


def _gss_call_rule(
        cfg: Configuration,
        rsm: pyformlang.rsa.RecursiveAutomaton,
        gss_graph: nx.MultiDiGraph,
) -> set[Configuration]:
    rsm_to_states: dict[Symbol, set[RsmState]] = get_rsm_from_state_edges(
        rsm, cfg.rsm_state
    )
    ans = set()

    common_non_terms = rsm.labels & set(rsm_to_states.keys())
    for non_term in common_non_terms:
        box_dfa = rsm.get_box(non_term).dfa
        for start_st in box_dfa.start_states:
            new_rsm_state = RsmState(non_term, start_st)
            new_gss_node = GssNode(new_rsm_state, cfg.graph_node)

            if (
                    new_gss_node in gss_graph.nodes
                    and gss_graph.nodes[new_gss_node]["pop_set"]
            ):
                for rsm_to_state, g_node in itertools.product(
                        rsm_to_states[non_term], gss_graph.nodes[new_gss_node]["pop_set"]
                ):
                    gss_graph.add_edge(new_gss_node, cfg.gss_node, label=rsm_to_state)
                    ans.add(Configuration(rsm_to_state, g_node, cfg.gss_node))
            else:
                for rsm_to_state in rsm_to_states[non_term]:
                    gss_graph.add_node(new_gss_node, pop_set=None)
                    gss_graph.add_edge(new_gss_node, cfg.gss_node, label=rsm_to_state)
                    ans.add(Configuration(new_rsm_state, cfg.graph_node, new_gss_node))

    return ans


def _gss_return_rule(
        cfg: Configuration,
        rsm: pyformlang.rsa.RecursiveAutomaton,
        gss_graph: nx.MultiDiGraph,
        init_gss_node: GssNode,
) -> tuple[set[Configuration], set[tuple[int, int]]]:
    current_box_dfa = rsm.get_box(cfg.rsm_state.label).dfa
    if cfg.rsm_state.state not in current_box_dfa.final_states:
        return set(), set()

    if gss_graph.nodes[cfg.gss_node]["pop_set"] is None:
        gss_graph.nodes[cfg.gss_node]["pop_set"] = set()
    gss_graph.nodes[cfg.gss_node]["pop_set"].add(cfg.graph_node)

    gss_to_nodes = get_graph_from_node_edges(gss_graph, cfg.gss_node)

    ans, buffer = set(), set()

    for ret_rsm_st, to_nodes in gss_to_nodes.items():
        for to_node in to_nodes:
            if to_node == init_gss_node:
                buffer.add((cfg.gss_node.graph_node, cfg.graph_node))
                continue

            ans.add(Configuration(ret_rsm_st.value, cfg.graph_node, to_node))

    return ans, buffer


def gll_based_cfpq(
        rsm: pyformlang.rsa.RecursiveAutomaton,
        graph: nx.DiGraph,
        start_nodes: set[int] = None,
        final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    gss_graph = nx.MultiDiGraph()
    processed: set[Configuration] = set()
    wait_processing: set[Configuration] = set()

    def _update_wait_processing(_buf: set[Configuration]):
        for _cfg in _buf:
            if _cfg not in wait_processing and _cfg not in processed:
                wait_processing.add(_cfg)

    start_nodes = set(start_nodes) if start_nodes else set(graph.nodes)
    final_nodes = set(final_nodes) if final_nodes else set(graph.nodes)

    magic_rsm_state = RsmState(Symbol("|_$_|"), State("|_$_|"))
    init_gss_node = GssNode(magic_rsm_state, -100)

    start_lbl = rsm.initial_label
    for g_start_nd in start_nodes:
        for box_start_st in rsm.get_box(start_lbl).dfa.start_states:
            rsm_state = RsmState(start_lbl, box_start_st)
            gss_node = GssNode(rsm_state, g_start_nd)

            gss_graph.add_node(gss_node, pop_set=None)
            gss_graph.add_edge(gss_node, init_gss_node, magic_rsm_state)
            wait_processing.add(Configuration(rsm_state, g_start_nd, gss_node))

    mb_ans = set()

    while wait_processing:
        buffer = set()
        cfg = wait_processing.pop()
        processed.add(cfg)

        buffer |= _gss_eval_rule(cfg, rsm, graph)
        buffer |= _gss_call_rule(cfg, rsm, gss_graph)
        _buffer, _mb_ans = _gss_return_rule(cfg, rsm, gss_graph, init_gss_node)
        buffer |= _buffer
        mb_ans |= _mb_ans

        _update_wait_processing(buffer)

    return {
        (mb_start, mb_final)
        for mb_start, mb_final in mb_ans
        if mb_start in start_nodes and mb_final in final_nodes
    }
