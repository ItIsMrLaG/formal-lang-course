from itertools import product
from typing import Any
from typing import Set

import networkx as nx
import pyformlang
import scipy as sp

from project.task6_hellings_cfpq_wcnf import revert_cfg

from project.task6_hellings_cfpq_wcnf import cfg_to_weak_normal_form

__all__ = ["matrix_based_cfpq"]


def matrix_based_cfpq(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> set[tuple[int, int]]:
    wcnf_cfg = cfg_to_weak_normal_form(cfg)
    term_to_vars, vars_body_to_head = revert_cfg(wcnf_cfg)

    n = graph.number_of_nodes()
    idx_to_node = {i: node for i, node in enumerate(graph.nodes)}
    node_to_idx = {node: i for i, node in idx_to_node.items()}

    bool_var_ms: dict[Any, sp.sparse.csc_matrix] = {
        var: sp.sparse.csc_matrix((n, n), dtype=bool) for var in wcnf_cfg.variables
    }

    for v1, v2, lb in graph.edges.data("label"):
        term_vars = term_to_vars.get(pyformlang.cfg.Terminal(lb), set())
        idx1, idx2 = node_to_idx[v1], node_to_idx[v2]
        for var in term_vars:
            bool_var_ms[var][idx1, idx2] = True

    for v, var in product(graph.nodes, wcnf_cfg.get_nullable_symbols()):
        idx = node_to_idx[v]
        bool_var_ms[var][idx, idx] = True

    recently_updated = list(wcnf_cfg.variables)
    while recently_updated:
        updated_var = recently_updated.pop(0)
        for body, heads in vars_body_to_head.items():
            if updated_var not in body:
                continue

            new_matrix: sp.sparse.csc_matrix = (
                bool_var_ms[body[0]] @ bool_var_ms[body[1]]
            )
            for head in heads:
                old_matrix = bool_var_ms[head]
                bool_var_ms[head] += new_matrix
                if (old_matrix != bool_var_ms[head]).count_nonzero():
                    recently_updated.append(head)

    start_var = wcnf_cfg.start_symbol
    if start_var not in bool_var_ms:
        return set()

    return {
        (idx_to_node[idx1], idx_to_node[idx2])
        for idx1, idx2 in zip(*bool_var_ms[start_var].nonzero())
        if idx_to_node[idx1] in start_nodes and idx_to_node[idx2] in final_nodes
    }
