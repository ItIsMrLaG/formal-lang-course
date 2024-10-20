from typing import Callable

import networkx as nx
import pyformlang


def swap_cfg_objs(
    cfg: pyformlang.cfg.CFG,
    get_new_obj: Callable,
    obj_if_empty_prod_body=None,
) -> pyformlang.cfg.CFG:
    productions = set()
    for production in cfg.productions:
        production_body = [get_new_obj(obj) for obj in production.body]

        if not production_body and obj_if_empty_prod_body:
            production_body.append(obj_if_empty_prod_body)

        productions.add(pyformlang.cfg.Production(production.head, production_body))

    return pyformlang.cfg.CFG(start_symbol=cfg.start_symbol, productions=productions)


def cfg_to_weak_normal_form(cfg: pyformlang.cfg.CFG) -> pyformlang.cfg.CFG:
    """cfg.to_normal_form() ~ cfg.to_weak_normal_form() if cfg doesn't contain Epsilons
    PLAN:
        1. pyformlang.cfg.Epsilon --> pyformlang.cfg.Terminal
        2. to_normal_form()
        3. pyformlang.cfg.Terminal --> pyformlang.cfg.Epsilon
    """

    _epsilon_terminal = pyformlang.cfg.Terminal("#EPSILON#")
    while _epsilon_terminal in cfg.terminals or _epsilon_terminal in cfg.variables:
        _epsilon_terminal = pyformlang.cfg.Terminal(f"#{_epsilon_terminal.value}#")

    _cfg = swap_cfg_objs(
        cfg=cfg,
        get_new_obj=lambda obj: _epsilon_terminal
        if isinstance(obj, pyformlang.cfg.Epsilon)
        else obj,
        obj_if_empty_prod_body=_epsilon_terminal,
    )

    return swap_cfg_objs(
        cfg=_cfg.to_normal_form(),
        get_new_obj=lambda obj: pyformlang.cfg.Epsilon()
        if obj == _epsilon_terminal
        else obj,
    )


def _revert_cfg(cfg: pyformlang.cfg.CFG) -> tuple[dict, dict]:
    term_to_vars = {}  # [A -> a] to {a: {A}}
    vars_body_to_head = {}  # [A -> BC] to {BC: {A}}
    for production in cfg.productions:
        if len(production.body) == 2:
            vars_body_to_head.setdefault(tuple(production.body), set()).add(
                production.head
            )

        if len(production.body) == 1 and isinstance(
            production.body[0], pyformlang.cfg.Terminal
        ):
            term_to_vars.setdefault(production.body[0], set()).add(production.head)

    return term_to_vars, vars_body_to_head


def hellings_based_cfpq(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    wcnf_cfg = cfg_to_weak_normal_form(cfg)
    term_to_vars, vars_body_to_head = _revert_cfg(wcnf_cfg)

    edges = {
        (v1, pyformlang.cfg.Terminal(lb), v2)
        for (v1, v2, lb) in graph.edges.data("label")
    }

    new_edges = {
        (v1, var, v2)
        for (v1, term, v2) in edges
        if term in term_to_vars
        for var in term_to_vars[term]
    } | {(v, var, v) for v in graph.nodes for var in wcnf_cfg.get_nullable_symbols()}

    q = [edge for edge in new_edges]

    def eval_update(e1, e2, buffer: set):
        s1, a, f1 = e1
        s2, b, f2 = e2
        ab = tuple([a, b])
        if f1 == s2 and ab in vars_body_to_head:
            for var in vars_body_to_head[ab]:
                _new_edge = (s1, var, f2)
                if _new_edge not in new_edges:
                    buffer.add(_new_edge)
                    q.append(_new_edge)

    while q:
        edge1 = q.pop(0)
        buffer = set()
        for edge2 in new_edges:
            eval_update(edge1, edge2, buffer)
            eval_update(edge2, edge1, buffer)
        new_edges |= buffer

    start_var = wcnf_cfg.start_symbol
    return {
        (v1, v2)
        for v1, var, v2 in new_edges
        if v1 in start_nodes and var.value == start_var.value and v2 in final_nodes
    }
