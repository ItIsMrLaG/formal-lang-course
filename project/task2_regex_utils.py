from pyformlang.finite_automaton import DeterministicFiniteAutomaton
from pyformlang.regular_expression import Regex

__all__ = [
    "regex_to_dfa",
]


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    nfa = Regex(regex).to_epsilon_nfa()
    return nfa.to_deterministic().minimize()
