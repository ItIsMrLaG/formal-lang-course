from pathlib import Path
from typing import Tuple

import networkx as nx
import pytest

from project.task1_graph_utils import get_graph_meta_by_name
from project.task1_graph_utils import read_graph_from_dot
from project.task1_graph_utils import save_labeled_two_cycles_graph_to_dot
from project.task1_graph_utils import get_graph_meta_from_json

PATH_TO_RES_DIR = Path("resources/res_task1_graph_utils")
PATH_TO_RES_DIR_DATASET = PATH_TO_RES_DIR / "dataset_graphs"


def test_get_graph_meta_by_name_on_unknown_graph():
    with pytest.raises(FileNotFoundError):
        get_graph_meta_by_name("strange name")


@pytest.mark.parametrize(
    "name,expected_as_json",
    [
        pytest.param("wc", PATH_TO_RES_DIR_DATASET / "wc_graph_meta_result.json", id="wc"),
        pytest.param("atom", PATH_TO_RES_DIR_DATASET / "atom_graph_meta_result.json", id="atom"),
        pytest.param("core", PATH_TO_RES_DIR_DATASET / "core_graph_meta_result.json", id="core"),
    ],
)
def test_get_graph_meta_by_name_on_existing_in_dataset_graphs(name: str, expected_as_json: Path):
    expected_meta = get_graph_meta_from_json(expected_as_json)
    result_meta = get_graph_meta_by_name(name)
    assert expected_meta == result_meta


def test_save_labeled_two_cycles_graph_to_dot_with_zero_cycle_sizes(tmp_path):
    test_path = tmp_path / "tmp_file.dot"
    with pytest.raises(IndexError):
        save_labeled_two_cycles_graph_to_dot((0, 0), ('fstG', 'sndG'), test_path)


@pytest.mark.parametrize(
    "sizes,labels,expected_graph_as_dot",
    [
        pytest.param((2, 2), ("fstG", "sndG"), PATH_TO_RES_DIR / "two_cycles_labeled_small_graph.dot",
                     id="small graph"),
        pytest.param((4, 5), ("fstG", "sndG"), PATH_TO_RES_DIR / "two_cycles_labeled_medium_graph.dot",
                     id="medium graph"),
    ],
)
def test_save_labeled_two_cycles_graph_to_dot(
        sizes: Tuple[int, int],
        labels: Tuple[str, str],
        expected_graph_as_dot: Path, tmp_path
):
    test_path = tmp_path / "tmp_file.dot"
    expected_graph = nx.DiGraph(read_graph_from_dot(expected_graph_as_dot))

    save_labeled_two_cycles_graph_to_dot(sizes, labels, test_path)
    result_graph = nx.DiGraph(read_graph_from_dot(test_path))

    assert nx.is_isomorphic(
        expected_graph,
        result_graph,
        edge_match=dict.__eq__,
        node_match=dict.__eq__
    )
