from pathlib import Path

from estimagic.dashboard.create_short_database_names import _causes_name_clash
from estimagic.dashboard.create_short_database_names import create_short_database_names


def test_create_short_database_names_no_conflicts_in_last_element():
    inputs = ["a/db1.db", "b/db2.db", "c/db3.csv"]
    expected_keys = ["db1", "db2", "db3"]
    expected = {k: v for k, v in zip(expected_keys, inputs)}
    res = create_short_database_names(inputs)
    assert expected == res


def test_create_short_database_names_different_stems_same_name():
    inputs = ["a/db.db", "b/db.db", "c/db.csv"]
    expected_keys = ["a/db", "b/db", "c/db"]
    expected = {k: v for k, v in zip(expected_keys, inputs)}
    res = create_short_database_names(inputs)
    assert expected == res


def test_create_short_database_names_mixed_stems_mixed_names():
    inputs = ["a/db.db", "a/db2.db", "c/db.csv"]
    expected_keys = ["a/db", "db2", "c/db"]
    expected = {k: v for k, v in zip(expected_keys, inputs)}
    res = create_short_database_names(inputs)
    assert expected == res


def test_causes_name_clash_no_clash():
    candidate = ("a", "db")
    path_list = [Path("b/db"), Path("c/db"), Path("a/db2")]
    expected = False
    res = _causes_name_clash(candidate, path_list)
    assert expected == res


def test_causes_name_clash_with_clash():
    candidate = ("db",)
    path_list = [Path("a/db"), Path("b/db"), Path("c/db2")]
    expected = True
    res = _causes_name_clash(candidate, path_list)
    assert expected == res


# no tests for create_dashboard_link
