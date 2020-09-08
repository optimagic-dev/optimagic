from pathlib import Path


def create_short_database_names(path_list):
    """Generate short but unique names from each path for each full database path.

    Args:
        path_list (list): Strings or pathlib.Paths to the optimizations' databases.

    Returns:
        short_name_to_path (dict): Mapping from the new unique names to their full path.

    Example:

    >>> pl = ["bla/blubb/blabb.db", "a/b", "bla/blabb"]
    >>> create_short_database_names(pl)
    {'blubb/blabb': 'bla/blubb/blabb.db', 'b': 'a/b', 'bla/blabb': 'bla/blabb'}

    """
    no_suffixes = [Path(p).resolve().with_suffix("") for p in path_list]
    # The assert statement makes sure that the while loop terminates
    assert len(set(no_suffixes)) == len(
        no_suffixes
    ), "path_list must not contain duplicates."
    short_name_to_path = {}
    for path, path_with_suffix in zip(no_suffixes, path_list):
        parts = tuple(reversed(path.parts))
        needed_parts = 1
        candidate = parts[:needed_parts]
        while _causes_name_clash(candidate, no_suffixes):
            needed_parts += 1
            candidate = parts[:needed_parts]

        short_name = "/".join(reversed(candidate))
        short_name_to_path[short_name] = path_with_suffix
    return short_name_to_path


def _causes_name_clash(candidate, path_list, allowed_occurences=1):
    """Determine if candidate leads to a name clash.

    Args:
        candidate (tuple): Tuple with parts of a path.
        path_list (list): List of pathlib.Paths.
        allowed_occurences (int): How often a name can occur before we call it a clash.

    Returns:
        bool

    """
    duplicate_counter = -allowed_occurences
    for path in path_list:
        parts = tuple(reversed(path.parts))
        if len(parts) >= len(candidate) and parts[: len(candidate)] == candidate:
            duplicate_counter += 1
    return duplicate_counter > 0
