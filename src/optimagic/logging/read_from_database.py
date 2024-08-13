"""Functions to generate, load, write to and read from databases.

The functions here are meant for internal use in optimagic, e.g. for logging during
the optimization and reading from the database. They do not require
detailed knowledge of databases in general but some knowledge of the schema
(e.g. table names) of the database we use for logging.

Therefore, users who simply want to read the database should use the functions in
``read_log.py`` instead.

"""


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    """Convert a list of dicts to a dict of lists.

    Args:
        list_of_dicts (list): List of dictionaries. All dictionaries have the same keys.

    Returns:
        dict

    Examples:
        >>> list_of_dicts_to_dict_of_lists([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        {'a': [1, 3], 'b': [2, 4]}

    """
    return {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}


def dict_of_lists_to_list_of_dicts(dict_of_lists):
    """Convert a dict of lists to a list of dicts.

    Args:
        dict_of_lists (dict): Dictionary of lists where all lists have the same length.

    Returns:
        list

    Examples:

        >>> dict_of_lists_to_list_of_dicts({'a': [1, 3], 'b': [2, 4]})
        [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]

    """
    return [
        dict(zip(dict_of_lists, t, strict=False))
        for t in zip(*dict_of_lists.values(), strict=False)
    ]
