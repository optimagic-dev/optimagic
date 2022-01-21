REGISTRY = {
    list: {
        "flatten": lambda tree: (tree, None),
        "unflatten": lambda aux_data, children: children,
    },
    dict: {
        "flatten": lambda tree: (list(tree.values()), list(tree)),
        "unflatten": lambda aux_data, children: dict(zip(aux_data, children)),
    },
    tuple: {
        "flatten": lambda tree: (list(tree), None),
        "unflatten": lambda aux_data, children: tuple(children),
    },
}


def tree_flatten(tree):
    flat = _tree_just_flatten(tree)
    dummy_flat = ["*"] * len(flat)
    treedef = tree_unflatten(tree, dummy_flat)
    return flat, treedef


def _tree_just_flatten(tree):
    out = []
    tree_type = type(tree)

    if tree_type not in REGISTRY:
        out.append(tree)
    else:
        subtrees, info = REGISTRY[tree_type]["flatten"](tree)
        for subtree in subtrees:
            if type(subtree) in REGISTRY:
                out += _tree_just_flatten(subtree)
            else:
                out.append(subtree)
    return out


def tree_unflatten(tree, flat):
    flat = iter(flat)
    tree_type = type(tree)

    if tree_type not in REGISTRY:
        return next(flat)
    else:
        items, info = REGISTRY[tree_type]["flatten"](tree)
        unflattened_items = []
        for item in items:
            if type(item) in REGISTRY:
                unflattened_items.append(tree_unflatten(item, flat))
            else:
                unflattened_items.append(next(flat))
        return REGISTRY[tree_type]["unflatten"](info, unflattened_items)


def tree_map(func, tree):
    flat, treedef = tree_flatten(tree)
    modified = [func(i) for i in flat]
    return tree_unflatten(treedef, modified)


def tree_multimap(func, *trees):
    flat_trees, treedefs = [], []
    for tree in trees:
        flat, treedef = tree_flatten(tree)
        flat_trees.append(flat)
        treedefs.append(treedef)

    for treedef in treedefs:
        if treedef != treedefs[0]:
            raise ValueError("All trees must have the same structure.")

    modified = [func(*item) for item in zip(*flat_trees)]

    out = tree_unflatten(treedefs[0], modified)
    return out
