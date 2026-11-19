"""Tools for defining the options of algorithm wrappers.

The module contains what wrappers need beyond plain dataclass fields:

- :func:`option_group` validates a dataclass that bundles related options.
- :class:`DeprecatedOption` and :func:`replace_deprecated_options` keep old option
  names working for one deprecation cycle.
- :func:`resolve_changed_default` announces a default that is about to change.

"""

from __future__ import annotations

import warnings
from dataclasses import MISSING, dataclass, is_dataclass, replace
from typing import Any, Callable, Mapping, TypeVar, get_args

import pydantic
from pydantic_core import core_schema

from optimagic.exceptions import InvalidAlgoOptionError
from optimagic.typing import DEFAULT_PYDANTIC_CONFIG, validated_dataclass

OptionGroupT = TypeVar("OptionGroupT")


def option_group(cls: type[OptionGroupT]) -> type[OptionGroupT]:
    """Add validation to a frozen dataclass that bundles related algorithm options.

    Invalid entries raise an ``InvalidAlgoOptionError`` when the group is created. An
    algorithm field that is annotated with the decorated class only accepts instances
    of the class. Dictionaries are rejected.

    If the group belongs to a feature that can be switched on and off, annotate the
    field of the algorithm with ``bool | <group class>`` instead of adding a switch
    to the group: ``True`` switches the feature on with default entries, and an
    instance switches it on with the given entries.

    Args:
        cls: A frozen dataclass with one attribute per option of the group.

    Returns:
        The class with validation added.

    """

    def make_error(e: pydantic.ValidationError) -> Exception:
        msg = f"The following entries of {cls.__name__} are invalid:\n\n{e}"
        return InvalidAlgoOptionError(msg)

    out = validated_dataclass(config=DEFAULT_PYDANTIC_CONFIG, make_error=make_error)(
        cls
    )

    def get_schema(source: Any, handler: Any) -> core_schema.CoreSchema:
        # Without this, pydantic would also accept a dictionary with the entries of
        # the group wherever the group is used as a field of an algorithm.
        return core_schema.is_instance_schema(out)

    setattr(out, "__get_pydantic_core_schema__", get_schema)  # noqa: B010
    return out


@dataclass(frozen=True)
class DeprecatedOption:
    """An old spelling of an algorithm option that still works but warns."""

    name: str
    """The deprecated name of the option."""

    replacement: str | Callable[[Any, Mapping[str, Any]], dict[str, Any]]
    """The new spelling of the option.

    Either the name of the option that takes over the value unchanged, or a function
    that returns a dictionary of new option names and values. The function is called
    with the deprecated value and with all options that were passed together with it,
    for the case that the new spelling depends on another option.

    An entry of an option group is addressed as ``"<group option>.<entry>"``, e.g.
    ``"trustregion_options.shrinking_factor"``. If the option also accepts a boolean
    (``bool | <group class>``), setting an entry switches the feature on, and the
    option itself can be set to ``True`` or ``False``.

    """

    removed_in: str
    """The optimagic version in which the deprecated name stops working."""


def replace_deprecated_options(options: Mapping[str, Any], algo: Any) -> dict[str, Any]:
    """Replace deprecated algorithm options by their new spelling.

    Every deprecated option that is found emits a ``FutureWarning`` that shows the new
    spelling.

    Args:
        options: The options as passed by the user.
        algo: The algorithm class if the options are used to create an algorithm, or
            the configured algorithm if the options are applied to it. With a
            configured algorithm, entries of option groups that are not replaced keep
            their current values instead of falling back to their defaults.

    Returns:
        The options with all deprecated names replaced.

    Raises:
        InvalidAlgoOptionError: If a deprecated option is combined with the option or
            option group that replaces it, or if two deprecated options translate to
            different values of the same new option.

    """
    out = dict(options)
    present = [dep for dep in algo.__deprecated_options__ if dep.name in out]
    if not present:
        return out

    algo_name = algo.name
    group_updates: dict[str, dict[str, Any]] = {}

    for dep in present:
        value = out.pop(dep.name)
        if isinstance(dep.replacement, str):
            new = {dep.replacement: value}
        else:
            new = dep.replacement(value, options)

        warnings.warn(_deprecation_message(dep, new, algo), FutureWarning, stacklevel=3)

        for target, new_value in new.items():
            group, _, entry = target.partition(".")
            if not entry:
                if target in options:
                    raise _mixed_spelling_error(dep.name, target, algo_name)
                is_contradiction = target in group_updates or (
                    target in out and not _is_same(out[target], new_value)
                )
                if is_contradiction:
                    raise _conflicting_values_error(dep.name, target, algo_name)
                out[target] = new_value
            else:
                if group in options:
                    raise _mixed_spelling_error(dep.name, group, algo_name)
                if group in out:
                    raise _conflicting_values_error(dep.name, group, algo_name)
                updates = group_updates.setdefault(group, {})
                if entry in updates and not _is_same(updates[entry], new_value):
                    raise _conflicting_values_error(dep.name, target, algo_name)
                updates[entry] = new_value

    for group, updates in group_updates.items():
        out[group] = replace(_get_option_group(algo, group), **updates)

    return out


def _get_option_group(algo: Any, group: str) -> Any:
    """Get the option group that entries of deprecated options are merged into.

    This is the current value of the group for a configured algorithm and its default
    for an algorithm class. If the value is not a group but a boolean or None, i.e.
    the option is annotated as ``bool | <group class>``, it is a group with default
    entries.

    """
    if not isinstance(algo, type):
        value = getattr(algo, group)
        algo_cls = type(algo)
    else:
        field = algo.__dataclass_fields__[group]
        has_factory = field.default_factory is not MISSING
        value = field.default_factory() if has_factory else field.default
        algo_cls = algo

    if is_dataclass(value):
        return value

    annotation = algo_cls.__pydantic_fields__[group].annotation
    candidates = [annotation, *get_args(annotation)]
    group_cls = next(c for c in candidates if isinstance(c, type) and is_dataclass(c))
    return group_cls()


def resolve_changed_default(
    value: Any,
    *,
    option: str,
    old_default: Any,
    new_default: Any,
    algo_name: str,
    changed_in: str,
) -> Any:
    """Get the value of an option whose default is about to change.

    During the deprecation cycle the option has the default ``None``, which stands
    for "not set by the user". In this case the old default is used and a
    ``FutureWarning`` announces the new one.

    Args:
        value: The value of the option. ``None`` if the user did not set it.
        option: The name of the option.
        old_default: The default that is still used.
        new_default: The default that will be used from ``changed_in`` on.
        algo_name: The name of the algorithm.
        changed_in: The optimagic version in which the default changes.

    Returns:
        ``value`` if it is not ``None``, otherwise ``old_default``.

    """
    if value is not None:
        return value

    msg = (
        f"The default of the option `{option}` of the algorithm {algo_name} will "
        f"change from {old_default!r} to {new_default!r} in optimagic {changed_in}. "
        f"Set `{option}` explicitly to choose one of them and to silence this warning."
    )
    warnings.warn(msg, FutureWarning, stacklevel=2)
    return old_default


def _deprecation_message(
    dep: DeprecatedOption, new: Mapping[str, Any], algo: Any
) -> str:
    msg = (
        f"The option `{dep.name}` of the algorithm {algo.name} is deprecated and will "
        f"be removed in optimagic {dep.removed_in}. "
    )
    if not new:
        return msg + "The value you passed has no effect."

    flat: dict[str, Any] = {}
    groups: dict[str, dict[str, Any]] = {}
    for target, value in new.items():
        group, _, entry = target.partition(".")
        if entry:
            groups.setdefault(group, {})[entry] = value
        else:
            flat[target] = value

    parts = [f"{name}={value!r}" for name, value in flat.items()]
    for group, entries in groups.items():
        group_cls = type(_get_option_group(algo, group)).__name__
        args = ", ".join(f"{name}={value!r}" for name, value in entries.items())
        parts.append(f"{group}={group_cls}({args})")
    return msg + f"Use `{', '.join(parts)}` instead."


def _mixed_spelling_error(old: str, new: str, algo_name: str) -> InvalidAlgoOptionError:
    msg = (
        f"The deprecated option `{old}` of the algorithm {algo_name} cannot be "
        f"combined with `{new}`, which replaces it. Use only `{new}`."
    )
    return InvalidAlgoOptionError(msg)


def _conflicting_values_error(
    old: str, new: str, algo_name: str
) -> InvalidAlgoOptionError:
    msg = (
        f"The deprecated option `{old}` of the algorithm {algo_name} translates to "
        f"a value of `{new}` that contradicts another deprecated option you passed. "
        f"Use `{new}` instead of the deprecated options."
    )
    return InvalidAlgoOptionError(msg)


def _is_same(left: Any, right: Any) -> bool:
    try:
        return bool(left == right)
    except Exception:
        return left is right
