"""Tools for defining the options of algorithm wrappers.

The module contains what wrappers need beyond plain dataclass fields:

- :func:`option_group` validates a dataclass that bundles related options.
- :class:`DeprecatedOption` and :func:`replace_deprecated_options` keep old option
  names working for one deprecation cycle.
- :func:`resolve_changed_default` announces a default that is about to change.

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence, TypeVar

import pydantic

from optimagic.exceptions import InvalidAlgoOptionError
from optimagic.typing import DEFAULT_PYDANTIC_CONFIG, validated_dataclass

OptionGroupT = TypeVar("OptionGroupT")


def option_group(cls: type[OptionGroupT]) -> type[OptionGroupT]:
    """Add validation to a frozen dataclass that bundles related algorithm options.

    An algorithm field that is annotated with the decorated class accepts an instance
    of the class or a dictionary with the same entries. Invalid entries raise an
    ``InvalidAlgoOptionError`` when the group or the algorithm is created.

    Args:
        cls: A frozen dataclass with one attribute per option of the group.

    Returns:
        The class with validation added.

    """

    def make_error(e: pydantic.ValidationError) -> Exception:
        msg = f"The following entries of {cls.__name__} are invalid:\n\n{e}"
        return InvalidAlgoOptionError(msg)

    return validated_dataclass(config=DEFAULT_PYDANTIC_CONFIG, make_error=make_error)(
        cls
    )


@dataclass(frozen=True)
class DeprecatedOption:
    """An old spelling of an algorithm option that still works but warns."""

    name: str
    """The deprecated name of the option."""

    replacement: str | Callable[[Any], dict[str, Any]]
    """The new spelling of the option.

    Either the name of the option that takes over the value unchanged or a function
    that maps the deprecated value to a dictionary of new option names and values. An
    entry of an option group is addressed as ``"<group option>.<entry>"``, e.g.
    ``"restart_options.use_restarts"``.

    """

    removed_in: str
    """The optimagic version in which the deprecated name stops working."""


def replace_deprecated_options(
    options: Mapping[str, Any],
    deprecated: Sequence[DeprecatedOption],
    algo_name: str,
    current: Any = None,
) -> dict[str, Any]:
    """Replace deprecated algorithm options by their new spelling.

    Every deprecated option that is found emits a ``FutureWarning`` that shows the new
    spelling.

    Args:
        options: The options as passed by the user.
        deprecated: The deprecated options of the algorithm.
        algo_name: The name of the algorithm. Used in warnings and error messages.
        current: The configured algorithm the options are applied to, if there is
            one. Entries of option groups that are not replaced keep the values they
            have in ``current`` instead of falling back to their defaults.

    Returns:
        The options with all deprecated names replaced.

    Raises:
        InvalidAlgoOptionError: If a deprecated option is combined with its new
            spelling or if two deprecated options translate to different values of
            the same new option.

    """
    out = dict(options)
    present = [dep for dep in deprecated if dep.name in out]
    if not present:
        return out

    group_updates: dict[str, dict[str, Any]] = {}

    for dep in present:
        value = out.pop(dep.name)
        if isinstance(dep.replacement, str):
            new = {dep.replacement: value}
        else:
            new = dep.replacement(value)

        warnings.warn(
            _deprecation_message(dep, new, algo_name), FutureWarning, stacklevel=3
        )

        for target, new_value in new.items():
            group, _, entry = target.partition(".")
            if not entry:
                # A converter may set an option as a side effect, e.g. a noise level
                # that implies `noisy=True`. This is compatible with the same value
                # being passed explicitly.
                is_side_effect = not isinstance(dep.replacement, str) and (
                    target in options and _is_same(options[target], new_value)
                )
                if target in options and not is_side_effect:
                    raise _mixed_spelling_error(dep.name, target, algo_name)
                if target in out and not _is_same(out[target], new_value):
                    raise _conflicting_values_error(dep.name, target, algo_name)
                out[target] = new_value
            else:
                if group in options and (
                    not isinstance(options[group], Mapping) or entry in options[group]
                ):
                    raise _mixed_spelling_error(dep.name, group, algo_name)
                updates = group_updates.setdefault(group, {})
                if entry in updates and not _is_same(updates[entry], new_value):
                    raise _conflicting_values_error(dep.name, target, algo_name)
                updates[entry] = new_value

    for group, updates in group_updates.items():
        if group in options:
            out[group] = {**options[group], **updates}
        elif current is not None:
            out[group] = replace(getattr(current, group), **updates)
        else:
            out[group] = updates

    return out


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
    dep: DeprecatedOption, new: Mapping[str, Any], algo_name: str
) -> str:
    msg = (
        f"The option `{dep.name}` of the algorithm {algo_name} is deprecated and will "
        f"be removed in optimagic {dep.removed_in}. "
    )
    if not new:
        return msg + "The value you passed has no effect anymore."

    nested: dict[str, Any] = {}
    for target, value in new.items():
        group, _, entry = target.partition(".")
        if entry:
            nested.setdefault(group, {})[entry] = value
        else:
            nested[target] = value
    spelling = ", ".join(f"{name}={value!r}" for name, value in nested.items())
    return msg + f"Use `{spelling}` instead."


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
