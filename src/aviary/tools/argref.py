import uuid
from collections.abc import Callable, Mapping, Sequence
from functools import update_wrapper
from inspect import signature
from itertools import starmap
from types import UnionType
from typing import Any, Union, get_args, get_origin

from docstring_parser import compose, parse

from aviary.utils import is_coroutine_callable


def make_pretty_id(prefix: str = "") -> str:
    """
    Get an ID that is made using an optional prefix followed by part of an uuid4.

    For example:
    - No prefix: "ff726cd1"
    - With prefix of "foo": "foo-ff726cd1"
    """
    uuid_frags: list[str] = str(uuid.uuid4()).split("-")
    if not prefix:
        return uuid_frags[0]
    return prefix + "-" + uuid_frags[0]


DEFAULT_ARGREF_NOTE = "(Pass a string key instead of the full object)"
LIST_ARGREF_NOTE = "(Pass comma-separated string keys instead of the full object)"


def argref_wrapper(wrapper, wrapped, args_to_skip: set[str]):
    """Inject the ARGREF_NOTE into the Args."""
    # normal wraps
    wrapped_func = update_wrapper(wrapper, wrapped)
    # when we modify wrapped_func's annotations, we don't want to mutate wrapped
    wrapped_func.__annotations__ = wrapped_func.__annotations__.copy()
    # now adjust what we need
    for a in wrapped_func.__annotations__:
        if a in args_to_skip:
            continue
        wrapped_func.__annotations__[a] = str

    orig_annots = wrapped.__annotations__

    # now add note to docstring for all relevant Args
    if wrapped_func.__doc__:
        ds = parse(wrapped_func.__doc__)
        for param in ds.params:
            if param.arg_name in args_to_skip:
                continue

            note = DEFAULT_ARGREF_NOTE

            if (
                param.type_name is None
                and (type_hint := orig_annots.get(param.arg_name)) is not None
            ):
                param.type_name = _type_to_str(type_hint)
                if list in {type_hint, get_origin(type_hint)}:
                    note = LIST_ARGREF_NOTE

            param.description = (param.description or "") + f" {note}"

        wrapped_func.__doc__ = compose(ds)

    return wrapped_func


def argref_by_name(  # noqa: C901, PLR0915
    fxn_requires_state: bool = False,
    prefix: str = "",
    return_direct: bool = False,
    type_check: bool = False,
    args_to_skip: set[str] | None = None,
):
    """Decorator to allow args to be a string key into a refs dict instead of the full object.

    This can prevent LLM-powered tool selections from getting confused by full objects,
    instead it enables them to work using named references. If a reference is not found, it
    will fallback on passing the original argument unless it is the first argument. If the
    first argument str is not found in the state object, it will raise an error.

    Args:
        fxn_requires_state: Whether to pass the state object to the decorated function.
        prefix: A prefix to add to the generated reference ID.
        return_direct: Whether to return the result directly or update the state object.
        type_check: Whether to type-check arguments with respect to the wrapped function's
            type annotations.
        args_to_skip: If provided, a set of argument names that should not be referenced by name.

    Example 1:
        >>> @argref_by_name()  # doctest: +SKIP
        >>> def my_func(foo: float): ...  # doctest: +SKIP

    Example 2:
        >>> def my_func(foo: float, bar: float) -> list[float]:
        ...     return [foo, bar]
        >>> wrapped_fxn = argref_by_name()(my_func)
        >>> # Equivalent to my_func(state.refs["foo"])
        >>> wrapped_fxn("foo", state=state)  # doctest: +SKIP

    Working with lists:
    - If you return a list, the decorator will create a new reference for each item in the list.
    - If you pass multiple args that are strings, the decorator will assume those are the keys.
    - If you need to pass a string, then use a keyword argument.

    Example 1:
        >>> @argref_by_name()  # doctest: +SKIP
        >>> def my_func(foo: float, bar: float) -> list[float]:  # doctest: +SKIP
        ...     return [foo, bar]  # doctest: +SKIP

    Example 2:
        >>> def my_func(foo: float, bar: float) -> list[float]:
        ...     return [foo, bar]
        >>> wrapped_fxn = argref_by_name()(my_func)
        >>> # Returns a multiline string with the new references
        >>> # Equivalent to my_func(state.refs["a"], state.refs["b"])
        >>> wrapped_fxn("a", "b", state=state)  # doctest: +SKIP
    """
    args_to_skip = (args_to_skip or set()) | {"state", "return"}

    def decorator(func):  # noqa: C901, PLR0915
        def get_call_args(*args, **kwargs):  # noqa: C901
            if "state" not in kwargs:
                raise ValueError(
                    "argref_by_name decorated function must have a 'state' argument. "
                    f"Function signature: {func.__name__}({', '.join(func.__annotations__)}) "
                    f" received args: {args} kwargs: {kwargs}"
                )
            # pop the state argument
            state = kwargs["state"] if fxn_requires_state else kwargs.pop("state")

            # now convert the keynames to actual references (if they are a string)
            # tuple is (arg, if was dereferenced)
            def maybe_deref_arg(arg, must_exist: bool) -> tuple[Any, bool]:
                try:
                    refs = state.refs
                except AttributeError as e:
                    raise AttributeError(
                        "The state object must have a 'refs' attribute to use argref_by_name decorator."
                    ) from e

                if arg in refs:
                    return [refs[arg]], True

                # sometimes it is not correctly converted to a tuple
                # so as an attempt to be helpful...
                if (
                    isinstance(arg, str)
                    and len(split_args := [a.strip() for a in arg.split(",")]) > 1
                ):
                    if not (missing := [a for a in split_args if a not in refs]):
                        return [refs[a] for a in split_args], True

                    if must_exist:
                        # Error message for the agent - cast back to comma-separated, since that's the format the agent
                        # is expected to use.
                        raise KeyError(
                            f'The following keys are not present in the current key-value store: "{", ".join(missing)}"'
                        )

                if not must_exist:
                    return arg, False

                # Error message for the agent
                raise KeyError(
                    f"Key is not present in the current key-value store: {arg!r}"
                )

            # the split thing makes it complicated and we cannot use comprehension
            deref_args = []
            for i, arg in enumerate(args):
                # In order to support *args, allow arguments that are either ref keys or strings
                a, dr = maybe_deref_arg(arg, must_exist=False)
                if dr:
                    deref_args.extend(a)
                else:
                    if i == 0 and isinstance(arg, str):
                        # This is a bit of a heuristic, but if the first arg is a string and not found
                        # likely the user intended to use a reference
                        raise KeyError(f"The key {arg} is not found in state.")
                    deref_args.append(a)

            deref_kwargs = {}
            for k, v in kwargs.items():
                if args_to_skip and k in args_to_skip:
                    deref_kwargs[k] = v
                    continue

                # In the kwarg case, force arguments to be ref keys (unless in args_to_skip)
                a, _ = maybe_deref_arg(v, must_exist=True)
                if len(a) > 1:
                    # We got multiple items, so pass the whole list
                    deref_kwargs[k] = a
                else:
                    # We only got one item - pass it directly
                    deref_kwargs[k] = a[0]

            return deref_args, deref_kwargs, state

        def update_state(state, result):
            if return_direct:
                return result
            # if it returns a list, rather than storing the list as a single reference
            # we store each item in the list as a separate reference
            if isinstance(result, list):
                msg = []
                for item in result:
                    new_name = make_pretty_id(prefix)
                    state.refs[new_name] = item
                    msg.append(f"{new_name} ({item.__class__.__name__}): {item!s}")
                return "\n".join(msg)
            new_name = make_pretty_id(prefix)
            state.refs[new_name] = result
            return f"{new_name} ({result.__class__.__name__}): {result!s}"

        def wrapper(*args, **kwargs):
            args, kwargs, state = get_call_args(*args, **kwargs)
            if type_check:
                _check_arg_types(func, args, kwargs)
            result = func(*args, **kwargs)
            return update_state(state, result)

        async def awrapper(*args, **kwargs):
            args, kwargs, state = get_call_args(*args, **kwargs)
            if type_check:
                _check_arg_types(func, args, kwargs)
            result = await func(*args, **kwargs)
            return update_state(state, result)

        if is_coroutine_callable(func):
            awrapper = argref_wrapper(awrapper, func, args_to_skip)
            awrapper.requires_state = True  # type: ignore[attr-defined]
            return awrapper

        wrapper = argref_wrapper(wrapper, func, args_to_skip)
        wrapper.requires_state = True  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _check_arg_types(func: Callable, args, kwargs) -> None:
    annotations = {
        k: v for k, v in func.__annotations__.items() if k not in {"return", "state"}
    }

    sig = signature(func)
    param_names = list(sig.parameters.keys())

    # Elements are tuple of (param, expected, provided)
    wrong_types: list[tuple[str, str, str]] = []

    # Map positional arguments to their parameter names
    for idx, arg in enumerate(args):
        if idx >= len(param_names):
            break  # Extra arguments are handled by *args if any
        param = param_names[idx]
        expected_type = annotations.get(param)
        if expected_type and not _isinstance_with_generics(arg, expected_type):
            wrong_types.append((
                param,
                _type_to_str(expected_type),
                _type_to_str(type(arg)),
            ))

    # Check keyword arguments
    for param, arg in kwargs.items():
        expected_type = annotations.get(param)
        if expected_type and not _isinstance_with_generics(arg, expected_type):
            wrong_types.append((
                param,
                # sometimes need str for generics like Union
                _type_to_str(expected_type),
                _type_to_str(type(arg)),
            ))

    if wrong_types:
        raise TypeError(
            "The following arguments have incorrect types:\n"
            + "\n".join(
                f"- {param}: expected {expected}, got {provided}"
                for param, expected, provided in wrong_types
            )
        )


def _type_to_str(t) -> str:
    """
    Convert a Python type annotation into its string representation.

    Examples:
        type_to_str(int) -> "int"
        type_to_str(Union[int, float]) -> "int | float"
        type_to_str(list[str]) -> "list[str]"
    """
    origin = get_origin(t)
    args = get_args(t)

    if origin is Union:
        # Handle Union types, including the new | syntax in Python 3.10+
        return " | ".join(_type_to_str(arg) for arg in args)
    if origin is not None:
        # Handle generic types like list[str], dict[str, int], etc.
        origin_name = origin.__name__
        args_str = ", ".join(_type_to_str(arg) for arg in args)
        return f"{origin_name}[{args_str}]"
    if hasattr(t, "__name__"):
        # Handle basic types
        return t.__name__
    # Fallback for types without a __name__ attribute
    return str(t)


def _isinstance_with_generics(obj, expected_type) -> bool:  # noqa: C901, PLR0911
    """Like isinstance, but with support for generics."""
    origin = get_origin(expected_type)
    if origin is None:
        # Handle special cases like typing.Any
        if expected_type is Any:
            return True
        return isinstance(obj, expected_type)
    if origin in {UnionType, Union}:
        return any(
            _isinstance_with_generics(obj, arg) for arg in get_args(expected_type)
        )
    if origin in {list, Sequence}:
        if not isinstance(obj, Sequence):
            return False
        elem_type = get_args(expected_type)[0]
        return all(_isinstance_with_generics(elem, elem_type) for elem in obj)
    if origin in {dict, Mapping}:
        if not isinstance(obj, Mapping):
            return False
        key_type, val_type = get_args(expected_type)
        return all(
            _isinstance_with_generics(k, key_type)
            and _isinstance_with_generics(v, val_type)
            for k, v in obj.items()
        )
    if origin is tuple:
        if not isinstance(obj, tuple):
            return False
        elem_types = get_args(expected_type)
        if len(elem_types) == 2 and elem_types[1] is Ellipsis:  # noqa: PLR2004
            # Tuple of variable length
            return all(_isinstance_with_generics(elem, elem_types[0]) for elem in obj)
        if len(elem_types) != len(obj):
            return False
        return all(
            starmap(_isinstance_with_generics, zip(obj, elem_types, strict=True))
        )
    # Fallback to checking the origin type
    return isinstance(obj, origin)
