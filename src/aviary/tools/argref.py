import uuid
from functools import partial, update_wrapper

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


ARGREF_NOTE = "(set via a string key instead of the full object)"


def argref_wrapper(wrapper, wrapped):
    """Inject the ARGREF_NOTE into the Args."""
    # normal wraps
    wrapped_func = update_wrapper(wrapper, wrapped)
    # now adjust what we need
    for a in wrapped_func.__annotations__:
        if a in {"return", "state"}:
            continue
        wrapped_func.__annotations__[a] = str

    # now add note to docstring for all relevant Args
    ds = wrapped_func.__doc__
    if ds and "Args:" in ds:
        arg_doc = ds.split("Args:")[1].split("Returns:")[0].split("\n")
        for line in arg_doc:
            if line.strip():  # Filter whitespace
                ds = ds.replace(line, " ".join((line, ARGREF_NOTE)))
    wrapped_func.__doc__ = ds
    return wrapped_func


def argref_wraps(wrapped):
    """Enable decorator syntax with argref_wrapper."""
    return partial(argref_wrapper, wrapped=wrapped)


def argref_by_name(  # noqa: C901
    fxn_requires_state: bool = False, prefix: str = "", return_direct: bool = False
):
    """Decorator to allow args to be a string key into a refs dict instead of the full object.

    This can prevent LLM-powered tool selections from getting confused by full objects,
    instead it enables them to work using named references.

    Args:
        fxn_requires_state: Whether to pass the state object to the decorated function.
        prefix: A prefix to add to the generated reference ID.
        return_direct: Whether to return the result directly or update the state object.

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

    def decorator(func):  # noqa: C901
        def get_call_args(*args, **kwargs):
            if "state" not in kwargs:
                raise ValueError(
                    "argref_by_name decorated function must have a 'state' argument. "
                    f"Function signature: {func.__name__}({', '.join(func.__annotations__)}) "
                    f" received args: {args} kwargs: {kwargs}"
                )
            # pop the state argument
            state = kwargs["state"] if fxn_requires_state else kwargs.pop("state")

            # pop all string arguments
            other_args = []
            keyname_args = []
            deref_args = []
            largs = list(args)
            while largs and isinstance(largs[0], str):
                keyname_args.append(largs.pop(0))
            if largs:
                other_args.extend(largs)
            if not keyname_args and kwargs:
                name = next(iter(kwargs))
                keyname_args = [kwargs.pop(name)]

            # now convert the keynames to actual references
            for arg in keyname_args:
                try:
                    if arg in state.refs:
                        deref_args.append(state.refs[arg])
                    # sometimes it is not correctly converted to a tuple
                    # so as an attempt to be helpful...
                    elif all(a.strip() in state.refs for a in arg.split(",")):
                        deref_args.extend([
                            state.refs[a.strip()] for a in arg.split(",")
                        ])
                    else:
                        raise KeyError(
                            f"Key '{arg}' not found in state. Available keys: {list(state.refs.keys())}"
                        )
                except AttributeError as e:
                    raise AttributeError(
                        "The state object must have a 'refs' attribute to use argref_by_name decorator."
                    ) from e
            return deref_args, other_args, kwargs, state

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

        @argref_wraps(func)
        def wrapper(*args, **kwargs):
            args, other_args, kwargs, state = get_call_args(*args, **kwargs)
            result = func(*args, *other_args, **kwargs)
            return update_state(state, result)

        @argref_wraps(func)
        async def awrapper(*args, **kwargs):
            args, other_args, kwargs, state = get_call_args(*args, **kwargs)
            result = await func(*args, *other_args, **kwargs)
            return update_state(state, result)

        wrapper.requires_state = True
        awrapper.requires_state = True
        if is_coroutine_callable(func):
            return awrapper
        return wrapper

    return decorator
