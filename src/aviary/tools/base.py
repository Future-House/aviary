import inspect
import json
import uuid
from collections.abc import Awaitable, Callable, Iterable
from functools import partial
from itertools import starmap
from typing import Annotated, Any, Literal, NoReturn, Self, TypeAlias

from docstring_parser import DocstringParam, DocstringStyle, parse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FieldSerializationInfo,
    PlainSerializer,
    TypeAdapter,
    create_model,
    field_serializer,
    model_validator,
)
from pydantic.fields import FieldInfo

from aviary.message import Message
from aviary.utils import partial_format

try:
    from dicttoxml import dicttoxml
except ImportError:
    dicttoxml = None

# Mapping from python types to JSON schema types
# SEE: https://json-schema.org/understanding-json-schema/reference/numeric
type_map: dict[type | None, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "list",
    dict: "object",
    None: "null",
}
reverse_type_map = {v: k for k, v in type_map.items()}

# A string to denote an invalid tool. It can be used to indicate
# an attempt to use a non-existent tool, missing/invalid parameters,
# mangled output from the LLM, etc.
INVALID_TOOL_NAME = "INVALID"


class ToolCallFunction(BaseModel):
    arguments: dict[str, Any]
    name: str

    @model_validator(mode="before")
    @classmethod
    def deserialize_args(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data["arguments"], str | None):
            if not data["arguments"]:
                data["arguments"] = {}
            else:
                try:
                    data["arguments"] = json.loads(data["arguments"])
                except json.JSONDecodeError:
                    # If the arguments are not parseable, mark this ToolCall(Function) as invalid
                    # so we can enable "learn"ing what a valid tool call looks like
                    data["name"] = INVALID_TOOL_NAME
                    data["arguments"] = {}

        return data

    @field_serializer("arguments")
    def serialize_arguments(self, arguments: dict[str, Any]) -> str:
        return json.dumps(arguments)

    def __str__(self) -> str:
        arg_str = ", ".join([f"{k}='{v}'" for k, v in self.arguments.items()])
        return f"{self.name}({arg_str})"


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction

    @staticmethod
    def generate_id() -> str:
        """Generate a tool call ID of length 9 with values in [a-zA-Z0-9]."""
        return str(uuid.uuid4()).replace("-", "")[:9]

    @classmethod
    def from_tool(cls, tool: "Tool", *args, id: str | None = None, **kwargs) -> Self:  # noqa: A002
        """Create a ToolCall from a Tool and arguments.

        The *args is packaged into the ToolCallFunction's arguments dict with best effort.
        **kwargs is what is passed to toolcall because we have to use named parameters.
        """
        # convert args to kwargs by matching them with the tool's parameters
        for i, name in enumerate(tool.info.parameters.properties.keys()):
            if i < len(args):
                kwargs[name] = args[i]
        return cls(
            id=id or cls.generate_id(),
            function=ToolCallFunction(name=tool.info.name, arguments=kwargs),
        )

    @classmethod
    def from_name(cls, function_name: str, **kwargs) -> Self:
        return cls(
            id=cls.generate_id(),
            function=ToolCallFunction(name=function_name, arguments=kwargs),
        )

    def __str__(self) -> str:
        arg_str = ", ".join([f"{k}='{v}'" for k, v in self.function.arguments.items()])
        return f"{self.function.name}({arg_str})"


class ToolRequestMessage(Message):
    role: Literal["assistant"] = Field(
        default="assistant", description="Matching LiteLLM structure."
    )
    content: str | None = None
    function_call: None = None
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="List of ToolCalls to make concurrently and independently.",
    )

    def __str__(self) -> str:
        if not self.tool_calls:
            return super().__str__()
        base_msg = f"Tool request message {self.content or ''!r}"
        if len(self.tool_calls) == 1:
            return (
                f"{base_msg} for tool calls: "
                f"{self.tool_calls[0]} [id={self.tool_calls[0].id}]"
            )
        return f"{base_msg} for tool calls: " + "; ".join([
            f"{tc!s} [id={tc.id}]" for tc in self.tool_calls
        ])


class ToolResponseMessage(Message):
    content: str = Field(
        description=(
            "Response message content, required to be a string by OpenAI/Anthropic."
        ),
    )
    role: Literal["tool"] = Field(
        default="tool", description="Matching LiteLLM structure."
    )
    name: str = Field(description="Name of the tool that was called.")
    tool_call_id: str = Field(
        description=(
            "Propagated from ToolCall.id, enabling matching response with"
            " ToolRequestMessage."
        )
    )

    @classmethod
    def from_call(cls, call: ToolCall, content: str) -> Self:
        return cls(content=content, name=call.function.name, tool_call_id=call.id)

    @classmethod
    def from_request(
        cls, request: ToolRequestMessage, contents: Iterable[str]
    ) -> list[Self]:
        return list(
            starmap(cls.from_call, zip(request.tool_calls, contents, strict=True))
        )

    def __str__(self) -> str:
        return (
            f"Tool response message {self.content!r} for tool call ID"
            f" {self.tool_call_id} of tool {self.name!r}"
        )


def dict_serialize_exclude_none(
    value: dict[str, dict[str, Any]], info: FieldSerializationInfo
) -> dict[str, dict[str, Any]]:
    """Work around Pydantic not applying exclude_none to dict serializations."""
    if info.exclude_none:
        return {
            p_name: {k: v for k, v in config.items() if v is not None}
            for p_name, config in value.items()
        }
    return value


class Parameters(BaseModel):
    """Matches LiteLLM's desired "tools" schema."""

    model_config = ConfigDict(extra="allow")

    type: Literal["object"] = "object"
    properties: Annotated[
        dict[str, dict[str, Any]], PlainSerializer(dict_serialize_exclude_none)
    ]
    required: list[str]


class FunctionInfo(BaseModel):
    """
    Function-level (not arg-level) information.

    Matches LiteLLM's desired "tools" schema, and resembles inspect.Signature.
    """

    name: str
    description: str
    # SEE: https://github.com/openai/openai-openapi/blob/0f5de60a3d2b263dc2ac362371673f7a21811874/openapi.yaml#L7567-L7570
    parameters: Parameters

    def describe_str(self) -> str:
        for value in self.parameters.properties.values():
            if value.get("allOf") or not value.get("type"):
                raise NotImplementedError(
                    f"Complex types are not yet supported. Failed on: {self!r}"
                )
        # Start with the function prototype
        prototype = f"{self.name}("
        prototype += ", ".join([
            f"{arg['type']} {name}" for name, arg in self.parameters.properties.items()
        ])
        prototype += ")"

        # Function description
        indented_description_lines = "\n".join([
            f"    {line}" if line else "" for line in self.description.split("\n")
        ])
        description = f"DESCRIPTION:\n{indented_description_lines}\n"

        # Parameters description
        params_description = "PARAMETERS:\n"
        for name, arg in self.parameters.properties.items():
            param_desc = (
                f"    {name} ({arg['type']}):"
                f" {arg.get('description') or 'No description provided.'}\n"
            )
            params_description += param_desc

        # Constructing the full man page
        return (
            f"NAME: {self.name}\n\n"
            f"SYNOPSIS:\n    {prototype}\n\n"
            f"{description}\n{params_description}"
        )

    def describe_xml(self) -> str:
        try:
            return dicttoxml(
                self.model_dump(exclude_none=True, by_alias=True),
                custom_root="function_info",
                attr_type=False,
                xml_declaration=False,
            ).decode()
        except TypeError:
            raise ImportError(
                "XML description requires the 'xml' extra for 'dicttoxml'. Please:"
                " `pip install aviary[xml]`."
            ) from None

    def describe_json(self) -> str:
        return self.model_dump_json(exclude_none=True, by_alias=True)

    def __str__(self):
        return self.describe_str()


def _raises(exc: Exception) -> NoReturn:
    """Work around lambda not supporting raise statement."""
    raise exc


class Tool(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["function"] = "function"
    info: FunctionInfo = Field(
        alias="function",
        description=(
            "The serialization alias of 'function' is to match LiteLLM structure on"
            " serialization, and the validation alias enables deserialization."
        ),
    )

    def __init__(
        self,
        tool_fn: Callable[..., Any] | Callable[..., Awaitable[Any]] = (
            lambda *_, **__: _raises(
                NotImplementedError("Please provide a tool function to call.")
            )
        ),
        **kwargs,
    ):
        super().__init__(**kwargs)
        # NOTE: this Callable is excluded from serialization
        self._tool_fn = tool_fn
        self._force_pickle_fn = False

    def __getstate__(self) -> dict[Any, Any]:
        # Prevent _tool_fn from being pickled, SEE: https://stackoverflow.com/a/2345953
        state = super().__getstate__()
        # allow forcing pickle, e.g., for cloud pickle sending
        if self._force_pickle_fn:
            return state
        state["__dict__"] = state["__dict__"].copy()
        state["__dict__"].pop("_tool_fn", None)
        return state

    @staticmethod
    def _get_param_desc(param: DocstringParam, include_type: bool) -> str:
        if not include_type or not param.type_name:
            return param.description or ""
        return f"({param.type_name}): {param.description or ''}"

    @classmethod
    def from_function(
        cls,
        function: Callable[..., Any] | Callable[..., Awaitable[Any]],
        docstring_style: DocstringStyle = DocstringStyle.AUTO,
        allow_empty_param_descriptions: bool = False,
        types_in_param_descriptions: bool = False,
        **formats,
    ) -> "Tool":
        """Hydrate this class via inspection from a free function with a docstring."""
        fxn_name = function.__name__
        # now we parse descriptions from the docstring
        docstring = parse(function.__doc__, style=docstring_style)  # type: ignore[arg-type]  # SEE: https://github.com/rr-/docstring_parser/issues/88
        if not docstring.description:
            raise ValueError(f"Missing docstring for function {fxn_name}.")
        # now we parse descriptions from the docstring
        try:
            # Don't include anything below \f, matching FastAPI's solution for this
            # SEE: https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#advanced-description-from-docstring
            description_stop_index: int | None = docstring.description.index("\\f")
        except ValueError:
            description_stop_index = None
        field_definitions: dict[str, tuple[type, FieldInfo]] = {}
        required: dict[str, bool] = {}
        annotations = function.__annotations__
        for pname, parameter in inspect.signature(function).parameters.items():
            if pname == "state":
                # NOTE: ToolRequestMessage passes state for us, not the LLM
                continue
            d = next(
                (
                    cls._get_param_desc(
                        p, include_type=types_in_param_descriptions
                    ).replace("\n", " ")
                    for p in docstring.params
                    if p.arg_name == pname
                ),
                "",
            )
            if not d and not allow_empty_param_descriptions:
                raise ValueError(f"Missing description for parameter {pname}.")
            required[pname] = parameter.default == inspect.Parameter.empty
            field_config: dict[str, Any] = {}
            if description := partial_format(d, **formats):
                field_config["description"] = description
            if not required[pname]:
                field_config["default"] = parameter.default

            # Annotation resolution order:
            # 1. function.__annotations__: type-hints in function signature or injected
            #    by argref_by_name. If a function has an opinion on a type hint, take it
            #    at face-value.
            # 2. parameter.annotation - this will descend into wrapped functions. For
            #    argref_by_name, this is undesirabe, since the wrapper overwrites type hints.
            #    Hence, this is second in resolution order.
            field_definitions[pname] = (
                annotations.get(pname) or parameter.annotation or type(None),
                Field(**field_config),  # type: ignore[pydantic-field]
            )

        json_schema = create_model(  # type: ignore[call-overload]
            "FieldDefinitions", **field_definitions
        ).model_json_schema()
        json_schema.pop("title")  # Remove the throwaway model name
        if "required" not in json_schema:
            # The API schema doesn't require this, and gpt-3.5-turbo doesn't
            # need this, but claude-3-haiku-20240307 does
            json_schema["required"] = []
        return cls(
            tool_fn=function,
            info=FunctionInfo(
                name=fxn_name,
                description=partial_format(
                    docstring.description[:description_stop_index].strip(), **formats
                ),
                parameters=json_schema,
            ),
        )


def wraps_doc_only(wrapped):
    """A decorator to copy only the docstring from the wrapped function.

    You cannot use functools wraps directly because it will set the __wrapped__ attribute,
    which causes inspect.signature to inspect the wrapped function instead of the wrapper.

    Usage:
        def my_documented_function(foo):
            '''This is a function that does something with foo.'''
            pass

        @wraps_doc_only(my_documented_function)
        def my_other_function(foo, state):
            pass

    In this example, the second function can have different arguments, types, etc. and only the docstring
    will be copied over.
    """

    def _wraps_doc_only(wrapper, wrapped):
        wrapper.__doc__ = wrapped.__doc__
        return wrapper

    return partial(_wraps_doc_only, wrapped=wrapped)


# Conveniences for deserialization
Messages: TypeAlias = list[ToolRequestMessage | ToolResponseMessage | Message]
MessagesAdapter = TypeAdapter(Messages)
Tools: TypeAlias = list[Tool]
ToolsAdapter = TypeAdapter(Tools)
