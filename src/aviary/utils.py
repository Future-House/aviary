import base64
import contextlib
import inspect
import io
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import JsonValue
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.ext.asyncio import AsyncSession as _AsyncSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm.decl_api import _TypeAnnotationMapType
from sqlalchemy.types import JSON

if TYPE_CHECKING:
    import numpy as np


def partial_format(value: str, **formats: dict[str, Any]) -> str:
    """Partially format a string given a variable amount of formats."""
    for template_key, template_value in formats.items():
        with contextlib.suppress(KeyError):
            value = value.format(**{template_key: template_value})
    return value


def encode_image_to_base64(img: "np.ndarray") -> str:
    """Encode an image to a base64 string, to be included as an image_url in a Message."""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Image processing requires the 'image' extra for 'Pillow'. Please:"
            " `pip install aviary[image]`."
        ) from e

    image = Image.fromarray(img)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return (
        f'data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode("utf-8")}'
    )


def is_coroutine_callable(obj) -> bool:
    """Get if the input object is awaitable."""
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return inspect.iscoroutinefunction(obj)
    if callable(obj):
        return inspect.iscoroutinefunction(obj.__call__)
    return False


class Base(DeclarativeBase):
    # Allowing dict to correspond with arbitrary JSON
    type_annotation_map: ClassVar[_TypeAnnotationMapType] = {JsonValue: JSON}


class DBBackend:
    engine: ClassVar[AsyncEngine | None] = None
    Session: ClassVar[async_sessionmaker[_AsyncSession] | None] = None
    uri: ClassVar[str | None] = None

    @classmethod
    async def populate_session(
        cls,
        uri: str | None = None,
        base: type[DeclarativeBase] = Base,
        engine_kwargs: dict[str, Any] | None = None,
        sessionmaker_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if uri is not None:
            if cls.uri is not None:
                raise RuntimeError(
                    "Attempting to initialize DB backend connection, but one is already"
                    " established."
                )
            cls.uri = uri
        assert (
            cls.uri is not None
        ), "Cannot initialize DB backend without specifying a URI."
        cls.engine = engine = create_async_engine(cls.uri, **(engine_kwargs or {}))
        async with engine.begin() as conn:
            await conn.run_sync(base.metadata.create_all)
        cls.Session = async_sessionmaker(bind=engine, **(sessionmaker_kwargs or {}))

    @classmethod
    @asynccontextmanager
    async def begin_session(cls) -> AsyncIterator[_AsyncSession]:
        if cls.Session is None:
            raise RuntimeError(
                "No database connection established. Please call `await"
                " DBBackend.init_connection(...)` or manually populate `cls.Session`"
                " before attempting to persist or rehydrate contexts."
            )
        async with cls.Session.begin() as session:
            yield session
