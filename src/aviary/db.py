from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from sqlalchemy import Column, DateTime, ForeignKey, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.orm.decl_api import _TypeAnnotationMapType
from sqlalchemy.sql import func
from sqlalchemy.types import JSON

from aviary.utils import DBBackend
from aviary.version import __version__

logger = logging.getLogger(__name__)


class EnvironmentBase(DeclarativeBase):
    # Allowing dict to correspond with arbitrary JSON
    type_annotation_map: ClassVar[_TypeAnnotationMapType] = {JsonValue: JSON}


class EnvironmentDBBackend(DBBackend):
    @classmethod
    async def get_environment_instances(
        cls, name: str | None = None, environment_id: uuid.UUID | None = None
    ) -> list[EnvironmentDBSchema]:
        """Given a name, pull associated environment instance data.

        Args:
            name: The name of the environment this instance is associated with
            environment_id: The UUID of a specific environment instances

        Returns:
           A list of environment UUIDs.
        """
        async with cls.begin_session() as session:
            if environment_id is not None:
                query = select(EnvironmentDB).where(EnvironmentDB.id == environment_id)
            elif name is not None:
                query = select(EnvironmentDB).where(EnvironmentDB.name == name)
            else:
                raise ValueError("Must provide either a name or environment_id")

            results = (await session.execute(query)).scalars().all()
            return [EnvironmentDBSchema.model_validate(result) for result in results]

    @classmethod
    async def get_environment_frames(
        cls, environment_id: uuid.UUID | None, frame_id: uuid.UUID | None
    ) -> list[FrameDBSchema]:
        """Given a environment_id and frame_id data, get environment frame data.

        Args:
            environment_id: The environment instance we are looking for
            frame_id: The specific frame we are looking for

        Returns:
           A list of FrameDB.
        """
        async with cls.begin_session() as session:
            if frame_id is not None:
                query = select(FrameDB).where(FrameDB.id == frame_id)
            elif environment_id is not None:
                query = select(FrameDB).where(FrameDB.environment_id == environment_id)
            else:
                raise ValueError("Must provide either a environment_id or frame_id")
            results = (await session.execute(query)).scalars().all()
            return [FrameDBSchema.model_validate(result) for result in results]

    @classmethod
    async def add_environment_frame(
        cls,
        environment_id: uuid.UUID,
        state: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        frame_id = uuid.uuid4()
        async with cls.begin_session() as session:
            session.add(
                FrameDB(
                    id=frame_id,
                    environment_id=environment_id,
                    state=state,
                    supplemental_data=metadata,
                )
            )
        return frame_id

    @classmethod
    async def add_environment_instance(
        cls,
        name: str,
    ) -> uuid.UUID:
        environment_id = uuid.uuid4()
        async with cls.begin_session() as session:
            session.add(
                EnvironmentDB(id=environment_id, name=name, aviary_version=__version__)
            )
        return environment_id


class EnvironmentDB(EnvironmentBase):
    """Rows here correspond to a single environment trajectory, a single named set of frames."""

    __tablename__ = "environments"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    name: Mapped[str] = mapped_column(index=True)
    aviary_version: Mapped[str] = mapped_column(default=__version__)


class FrameDB(EnvironmentBase):
    """Rows are a snapshot of an environment after each step.

    Note: corresponds to a frame in the renderer object

    """

    __tablename__ = "frames"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    environment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(EnvironmentDB.id), index=True
    )
    state: Mapped[JsonValue]  # Just for human readability
    supplemental_data: Mapped[JsonValue]
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class EnvironmentDBSchema(BaseModel):
    """Sister model for EnvironmentDB."""

    id: uuid.UUID
    created_at: datetime = Field(default_factory=datetime.now)
    name: str
    aviary_version: str = __version__

    model_config = ConfigDict(from_attributes=True)


class FrameDBSchema(BaseModel):
    """Sister model for FrameDB."""

    id: uuid.UUID
    environment_id: uuid.UUID
    state: JsonValue
    supplemental_data: JsonValue
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(from_attributes=True)
