import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from aviary.env import Frame

try:
    from boto3 import client
except ImportError:
    client = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class Renderer(BaseModel):
    id: UUID | int | str = Field(
        default_factory=lambda: str(uuid4()).replace("-", "")[:16]
    )
    frames: list[Frame] = []
    prefix: str
    name: str = Field(
        default="Trajectory",
        description="Name of the renderer, used in the manifest file.",
    )

    @field_validator("prefix")
    @classmethod
    def check_prefix_is_alphanum(cls, v: str) -> str:
        if not v.isalnum():
            raise ValueError("Prefix must be an alphanumeric string")
        return v

    def append(self, frame: Frame) -> None:
        self.frames.append(frame)

    def _make_filename(self, index: int) -> str:
        return f"{self.prefix}_{self.id!s}_{index}.json"

    def _render(self) -> dict[str, dict[str, Any]]:
        """Get a mapping of filenames to serialized frame with renderer metadata."""
        return {
            self._make_filename(i): frame.model_dump()
            | {
                "index": i + 1,
                "prev_frame": self._make_filename(i - 1) if i > 0 else None,
                "next_frame": (
                    self._make_filename(i + 1) if i + 1 < len(self.frames) else None
                ),
            }
            for i, frame in enumerate(self.frames)
        }

    def build(  # noqa: C901
        self,
        build_dir: str | os.PathLike,
        indent: int = 4,
        r2_bucket: str | None = None,
        extra_files: list[str | os.PathLike] | None = None,
    ) -> None:
        name_to_data = self._render()
        if not name_to_data:
            logger.warning("No frames to render.")
            return
        build_dir_path = Path(build_dir)
        file_list = []
        for name, data in name_to_data.items():
            path = build_dir_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:
                json.dump(data, f, indent=indent)
            file_list.append(name)

        # now write manifest (summary) file, and we know name exists
        # because we check name_to_data has values
        # pylint: disable-next=undefined-loop-variable
        first_name, last_name = next(iter(name_to_data.keys())), name
        # NOTE: we have the '-info' prefix so that we can use a prefix-filter
        manifest_fn = f"{self.prefix}_info_{self.id!s}.json"
        with (build_dir_path / manifest_fn).open("w") as f:
            json.dump(
                {"name": self.name, "first": first_name, "last": last_name},
                f,
                indent=indent,
            )
        file_list.append(manifest_fn)

        # copy the extra files
        if extra_files:
            for p in extra_files:
                path = Path(p)
                output_path = build_dir_path / path.name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copyfile(path, output_path)
                    file_list.append(output_path.name)
                except FileNotFoundError:
                    logger.warning(f"Failed to copy file from {path} to {output_path}.")

        if r2_bucket:  # Upload to bucket
            try:
                CF_ACCOUNT_ID = os.environ["CF_ACCOUNT_ID"]
                CF_ACCESS_KEY_ID = os.environ["CF_ACCESS_KEY_ID"]
                CF_SECRET_ACCESS_KEY = os.environ["CF_SECRET_ACCESS_KEY"]
                if not (CF_ACCOUNT_ID and CF_ACCESS_KEY_ID and CF_SECRET_ACCESS_KEY):
                    raise ValueError("Empty Cloudflare R2 credentials")  # noqa: TRY301
            except (KeyError, ValueError) as exc:
                raise ValueError("Cloudflare R2 credentials unset.") from exc
            try:
                s3 = client(
                    service_name="s3",
                    endpoint_url=f"https://{CF_ACCOUNT_ID}.r2.cloudflarestorage.com",
                    aws_access_key_id=CF_ACCESS_KEY_ID,
                    aws_secret_access_key=CF_SECRET_ACCESS_KEY,
                    region_name="auto",
                )
            except TypeError:
                raise ImportError(
                    "Bucket uploads requires the 'cloud' extra for 'boto3'. Please:"
                    " `pip install aviary[cloud]`."
                ) from None

            for fn in file_list:
                with (build_dir_path / fn).open("rb") as d:
                    s3.upload_fileobj(d, r2_bucket, fn)
