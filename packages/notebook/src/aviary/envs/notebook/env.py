import asyncio
import contextlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, ClassVar, Generic, Self, TypeAlias, cast

import aiodocker
import nbformat
from aviary.core import Environment, Messages, Tool, ToolRequestMessage
from aviary.message import EnvStateMessage
from jupyter_client.manager import AsyncKernelManager
from nbformat import NotebookNode
from numpy.typing import NDArray
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from . import config as cfg
from . import utils

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar

logger = logging.getLogger(__name__)


class NBEnvironmentState(BaseModel):
    nb_path: Path
    work_dir: Path
    use_docker: bool
    language: utils.NBLanguage

    total_reward: float = 0.0
    done: bool = False
    answer: str | float | int | dict[str, Any] | None = None
    actions: list[str] = Field(default_factory=list)

    # We mark these as PrivateAttrs (despite using them publicly)
    # for two reasons: (A) they are not set at initialization, and
    # (B) we don't want them to be serialized. This is cleaner than
    # setting None defaults.
    _nb: NotebookNode = PrivateAttr()
    _kernel_manager: AsyncKernelManager = PrivateAttr()
    _docker_client: aiodocker.Docker = PrivateAttr()
    _container: aiodocker.containers.DockerContainer = PrivateAttr()

    @model_validator(mode="after")
    def post_init(self) -> Self:
        if not self.nb_path.parent == self.work_dir:
            raise ValueError(
                f"Notebook {self.nb_path} is not in working directory {self.work_dir}"
            )
        if self.nb_path.exists():
            self.reload_nb()
        else:
            self._nb = nbformat.v4.new_notebook()
            self._nb.metadata.kernelspec = self.language.make_kernelspec()
        return self

    def save_nb(self):
        """Saves the notebook to disk."""
        nbformat.write(self._nb, self.nb_path)

    def reload_nb(self):
        """Reloads the notebook from disk."""
        self._nb = nbformat.read(self.nb_path, as_version=4)

    @classmethod
    async def create(cls, **kwargs) -> Self:
        self = cls(**kwargs)
        if self.use_docker:
            await self.start_container()
        else:
            await self.start_kernel()
        return self

    async def start_kernel(self):
        kernel_name = self.language.make_kernelspec()["name"]
        self._kernel_manager = AsyncKernelManager(kernel_name=kernel_name)
        await self._kernel_manager.start_kernel(cwd=str(self.work_dir))

    async def start_container(self):
        self._docker_client = aiodocker.Docker()
        self._container = await self._docker_client.containers.run(
            config={
                "Image": cfg.NB_ENVIRONMENT_DOCKER_IMAGE,
                "Cmd": ["sleep", "infinity"],
                "HostConfig": {"Binds": [f"{self.work_dir}:/workspace"]},
                "WorkingDir": "/workspace",
                "Tty": True,
            }
        )

    @property
    def cells(self) -> list[NotebookNode]:
        return self._nb.cells

    def get_container_path(self, path: Path) -> Path:
        return Path("/workspace") / path.relative_to(self.work_dir)

    async def close(self):
        if self.use_docker:
            # Docker cleanup
            await self._container.stop()
            await self._container.delete()
        else:
            # Kernel cleanup
            await self._kernel_manager.shutdown_kernel()


TListDir: TypeAlias = dict[str, "list[str] | TListDir"]

TNBEnvState = TypeVar(
    "TNBEnvState", bound=NBEnvironmentState, default=NBEnvironmentState
)


class NBEnvironment(Environment[TNBEnvState], Generic[TNBEnvState]):
    NOTEBOOK_NAME: ClassVar[str] = "notebook.ipynb"
    EXEC_TIMEOUT: ClassVar[float | None] = 300.0
    STATE_CLS: ClassVar[type[NBEnvironmentState]] = NBEnvironmentState

    state: TNBEnvState

    def __init__(
        self,
        work_dir: str | os.PathLike,
        nb_path: str | os.PathLike | None = None,
        use_tmp_work_dir: bool = True,
        language: utils.NBLanguage = utils.NBLanguage.PYTHON,
        run_notebook_on_edit: bool = False,
        cell_execution_timeout: int = 600,
        use_docker: bool = cfg.USE_DOCKER,
    ):
        """Initialize a notebook environment.

        Args:
            work_dir: A directory for the environment's assets (notebook, data files, etc.).
                Treat this as an isolated workspace that will be mounted in the container.
            nb_path: Path to the notebook file. If not provided, the notebook will be created
                at work_dir/notebook.ipynb. Note that this must be inside work_dir.
            use_tmp_work_dir: If True (default), the contents of `work_dir` will be copied to a
                temporary work directory.
            language: The programming language of the notebook. Defaults to Python.
            run_notebook_on_edit: If True (default), the whole notebook will be rerun
                after each edit. If False, only the cell that was edited will be rerun.
            cell_execution_timeout: The timeout for each cell execution.
            use_docker: If True, the environment will use a Docker container to run the notebook.
        """
        self.work_dir = Path(work_dir)
        self.nb_path = Path(nb_path) if nb_path else self.work_dir / self.NOTEBOOK_NAME

        self.use_tmp_work_dir = use_tmp_work_dir
        self.language = language
        self.use_docker = use_docker
        # If using docker, we must always run the full notebook
        self.run_notebook_on_edit = run_notebook_on_edit or use_docker
        self.cell_execution_timeout = cell_execution_timeout

    async def reset(self) -> tuple[Messages, list[Tool]]:
        nb_path, work_dir = self._set_work_dir()
        self.state = cast(
            TNBEnvState,
            await self.STATE_CLS.create(
                nb_path=nb_path,
                work_dir=work_dir,
                language=self.language,
                use_docker=self.use_docker,
            ),
        )

        self.tools = [
            Tool.from_function(self.edit_cell),
            Tool.from_function(self.list_workdir),
        ]

        init_obs = cast(Messages, [self.get_env_state_msg()])

        return init_obs, self.tools

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[Messages, float, bool, bool]:
        prev_reward = self.state.total_reward

        obs = cast(
            Messages,
            await self.exec_tool_calls(action, concurrency=False, handle_tool_exc=True),
        )
        reward = self.state.total_reward - prev_reward

        obs = [*obs, self.get_env_state_msg()]

        self.state.actions.append(str(action))
        return obs, reward, self.state.done, False

    # TOOLS

    async def edit_cell(self, contents: str, idx: int | None = None) -> str:
        """Edit the notebook by modifying a specific code cell.

        ONLY CODE CELLS ARE SUPPORTED. Do no attempt to write Markdown or raw text,
        though you are permitted (and encouraged) to write comments in the code cells.
        The cell will be automatically rerun if a successful edit is made.

        WRITE SHORT TO MEDIUM LENGTH CELLS FOR EASE OF DEBUGGING.

        Args:
            contents: Cell contents to insert. We assume the cell is a code block.
            idx: Index of the cell to edit. If not provided (None default),
                then appends a new cell.
        """
        try:
            # Sometimes the agent will try to enter a string instead of an int
            if idx is not None:
                try:
                    idx = int(idx)
                except (ValueError, TypeError):
                    idx = None
            if idx is None or idx >= len(self.state.cells):
                new_cell = nbformat.v4.new_code_cell(source=contents)
                self.state.cells.append(new_cell)
                new_idx = len(self.state.cells) - 1
                return f"Appended new cell (#{new_idx})."

            self.state.cells[idx].source = contents
            return f"Edited cell #{idx}."
        finally:
            self.state.save_nb()
            if self.run_notebook_on_edit:
                args = {}
            else:
                idx = len(self.state.cells) - 1 if idx is None else idx
                args = {"cell_idx": idx}
            await self.run_notebook(**args)

    def list_workdir(self) -> str:
        """Recursively lists the contents of the working directory.

        The contents is represented as a nested JSON dictionary.
        """
        return json.dumps(self._list_dir(self.state.work_dir), indent=2)

    # HELPERS

    def _list_dir(self, path: Path) -> TListDir:
        index: TListDir = {}
        for item in path.iterdir():
            if item.is_dir():
                if "directories" not in index:
                    index["directories"] = {}
                index[item.name] = self._list_dir(item)
            else:
                if "files" not in index:
                    index["files"] = []
                cast(list, index["files"]).append(item.name)
        return index

    async def run_notebook(self, cell_idx: int | None = None) -> str:
        """Run the entire notebook sequentially."""
        logger.debug("Starting notebook execution")
        if self.use_docker:
            if cell_idx is not None:
                raise ValueError("Cell index not supported for Docker")
            return await self._run_notebook_docker()
        return await self._run_notebook_local(cell_idx=cell_idx)

    async def _run_notebook_docker(self) -> str:
        """Run notebook using Docker container."""
        nb_path = str(self.state.get_container_path(self.state.nb_path))

        try:
            exec_command = [
                # Calls nbconvert to run the notebook and updates it inplace
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                nb_path,
                "--allow-errors",  # errors will be put in cell outputs instead of raised
                # "--debug",
            ]

            logger.debug(f"Executing notebook command: {' '.join(exec_command)}")
            exit_code, _, _ = await self._exec_cmd(exec_command)
            if exit_code != 0:
                raise ValueError(
                    f"Error executing the notebook in Docker (exit code: {exit_code})"
                )

        except TimeoutError as e:
            self.state.done = True
            raise TimeoutError(
                f"Notebook execution timed out after {self.EXEC_TIMEOUT} seconds"
            ) from e

        # Now reload from the local file
        self.state.reload_nb()
        return "Executed all cells."

    async def _run_notebook_local(self, cell_idx: int | None = None) -> str:
        """Run notebook using local kernel."""
        try:
            async with asyncio.timeout(self.cell_execution_timeout):
                client = self.state._kernel_manager.client()
                client.start_channels()
                error_messages = await utils.nbformat_run_notebook(  # noqa: F841
                    cells=self.state.cells, client=client, cell_idx=cell_idx
                )
        except TimeoutError as err:
            raise TimeoutError(
                "Notebook execution timed out after"
                f" {self.cell_execution_timeout} seconds"
            ) from err
        # if error_messages:
        #     self.state.notebook_runtime_errors.extend(error_messages)
        self.state.save_nb()
        logger.debug("Saved notebook to disk")
        self.state.reload_nb()
        logger.debug("Reloading notebook from disk")
        return "Executed all cells."

    def get_env_state_msg(self) -> EnvStateMessage:
        nb_path = self.state.get_container_path(self.state.nb_path)
        md_notebook, notebook_images = utils.view_notebook(
            cells=self.state.cells, language=self.language.value
        )
        # Write the markdown representation to disk
        self.state.nb_path.with_suffix(".md").write_text(md_notebook)

        return EnvStateMessage.create_message(
            text=(
                "Markdown representation of notebook contents"
                f" ({nb_path}):\n\n{md_notebook}"
            ),
            images=cast(list[NDArray[Any] | str], notebook_images),
        )

    async def close(self):
        if self.use_docker:
            # HACK: new assets written in /workspace are owned by the docker user, so we
            # cannot shutil.rmtree it. Need to revisit
            # Have to do this since wildcard expansion doesn't work
            if self.use_tmp_work_dir:
                await self._exec_cmd(["sh", "-c", "rm -r /workspace/*"])
            if hasattr(self, "state"):
                await self.state.close()
                await self.state._docker_client.close()
        else:
            if hasattr(self, "state"):
                await self.state.close()
            self._cleanup_tmp_work_dir()

    def _cleanup_tmp_work_dir(self) -> None:
        if self.use_tmp_work_dir:
            with contextlib.suppress(AttributeError):
                shutil.rmtree(self.state.work_dir)

    def _set_work_dir(self) -> tuple[Path, Path]:
        if not self.use_tmp_work_dir:
            return self.nb_path, self.work_dir

        self._cleanup_tmp_work_dir()
        tmp_work_dir = Path(mkdtemp())
        if self.work_dir.exists():
            shutil.copytree(self.work_dir, tmp_work_dir, dirs_exist_ok=True)
        else:
            logger.warning(
                f"Work dir {self.work_dir} does not exist, using empty tmp dir"
            )

        return tmp_work_dir / self.NOTEBOOK_NAME, tmp_work_dir

    async def _exec_cmd(self, cmd: list[str]) -> tuple[int, str, str]:
        """Execute a command in Docker container and return exit code, stdout, and stderr."""
        return await utils.exec_cmd(
            self.state._container, cmd, self.cell_execution_timeout
        )

    def _old_list_workdir(self) -> str:
        """This implementation mimics Unix `tree`. Not clear if this or the JSON rep is better."""

        def get_tree(start_path: Path, prefix: str = "") -> str:
            """Returns a directory tree structure starting from start_path as a string."""
            lines = [f"{prefix}{self.state.get_container_path(start_path).name}/"]
            prefix += "    "
            entries = sorted(start_path.iterdir(), key=lambda e: e.name)
            for index, entry in enumerate(entries):
                connector = "└── " if index == len(entries) - 1 else "├── "
                if entry.is_dir():
                    lines.append(f"{prefix}{connector}{entry.name}/")
                    extension = "    " if index == len(entries) - 1 else "│   "
                    lines.append(get_tree(entry, prefix + extension))
                else:
                    lines.append(f"{prefix}{connector}{entry.name}")

            return "\n".join(lines)

        return get_tree(self.state.work_dir)
