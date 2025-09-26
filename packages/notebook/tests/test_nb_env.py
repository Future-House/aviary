import base64
import json
import logging
import os
import pathlib
import shutil
import subprocess
import tempfile

import nbformat.v4 as nbf
import pytest
from aviary.message import EnvStateMessage

from aviary.envs.notebook import NBEnvironment
from aviary.envs.notebook.config import NB_ENVIRONMENT_DOCKER_IMAGE
from aviary.envs.notebook.utils import (
    NBLanguage,
    limit_notebook_output,
    process_cell_output,
    view_notebook,
)

logger = logging.getLogger(__name__)

IN_GITHUB_ACTIONS: bool = os.getenv("GITHUB_ACTIONS") == "true"

OUTPUT_TRUNCATED_STRING = "<...output truncated to"


class TestNBEnvironment:
    def docker_image_exists(self) -> bool:
        try:
            docker_path = shutil.which("docker")
            if not docker_path:
                logger.info("Docker is not installed on this system")
                return False
            result = subprocess.run(  # noqa: S603 override untrusted input flag
                [docker_path, "images", NB_ENVIRONMENT_DOCKER_IMAGE, "-q"],
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout.strip():
                logger.info(f"Docker image {NB_ENVIRONMENT_DOCKER_IMAGE} found")
                return True
            logger.info(f"Docker image {NB_ENVIRONMENT_DOCKER_IMAGE} not found")
            return False  # noqa: TRY300 ruff contradicts itself

        except subprocess.CalledProcessError:
            logger.info("Docker is not available on this system")
            return False

    def should_skip_docker_test(self, use_docker: bool) -> bool:
        if use_docker and (IN_GITHUB_ACTIONS or not self.docker_image_exists()):
            logger.info(
                f"Skipping docker test in CI environment: {IN_GITHUB_ACTIONS} and use_docker={use_docker}"
            )
            return True
        return False

    def should_skip_r_test(self, language: NBLanguage) -> bool:
        """Helper method to determine if R tests should be skipped."""
        if language != NBLanguage.R:
            return False

        try:
            from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel

            kernel_spec_manager = KernelSpecManager()
            kernel_spec_manager.get_kernel_spec("ir")
        except (NoSuchKernel, ImportError):
            return True
        else:
            return False

    @pytest.mark.parametrize("use_docker", [False, True])
    @pytest.mark.parametrize("language", [NBLanguage.PYTHON, NBLanguage.R])
    @pytest.mark.asyncio
    async def test_notebook_env(self, use_docker: bool, language: NBLanguage):
        if self.should_skip_docker_test(use_docker):
            pytest.skip(
                "Skipping docker test because in CI, docker is not available or image is not found"
            )

        if self.should_skip_r_test(language):
            pytest.skip("R kernel is not available")

        if language == NBLanguage.PYTHON:
            create_plot_instruction = (
                "import matplotlib.pyplot as plt\n"
                "import numpy as np\n\n"
                "# Generate random data\n"
                "x = np.random.rand(10)\n"
                "y = np.random.rand(10)\n\n"
                "# Create a basic plot\n"
                "plt.plot(x, y, 'o')\n"
                'plt.xlabel("X-axis")\n'
                'plt.ylabel("Y-axis")\n'
                'plt.title("Random Data Plot")\n'
                "plt.show()"
            )
        else:  # R
            create_plot_instruction = (
                "# Generate random data\n"
                "x <- runif(10)\n"
                "y <- runif(10)\n\n"
                "# Create a basic plot\n"
                "plot(x, y, pch=16,\n"
                '     xlab="X-axis",\n'
                '     ylab="Y-axis",\n'
                '     main="Random Data Plot")'
            )
        # Test that the environment can be reset and edited
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pathlib.Path(tmp)

            env = NBEnvironment(
                work_dir=work_dir, nb_path=work_dir / "test.ipynb", language=language
            )
            env.use_docker = use_docker
            obs, tools = await env.reset()

            assert len(obs) == 1
            assert isinstance(obs[0], EnvStateMessage)
            assert isinstance(obs[0].content, str)
            assert obs[0].content.count("Cell") == 0

            # Will edit the cell and run the notebook
            await env.edit_cell(contents="print('hello')")
            current_nb_contents, images = view_notebook(
                cells=env.state.cells, language=env.language.value
            )
            assert current_nb_contents.count("Cell") == 1
            assert current_nb_contents.count("Output") == 1
            # once in code cell, once in output
            assert current_nb_contents.count("hello") == 2
            assert not images

            # Will create a new cell with a plot
            await env.edit_cell(contents=create_plot_instruction)
            current_nb_contents, images = view_notebook(
                cells=env.state.cells, language=env.language.value
            )
            assert current_nb_contents.count("Cell") == 2
            assert current_nb_contents.count("Output") == 2
            assert current_nb_contents.count("hello") == 2
            assert current_nb_contents.count("Random Data Plot") == 1
            assert current_nb_contents.count("<1>") == 1
            assert len(images) == 1
            assert isinstance(images[0], str)

            state_message = json.loads(env.get_env_state_msg().content or "")

            # One image one text
            assert len(state_message) == 2
            assert state_message[0]["type"] == "image_url"
            assert state_message[1]["type"] == "text"

            # Will edit the last cell with a plot so there should be no plot
            await env.edit_cell(contents="print('Goodbye')", idx=1)
            current_nb_contents, images = view_notebook(
                cells=env.state.cells, language=env.language.value
            )
            assert current_nb_contents.count("Cell") == 2
            assert current_nb_contents.count("Output") == 2
            assert current_nb_contents.count("hello") == 2
            assert current_nb_contents.count("Random Data Plot") == 0
            assert current_nb_contents.count("<1>") == 0
            assert current_nb_contents.count("Goodbye") == 2
            assert not images

            state_message = json.loads(env.get_env_state_msg().content or "")
            assert len(state_message) == 1
            assert state_message[0]["type"] == "text"

            await env.close()

    @pytest.mark.asyncio
    async def test_notebook_utils_integration(self):
        """Integration test for notebook processing with multiple cell types."""

        # Helper function to create a base64 encoded test image
        def create_test_image_base64():
            # Create a small 1x1 pixel PNG image
            return base64.b64encode(
                bytes.fromhex(
                    "89504E470D0A1A0A0000000D4948445200000001000000010804000000B51C0C020000000B4944415478DA636000000002000168DC0B380000000049454E44AE426082"
                )
            ).decode()

        # Test 1: Basic notebook with different output types
        cells = []

        # Code cell with stream output
        cell1 = nbf.new_code_cell(source='print("hello")')
        cell1.outputs = [
            nbf.new_output(output_type="stream", name="stdout", text="hello\n")
        ]
        cells.append(cell1)

        # Code cell with execute result
        cell2 = nbf.new_code_cell(source="2 + 2")
        cell2.outputs = [
            nbf.new_output(
                output_type="execute_result",
                data={"text/plain": "4"},
                execution_count=1,
            )
        ]
        cells.append(cell2)

        # Code cell with error output
        long_traceback = [
            "Traceback (most recent call last):",
            '  File "fake_file.py", line 123\n    some code\n' * 3000,
            "ValueError: Test error",
        ]
        cell3 = nbf.new_code_cell(source='raise ValueError("Test error")')
        cell3.outputs = [
            nbf.new_output(
                output_type="error",
                ename="ValueError",
                evalue="Test error",
                traceback=long_traceback,
            )
        ]
        cells.append(cell3)

        # Code cell with image output
        test_image = create_test_image_base64()
        cell4 = nbf.new_code_cell(source="display(plt.plot([1,2,3]))")
        cell4.outputs = [
            nbf.new_output(
                output_type="display_data",
                data={
                    "image/png": test_image,
                    "text/plain": "<Figure size 640x480 with 1 Axes>",
                },
            )
        ]
        cells.append(cell4)

        # Code cell with very long stream output
        long_output = "x" * 10000
        cell5 = nbf.new_code_cell(source='print("x" * 10000)')
        cell5.outputs = [
            nbf.new_output(output_type="stream", name="stdout", text=long_output)
        ]
        cells.extend([cell5, nbf.new_markdown_cell(source="# Test Markdown")])

        # Process the notebook
        md_content, images = view_notebook(cells, NBLanguage.PYTHON)

        # Verify the output
        assert "hello" in md_content
        assert "4" in md_content
        assert "ValueError: Test error" in md_content
        assert OUTPUT_TRUNCATED_STRING in md_content  # For long outputs
        assert "<1>" in md_content  # Image reference
        assert len(images) == 1
        assert images[0].startswith("data:image/png;base64,")
        assert "# Test Markdown" in md_content

    @pytest.mark.asyncio
    async def test_process_cell_output_stream(self):
        """Test stream output processing with length limits."""
        md: list[str] = []
        images: list[str] = []
        cell_streams: list[str] = []

        # Test normal stream output
        stream_output = nbf.new_output(
            output_type="stream", name="stdout", text="test stream"
        )
        process_cell_output(stream_output, md, images, cell_streams)
        assert cell_streams == ["test stream"]

        # Test very long stream output
        long_stream = "x" * 10000
        long_stream_output = nbf.new_output(
            output_type="stream", name="stdout", text=long_stream
        )
        process_cell_output(long_stream_output, md, images, cell_streams)
        # The stream output is collected in cell_streams and limited later in view_notebook
        assert len(cell_streams) == 2
        assert cell_streams[0] == "test stream"
        assert cell_streams[1] == long_stream

        # Test that the output is limited when processed through limit_notebook_output
        combined_stream = "\n".join(cell_streams)
        limited_output = limit_notebook_output(combined_stream)
        assert OUTPUT_TRUNCATED_STRING in limited_output
        assert len(limited_output) < len(combined_stream)
        assert len(limited_output) < 10000

    @pytest.mark.asyncio
    async def test_process_cell_output_execute_result(self):
        """Test execute result processing with length limits."""
        md: list[str] = []
        images: list[str] = []
        cell_streams: list[str] = []

        # Test normal execute result
        execute_result = nbf.new_output(
            output_type="execute_result", data={"text/plain": "normal output"}
        )
        process_cell_output(execute_result, md, images, cell_streams)
        assert "normal output" in md[-1]

        # Test very long execute result
        long_result = "x" * 10000
        long_execute_result = nbf.new_output(
            output_type="execute_result", data={"text/plain": long_result}
        )
        process_cell_output(long_execute_result, md, images, cell_streams)
        assert OUTPUT_TRUNCATED_STRING in md[-1]
        assert len(md[-1]) < 10000

    @pytest.mark.asyncio
    async def test_process_cell_output_error(self):
        """Test error output processing with length limits."""
        md: list[str] = []
        images: list[str] = []
        cell_streams: list[str] = []

        # Test normal error output
        error_output = nbf.new_output(
            output_type="error",
            ename="ValueError",
            evalue="Test error",
            traceback=["Traceback (most recent call last):", "ValueError: Test error"],
        )
        process_cell_output(error_output, md, images, cell_streams)
        assert "ValueError: Test error" in md[-1]

        # Test very long error traceback
        long_traceback = (
            ["Traceback (most recent call last):"]
            + ['  File "test.py", line {i}\n    test code\n' for i in range(1000)]
            + ["ValueError: Test error"]
        )
        long_error_output = nbf.new_output(
            output_type="error",
            ename="ValueError",
            evalue="Test error",
            traceback=long_traceback,
        )
        process_cell_output(long_error_output, md, images, cell_streams)
        assert OUTPUT_TRUNCATED_STRING in md[-1]
        assert len(md[-1]) < len("\n".join(long_traceback))

    @pytest.mark.asyncio
    async def test_process_cell_output_display_data(self):
        """Test display data processing with various data types and length limits."""
        md: list[str] = []
        images: list[str] = []
        cell_streams: list[str] = []

        def create_test_image_base64():
            return base64.b64encode(
                bytes.fromhex(
                    "89504E470D0A1A0A0000000D4948445200000001000000010804000000B51C0C020000000B4944415478DA636000000002000168DC0B380000000049454E44AE426082"
                )
            ).decode()

        test_image = create_test_image_base64()

        # Test normal display data with image
        display_data = nbf.new_output(
            output_type="display_data",
            data={"image/png": test_image, "text/plain": "Normal text"},
        )
        process_cell_output(display_data, md, images, cell_streams)
        assert len(images) == 1
        assert images[0].startswith("data:image/png;base64,")
        assert f"<{len(images)}>" in md[-1]

        # Test display data with very long text
        long_text = "x" * 10000
        long_display_data = nbf.new_output(
            output_type="display_data", data={"text/plain": long_text}
        )
        process_cell_output(long_display_data, md, images, cell_streams)
        assert OUTPUT_TRUNCATED_STRING in md[-1]
        assert len(md[-1]) < 10000

        # Test display data with ignored types
        html_display_data = nbf.new_output(
            output_type="display_data",
            data={
                "text/html": "<p>HTML content</p>",
                "text/latex": "\\text{LaTeX content}",
                "text/markdown": "# Markdown content",
            },
        )
        initial_md_len = len(md)
        initial_images_len = len(images)
        process_cell_output(html_display_data, md, images, cell_streams)
        assert len(md) == initial_md_len
        assert len(images) == initial_images_len

    @pytest.mark.asyncio
    async def test_process_cell_output_mixed_types(self):
        """Test processing of cells with multiple output types."""
        md: list[str] = []
        images: list[str] = []
        cell_streams: list[str] = []

        # Create a mixed output with stream and execute result
        stream_output = nbf.new_output(
            output_type="stream", name="stdout", text="stream output\n"
        )
        execute_result = nbf.new_output(
            output_type="execute_result", data={"text/plain": "execute result"}
        )

        process_cell_output(stream_output, md, images, cell_streams)
        process_cell_output(execute_result, md, images, cell_streams)

        assert "stream output" in cell_streams[0]
        assert "execute result" in md[-1]
