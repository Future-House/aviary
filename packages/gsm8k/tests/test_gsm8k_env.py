import pytest

from aviary.core import Environment, TaskDataset, ToolCall, ToolRequestMessage
from aviary.envs.gsm8k import CalculatorEnv, CalculatorEnvConfig


@pytest.mark.asyncio
async def test_calculator_env() -> None:
    problem = (
        "What is the answer to the ultimate question of life, the universe, and"
        " everything?"
    )
    answer = 42.0
    env = CalculatorEnv(
        problem_id="douglas-adams",
        problem=problem,
        answer=answer,
        config=CalculatorEnvConfig(
            correct_reward=1e4,
        ),
    )

    obs, tools = await env.reset()
    assert obs[0].content == problem
    assert len(tools) == 2

    # Run calculator
    response, reward, done, trunc = await env.step(
        ToolRequestMessage(tool_calls=[ToolCall.from_tool(tools[0], expr="4-3")])
    )
    assert not done
    assert reward == 0.0
    assert response[0].content == "1"

    # check answer
    response, reward, done, trunc = await env.step(
        ToolRequestMessage(tool_calls=[ToolCall.from_tool(tools[1], answer="42")])
    )
    assert reward == 1e4


def test_loading_from_name() -> None:
    env: CalculatorEnv = Environment.from_name(  # type: ignore[assignment]
        "calculator",
        problem_id="rhetorical",
        problem="I had a cake and I ate it. How many cakes do I have?",
        answer=0,
    )
    assert isinstance(env, CalculatorEnv)


@pytest.mark.parametrize(
    ("split", "first_answer"),
    [("train", 10.0), ("train_full", 72.0), ("val", 72.0), ("test", 18.0)],
)
def test_loading_gsm8k_from_name(split: str, first_answer: float) -> None:
    env = TaskDataset.from_name("gsm8k", split=split).get_new_env_by_idx(0)
    assert isinstance(env, CalculatorEnv)
    assert env.answer == first_answer


@pytest.mark.asyncio
async def test_calculator_basic_operations() -> None:
    """Test basic mathematical operations work correctly."""
    env = CalculatorEnv(
        problem_id="test",
        problem="Test problem",
        answer=42.0,
    )
    obs, tools = await env.reset()
    calculator_tool = tools[0]

    # Test basic arithmetic
    test_cases = [
        ("2 + 3", "5"),
        ("10 - 4", "6"),
        ("5 * 6", "30"),
        ("15 / 3", "5"),
        ("2 ** 3", "8"),
        ("17 // 5", "3"),
        ("17 % 5", "2"),
        ("-5", "-5"),
        ("+7", "7"),
    ]

    for expr, expected in test_cases:
        response, reward, done, trunc = await env.step(
            ToolRequestMessage(
                tool_calls=[ToolCall.from_tool(calculator_tool, expr=expr)]
            )
        )
        assert response[0].content == expected, (
            f"Expression {expr} should equal {expected}"
        )
        assert reward == 0.0  # Tool success reward
        assert not done


@pytest.mark.asyncio
async def test_calculator_complex_expressions() -> None:
    """Test complex mathematical expressions work correctly."""
    env = CalculatorEnv(
        problem_id="test",
        problem="Test problem",
        answer=42.0,
    )
    obs, tools = await env.reset()
    calculator_tool = tools[0]

    test_cases = [
        ("(2 + 3) * 4", "20"),
        ("2 * (3 + 4)", "14"),
        ("(10 - 2) / 4", "2"),
        ("2 ** (3 + 1)", "16"),
        ("abs(-10)", "10"),
        ("round(3.7)", "4"),
        ("min(5, 3, 8)", "3"),
        ("max(5, 3, 8)", "8"),
        ("abs(-5) + 3", "8"),
        ("round(7.2 / 2)", "4"),
    ]

    for expr, expected in test_cases:
        response, reward, done, trunc = await env.step(
            ToolRequestMessage(
                tool_calls=[ToolCall.from_tool(calculator_tool, expr=expr)]
            )
        )
        assert response[0].content == expected, (
            f"Expression {expr} should equal {expected}"
        )
        assert reward == 0.0
        assert not done


@pytest.mark.asyncio
async def test_calculator_security_blocks_dangerous_expressions() -> None:
    """Test that dangerous expressions are blocked by the secure evaluator."""
    env = CalculatorEnv(
        problem_id="test",
        problem="Test problem",
        answer=42.0,
        config=CalculatorEnvConfig(tool_failure_reward=-1.0),
    )
    obs, tools = await env.reset()
    calculator_tool = tools[0]

    # Test expressions that should be blocked for security
    dangerous_expressions = [
        "__import__('os').system('echo hello')",
        "exec('print(1)')",
        "eval('2+2')",
        "open('/etc/passwd')",
        "globals()",
        "locals()",
        "__builtins__",
        "dir()",
        "vars()",
        "getattr(int, '__add__')",
        "[x for x in range(10)]",  # List comprehensions
        "lambda x: x + 1",  # Lambda functions
        "print(42)",  # Function calls not in whitelist
        "import os",  # Import statements
    ]

    for expr in dangerous_expressions:
        response, reward, done, trunc = await env.step(
            ToolRequestMessage(
                tool_calls=[ToolCall.from_tool(calculator_tool, expr=expr)]
            )
        )
        # Should return error message, failure reward, and potentially terminate
        assert response[0].content is not None, f"Expression {expr} should have content"
        assert "Error using calculator:" in response[0].content, (
            f"Expression {expr} should be blocked"
        )
        assert reward == -1.0  # Tool failure reward


@pytest.mark.asyncio
async def test_calculator_handles_invalid_syntax() -> None:
    """Test that invalid mathematical syntax is handled gracefully."""
    env = CalculatorEnv(
        problem_id="test",
        problem="Test problem",
        answer=42.0,
        config=CalculatorEnvConfig(tool_failure_reward=-1.0),
    )
    obs, tools = await env.reset()
    calculator_tool = tools[0]

    invalid_expressions = [
        "2 +",  # Incomplete expression
        "* 3",  # Invalid syntax
        "2 3",  # Missing operator
        "((2 + 3)",  # Unmatched parentheses
        "2 + 3)",  # Unmatched parentheses
        "",  # Empty expression
        "abc",  # Undefined variable
        "2 ^ 3",  # Invalid operator (^ is XOR, not power)
    ]

    for expr in invalid_expressions:
        response, reward, done, trunc = await env.step(
            ToolRequestMessage(
                tool_calls=[ToolCall.from_tool(calculator_tool, expr=expr)]
            )
        )
        assert response[0].content is not None, f"Expression {expr} should have content"
        assert "Error using calculator:" in response[0].content, (
            f"Expression {expr} should cause an error"
        )
        assert reward == -1.0


@pytest.mark.asyncio
async def test_calculator_division_by_zero() -> None:
    """Test that division by zero is handled with a user-friendly error message."""
    env = CalculatorEnv(
        problem_id="test",
        problem="Test problem",
        answer=42.0,
        config=CalculatorEnvConfig(tool_failure_reward=-1.0),
    )
    obs, tools = await env.reset()
    calculator_tool = tools[0]

    # Test direct division by zero
    response, reward, done, trunc = await env.step(
        ToolRequestMessage(
            tool_calls=[ToolCall.from_tool(calculator_tool, expr="5 / 0")]
        )
    )
    assert response[0].content is not None, "Division by zero should have content"
    assert "Division by zero is not allowed" in response[0].content, (
        "Should show user-friendly division by zero message"
    )
    assert reward == -1.0

    # Test complex expression with division by zero
    response, reward, done, trunc = await env.step(
        ToolRequestMessage(
            tool_calls=[ToolCall.from_tool(calculator_tool, expr="(2 + 3) / (1 - 1)")]
        )
    )
    assert response[0].content is not None
    assert "Division by zero is not allowed" in response[0].content
    assert reward == -1.0


@pytest.mark.asyncio
async def test_calculator_float_to_int_conversion() -> None:
    """Test that float results are converted to int when possible."""
    env = CalculatorEnv(
        problem_id="test",
        problem="Test problem",
        answer=42.0,
    )
    obs, tools = await env.reset()
    calculator_tool = tools[0]

    # Test cases where float should be converted to int
    test_cases = [
        ("4.0", "4"),  # Direct float
        ("8 / 2", "4"),  # Division resulting in whole number
        ("2.5 * 2", "5"),  # Multiplication resulting in whole number
        ("3.7 + 0.3", "4"),  # Addition resulting in whole number (approximately)
    ]

    for expr, expected in test_cases:
        response, reward, done, trunc = await env.step(
            ToolRequestMessage(
                tool_calls=[ToolCall.from_tool(calculator_tool, expr=expr)]
            )
        )
        assert response[0].content == expected, (
            f"Expression {expr} should be converted to int {expected}"
        )


@pytest.mark.asyncio
async def test_calculator_preserves_float_when_needed() -> None:
    """Test that float results are preserved when they can't be converted to int."""
    env = CalculatorEnv(
        problem_id="test",
        problem="Test problem",
        answer=42.0,
    )
    obs, tools = await env.reset()
    calculator_tool = tools[0]

    # Test cases where float should be preserved
    test_cases = [
        ("7 / 2", "3.5"),
        ("2.5 + 1.2", "3.7"),
        ("3.14159", "3.14159"),
    ]

    for expr, expected in test_cases:
        response, reward, done, trunc = await env.step(
            ToolRequestMessage(
                tool_calls=[ToolCall.from_tool(calculator_tool, expr=expr)]
            )
        )
        assert response[0].content == expected, (
            f"Expression {expr} should preserve float {expected}"
        )
