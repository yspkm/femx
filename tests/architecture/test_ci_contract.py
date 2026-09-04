from pathlib import Path

import pytest

pytestmark = pytest.mark.architecture

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"


def _checkout_step_blocks(workflow: str) -> tuple[str, ...]:
    lines = workflow.splitlines()
    starts = [index for index, line in enumerate(lines) if "uses: actions/checkout@" in line]
    blocks: list[str] = []
    for start in starts:
        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if lines[index].startswith("      - name:")
            ),
            len(lines),
        )
        blocks.append("\n".join(lines[start:end]))
    return tuple(blocks)


def test_every_ci_checkout_preserves_historical_provenance() -> None:
    blocks = _checkout_step_blocks(CI_WORKFLOW.read_text(encoding="utf-8"))

    assert len(blocks) == 2
    for block in blocks:
        assert "persist-credentials: false" in block
        assert "fetch-depth: 0" in block
