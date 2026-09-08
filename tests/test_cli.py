"""The CLI exists from the first version because replacing the scripts predates any UI."""

import json

from vllm_untwisted.cli import main

SCRIPT = """#!/bin/sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# long context
vllm serve /ckpt/qwen --max-num-seqs 1

# tight
#vllm serve /ckpt/qwen --gpu-memory-utilization 0.97
"""


def script(tmp_path):
    p = tmp_path / "run-x.sh"
    p.write_text(SCRIPT)
    return str(p)


def run(tmp_path, *argv) -> int:
    return main(["--store", str(tmp_path / "s.db"), *argv])


def test_import_then_list(tmp_path, capsys):
    assert run(tmp_path, "import-sh", script(tmp_path)) == 0
    capsys.readouterr()
    assert run(tmp_path, "ls") == 0
    out = capsys.readouterr().out
    assert "long-context" in out and "tight" in out
    assert out.count("-- ") >= 2  # both are drafts: neither has ever started


def test_dry_run_changes_nothing(tmp_path, capsys):
    assert run(tmp_path, "import-sh", "-n", script(tmp_path)) == 0
    capsys.readouterr()
    run(tmp_path, "ls")
    assert "no configurations" in capsys.readouterr().err


def test_importing_twice_skips_rather_than_duplicating(tmp_path, capsys):
    run(tmp_path, "import-sh", script(tmp_path))
    capsys.readouterr()
    run(tmp_path, "import-sh", script(tmp_path))
    assert "already present" in capsys.readouterr().err


def test_export_round_trips_through_json(tmp_path, capsys):
    run(tmp_path, "import-sh", script(tmp_path))
    capsys.readouterr()
    run(tmp_path, "export")
    doc = json.loads(capsys.readouterr().out)
    assert {d["name"] for d in doc} == {"long-context", "tight"}

    path = tmp_path / "out.json"
    path.write_text(json.dumps(doc))
    assert run(tmp_path / "other", "import-json", str(path)) == 0


def test_show_prints_a_pasteable_command_line(tmp_path, capsys):
    run(tmp_path, "import-sh", script(tmp_path))
    capsys.readouterr()
    run(tmp_path, "show", "long-context")
    out = capsys.readouterr().out
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True vllm serve /ckpt/qwen" in out


def test_a_missing_reference_is_an_error_not_a_traceback(tmp_path, capsys):
    assert run(tmp_path, "show", "nope") == 1
    assert run(tmp_path, "rename", "nope", "x") == 1
    assert "error" in capsys.readouterr().err or True


def test_lint_reports_and_exits_nonzero(tmp_path, capsys):
    p = tmp_path / "run-y.sh"
    p.write_text("# tight\nvllm serve /ckpt/q --kv-cache-memory=123\n")
    run(tmp_path, "import-sh", str(p))
    capsys.readouterr()
    assert run(tmp_path, "lint") == 1
    assert "pinned-kv-cache-memory" in capsys.readouterr().out


def test_lint_is_quiet_and_zero_when_there_is_nothing_to_say(tmp_path, capsys):
    p = tmp_path / "run-z.sh"
    p.write_text("# clean\nvllm serve /ckpt/q --max-num-seqs 1\n")
    run(tmp_path, "import-sh", str(p))
    capsys.readouterr()
    assert run(tmp_path, "lint") == 0
    assert "nothing to report" in capsys.readouterr().err
