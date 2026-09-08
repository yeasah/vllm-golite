"""Reading invocations out of the scripts they currently live in."""

from vllm_untwisted.store.shell import parse

SCRIPT = """#!/bin/sh

unset VLLM_DISABLE_COMPILE_CACHE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BARE_ONE=nope

############ verified on 9/7/2026

# 3.00bpw w/turboquant, long context - 48t/s @ 187K
vllm serve /ckpt/qwen --max-num-seqs 1 --kv-cache-dtype turboquant_4bit_nc

# 3.00bpw w/fp8
#EXL3_RECONSTRUCT_THRESHOLD=0 vllm serve /ckpt/qwen --kv-cache-dtype fp8

############ prior testing, unverified

# kv offload test
#vllm serve /ckpt/qwen --kv-transfer-config '{"kv_connector":"X","kv_role":"kv_both"}'
"""


def write(tmp_path, text=SCRIPT, name="run-thing.sh"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_names_come_from_the_comments_they_were_trapped_in(tmp_path):
    got = [i.config.name for i in parse(write(tmp_path)).configs]
    assert got == ["3-00bpw-w-turboquant-long-context", "3-00bpw-w-fp8", "kv-offload-test"]


def test_commented_invocations_are_imported_as_inactive(tmp_path):
    r = parse(write(tmp_path))
    assert [i.active for i in r.configs] == [True, False, False]


def test_the_banner_becomes_provenance(tmp_path):
    r = parse(write(tmp_path))
    assert "verified on 9/7/2026" in r.configs[0].note
    assert "prior testing, unverified" in r.configs[2].note


def test_export_applies_to_everything_below_it(tmp_path):
    r = parse(write(tmp_path))
    assert all(i.config.env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
               for i in r.configs)


def test_a_bare_assignment_is_reported_and_not_applied(tmp_path):
    """sh sets it without exporting, so it never reaches the engine. Importing it would
    quietly change what the configuration does."""
    r = parse(write(tmp_path))
    assert all("BARE_ONE" not in i.config.env for i in r.configs)
    assert any("BARE_ONE" in w and "without `export`" in w for w in r.warnings)


def test_a_command_prefixed_with_assignments_is_not_a_bare_assignment(tmp_path):
    """The regression this cost: `A=1 B=2 vllm serve ...` starts with the same text as a
    bare assignment, so a regex over the whole line swallowed the entire invocation and
    the file imported nothing."""
    r = parse(write(tmp_path))
    fp8 = r.configs[1]
    assert fp8.config.env["EXL3_RECONSTRUCT_THRESHOLD"] == "0"
    assert fp8.config.model == "/ckpt/qwen"
    assert fp8.config.args == ["--kv-cache-dtype", "fp8"]


def test_a_command_prefix_overrides_a_file_export(tmp_path):
    script = SCRIPT + "\n# override\n#PYTORCH_CUDA_ALLOC_CONF=other vllm serve /ckpt/q\n"
    r = parse(write(tmp_path, script))
    assert r.configs[-1].config.env["PYTORCH_CUDA_ALLOC_CONF"] == "other"


def test_json_arguments_survive(tmp_path):
    r = parse(write(tmp_path))
    assert r.configs[2].config.args[-1] == '{"kv_connector":"X","kv_role":"kv_both"}'


def test_unset_is_reported_because_the_store_has_no_equivalent(tmp_path):
    assert any("clears an ambient variable" in w for w in parse(write(tmp_path)).warnings)


def test_a_pinned_port_is_dropped(tmp_path):
    script = "# named\nvllm serve /ckpt/q --port 8000 --max-num-seqs 1\n"
    r = parse(write(tmp_path, script))
    assert "--port" not in r.configs[0].config.args
    assert any("dropped --port" in w for w in r.warnings)


def test_a_tilde_is_expanded_and_said_so(tmp_path):
    script = "# named\nVLLM_LOGGING_CONFIG_PATH=~/x.json vllm serve /ckpt/q\n"
    r = parse(write(tmp_path, script))
    assert not r.configs[0].config.env["VLLM_LOGGING_CONFIG_PATH"].startswith("~")
    assert any("expanded" in w for w in r.warnings)


def test_names_stay_unique_within_a_file(tmp_path):
    script = "# same\nvllm serve /ckpt/a\n# same\nvllm serve /ckpt/b\n"
    names = [i.config.name for i in parse(write(tmp_path, script)).configs]
    assert len(set(names)) == 2


def test_an_unnamed_invocation_falls_back_to_the_file(tmp_path):
    r = parse(write(tmp_path, "vllm serve /ckpt/q\n", name="run-solo.sh"))
    assert r.configs[0].config.name.startswith("run-solo")


def test_a_local_checkpoint_beside_the_script_is_made_absolute(tmp_path):
    """The scripts name checkpoints relatively because sh runs them from that directory.
    A stored configuration has no working directory, and vLLM's error for a relative path
    that resolves to nothing is indistinguishable from a checkpoint never downloaded."""
    (tmp_path / "Qwen-exl3-3bpw").mkdir()
    r = parse(write(tmp_path, "# named\nvllm serve Qwen-exl3-3bpw --max-num-seqs 1\n"))
    assert r.configs[0].config.model == str(tmp_path / "Qwen-exl3-3bpw")
    assert any("relative to the script" in w for w in r.warnings)


def test_a_hub_identifier_is_left_alone(tmp_path):
    r = parse(write(tmp_path, "# named\nvllm serve turboderp/gemma-4-12B-it-exl3\n"))
    assert r.configs[0].config.model == "turboderp/gemma-4-12B-it-exl3"
    assert not r.warnings


def test_an_absolute_path_stays_absolute(tmp_path):
    r = parse(write(tmp_path, "# named\nvllm serve /ckpt/qwen\n"))
    assert r.configs[0].config.model == "/ckpt/qwen"
