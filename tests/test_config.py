"""The storage shape has to survive the invocations it is replacing."""

import shlex

from golite.engine import EngineConfig

# Straight out of ~/ckpt/run-qwen3.8-27b.sh: a JSON value with embedded quotes, which is
# the case that breaks any scheme storing the invocation as a string and re-splitting it.
KV_TRANSFER = '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both"}'


def cfg(**kw) -> EngineConfig:
    return EngineConfig(name="t", model="Qwen3.8-27B-exl3-3.00bpw-bq", **kw)


def test_json_values_survive_a_round_trip_through_a_shell():
    c = cfg(args=["--kv-transfer-config", KV_TRANSFER])
    assert shlex.split(c.command_line())[-1] == KV_TRANSFER


def test_multi_valued_flags_are_preserved():
    c = cfg(args=["--cudagraph-capture-sizes", "1", "2", "4"])
    assert c.argv(8000)[-3:] == ["1", "2", "4"]


def test_with_args_replaces_the_space_syntax():
    c = cfg(args=["--gpu-memory-utilization", "0.88", "--enable-prefix-caching"])
    out = c.with_args(gpu_memory_utilization="0.97")
    assert "--enable-prefix-caching" in out.args
    assert out.args.count("--gpu-memory-utilization") == 1
    assert "0.88" not in out.args and "0.97" in out.args


def test_with_args_replaces_the_equals_syntax():
    # Both syntaxes appear in the same script, which is why editing rendered text is
    # not a safe way to change a field.
    c = cfg(args=["--kv-cache-memory=1323302912", "--max-num-seqs", "1"])
    out = c.with_args(kv_cache_memory="999")
    assert not any(a.startswith("--kv-cache-memory=") for a in out.args)
    assert out.args[-2:] == ["--kv-cache-memory", "999"]


def test_with_args_can_remove_a_flag():
    c = cfg(args=["--max-model-len", "auto", "--max-num-seqs", "2"])
    assert "--max-model-len" not in c.with_args(max_model_len=None).args


def test_env_is_part_of_the_configuration():
    # PYTORCH_CUDA_ALLOC_CONF has to be set before the torch import, so it belongs to
    # the config rather than to whatever the manager happened to inherit.
    c = cfg(env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    assert c.command_line().startswith("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")


def test_port_is_the_supervisors_to_assign():
    assert "--port" not in cfg(args=["--max-num-seqs", "1"]).command_line()
