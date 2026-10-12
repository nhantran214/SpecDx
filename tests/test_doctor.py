import subprocess

import httpx

from specdx.doctor import (
    check_database,
    check_ollama,
    check_python,
    check_vendor,
    pull_commands,
    read_vram,
    recommend_profile,
    run_doctor,
)


def _ollama(hosts: dict[str, dict[str, str]], tools: frozenset[str] = frozenset({"m:1"})):
    """hosts: host name -> {model tag: digest}; a host absent from the dict is down."""
    def handle(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host not in hosts:
            raise httpx.ConnectError("refused", request=request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": n, "digest": d} for n, d in hosts[host].items()]})
        model = request.read().decode()
        caps = ["completion", "tools"] if any(t in model for t in tools) else ["completion"]
        return httpx.Response(200, json={"capabilities": caps})
    return httpx.MockTransport(handle)


def _status(checks, name):
    return next(c.status for c in checks if c.name == name)


def test_hosts_that_agree_pass_and_a_missing_model_is_listed_for_pulling() -> None:
    transport = _ollama({"a": {"m:1": "d1", "n:2": "d2"}, "b": {"m:1": "d1"}})
    checks, missing = check_ollama(["http://a", "http://b"], ["m:1", "n:2"], transport)
    assert _status(checks, "digest m:1") == "ok" and _status(checks, "tools m:1 @ http://a") == "ok"
    assert _status(checks, "tools n:2 @ http://a") == "warn"  # no native tool calling
    assert missing == {"http://b": ["n:2"]}
    assert pull_commands(missing, "http://a") == ["OLLAMA_HOST=http://b ollama pull n:2"]
    assert pull_commands({"http://a": ["x"]}, "http://a") == ["ollama pull x"]


def test_hosts_that_disagree_on_a_digest_fail_and_a_down_host_fails() -> None:
    transport = _ollama({"a": {"m:1": "d1"}, "b": {"m:1": "other"}})
    checks, _ = check_ollama(["http://a", "http://b", "http://c"], ["m:1"], transport)
    assert _status(checks, "digest m:1") == "fail"
    assert _status(checks, "ollama http://c") == "fail"


def test_another_projects_port_fails_before_any_connection() -> None:
    check = check_database("postgresql://u:p@localhost:55432/x")
    assert check.status == "fail" and "55434" in check.detail
    assert check_database("postgresql://u:p@localhost:1/x").status == "fail"  # unreachable


def test_the_vendor_check_runs_the_script_or_warns_without_a_source(tmp_path) -> None:
    assert check_vendor(src=tmp_path / "absent").status == "warn"

    def run(argv, **kwargs):
        assert argv[-1].endswith("scripts/verify_vendor.sh")
        assert kwargs["env"]["TONOMYX_SRC"] == str(tmp_path)
        return subprocess.CompletedProcess(argv, 0, "vendor/tonomyx matches 34c1d2c", "")

    assert check_vendor(run, tmp_path).status == "ok"


def test_python_below_three_twelve_fails() -> None:
    assert check_python((3, 11))[0].status == "fail"
    assert check_python((3, 13))[0].status == "ok"


def _smi(mib: str, code: int = 0):
    def run(argv, **kwargs):
        if argv[0] != "nvidia-smi":
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, code, mib, "")
    return run


def test_vram_picks_a_profile_and_a_broken_nvidia_smi_is_reported() -> None:
    assert read_vram(_smi("16303\n")) == (16303 / 1024, "16 GiB")
    assert read_vram(_smi("8192\n24564\n"))[0] == 24564 / 1024  # the largest GPU
    broken = read_vram(_smi("Failed to initialize NVML: Driver/library version mismatch\n", 18))
    assert broken == (None, "nvidia-smi failed: Failed to initialize NVML: Driver/library "
                            "version mismatch")
    assert read_vram(_smi("N/A\n"))[0] is None

    def absent(argv, **kwargs):
        raise FileNotFoundError(argv[0])

    assert read_vram(absent) == (None, "no local nvidia-smi")
    assert [recommend_profile(v) for v in (15.9, 24.0, 47.5, 80.0, None)] == [
        "gpu16", "gpu24", "gpu48", "gpu80", None]


def test_the_doctor_reports_every_check_and_its_verdict(tmp_path) -> None:
    transport = _ollama({"a": {"qwen3:14b": "d"}})
    report, healthy = run_doctor("gpu16", database_url="postgresql://u:p@localhost:55433/x",
                                 hosts=["http://a"], work_dir=tmp_path, transport=transport,
                                 run=_smi("16303\n"))
    assert not healthy  # the database check fails on another project's port
    assert "Recommended profile: gpu16" in report and "| postgresql | fail |" in report
    assert "ollama pull llama3.1:8b" in report and "| disk | " in report
