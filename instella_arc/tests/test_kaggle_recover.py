from __future__ import annotations

from instella_arc.kaggle_recover import (
    classify_kernel_status,
    resolve_owned_ref,
    sanitize_owner,
)


def test_resolve_owned_ref_returns_only_exact_target_slug() -> None:
    listing = """ref,title
OwnerOne/instella-arc-frozen-probe,Instella ARC Frozen Probe
OwnerOne/other-kernel,Other
ownerone/INSTELLA-ARC-FROZEN-PROBE,Duplicate case rendering
"""
    assert resolve_owned_ref(listing) == [
        "OwnerOne/instella-arc-frozen-probe",
        "ownerone/INSTELLA-ARC-FROZEN-PROBE",
    ]


def test_classify_kernel_status_is_ordered_and_exit_aware() -> None:
    assert classify_kernel_status(1, 'status "complete"') == "query_failure"
    assert classify_kernel_status(0, 'KernelWorkerStatus.COMPLETE') == "complete"
    assert classify_kernel_status(0, 'KernelWorkerStatus.RUNNING') == "running"
    assert classify_kernel_status(0, 'KernelWorkerStatus.ERROR') == "failure"
    assert classify_kernel_status(0, 'unrecognized') == "unknown"


def test_sanitize_owner_removes_ref_and_standalone_owner_case_insensitively() -> None:
    raw = "AmeerUser/instella-arc-frozen-probe belongs to ameeruser; notAmeerUserX stays"
    sanitized = sanitize_owner(raw, "AmeerUser/instella-arc-frozen-probe")
    assert "AmeerUser/instella-arc-frozen-probe" not in sanitized
    assert "ameeruser" not in sanitized.lower().replace("notameeruserx", "")
    assert "<redacted>/instella-arc-frozen-probe" in sanitized
    assert "notAmeerUserX" in sanitized
