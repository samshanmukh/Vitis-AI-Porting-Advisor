"""Butterbase database helpers for persisting analysis runs."""
import os
import requests

APP_ID = os.environ.get("BUTTERBASE_APP_ID", "app_op95thf85de7")
API_KEY = os.environ.get("BUTTERBASE_API_KEY", "bb_sk_1712fc41088ebdca35ee087d985d32e8a07c0c69")
BASE_URL = f"https://api.butterbase.ai/v1/{APP_ID}"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def save_run(extraction: dict, compatibility: dict, proposal_a: dict, proposal_b: dict) -> dict | None:
    payload = {
        "model_name": extraction.get("model_name", "unknown"),
        "total_ops": extraction.get("total_ops", 0),
        "unsupported_count": compatibility.get("unsupported_count", 0),
        "proposal_a_compatibility": proposal_a.get("vitis_compatibility", proposal_a.get("compatibility")),
        "proposal_b_compatibility": proposal_b.get("vitis_compatibility"),
        "proposal_a": proposal_a,
        "proposal_b": proposal_b,
        "unsupported_ops": compatibility.get("unsupported_ops", []),
    }
    try:
        r = requests.post(f"{BASE_URL}/analysis_runs", headers=HEADERS, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Butterbase save_run failed: {e}")
        return None


def list_runs(limit: int = 10) -> list:
    try:
        r = requests.get(
            f"{BASE_URL}/analysis_runs",
            headers=HEADERS,
            params={"order": "created_at.desc", "limit": limit},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else data.get("data", [])
    except Exception as e:
        print(f"Butterbase list_runs failed: {e}")
        return []
