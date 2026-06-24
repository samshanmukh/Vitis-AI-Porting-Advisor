# repo_fetch.py — resolve a model repo from a GitHub URL (or local path) to a local dir.
import os
import re
import subprocess
import tempfile

_CACHE: dict[str, str] = {}   # url -> cloned path (per process)


def resolve_repo(source: str) -> dict:
    """
    Accepts a GitHub URL (https://github.com/owner/repo[.git]) or a local directory path.
    Returns {"path": <local dir>, "origin": "github"|"local", "name": ...} or {"error": ...}.
    """
    source = (source or "").strip()
    if not source:
        return {"error": "empty source"}

    # local path → use as-is
    if not source.lower().startswith(("http://", "https://", "git@")):
        if os.path.isdir(source):
            return {"path": source, "origin": "local", "name": os.path.basename(source.rstrip("/"))}
        return {"error": f"not a directory: {source}"}

    # github url → shallow clone
    if "github.com" not in source:
        return {"error": "only github.com URLs are supported"}
    if source in _CACHE and os.path.isdir(_CACHE[source]):
        return {"path": _CACHE[source], "origin": "github", "name": _repo_name(source)}

    dest = tempfile.mkdtemp(prefix="dpu_scan_")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", source, dest],
            check=True, capture_output=True, text=True, timeout=120,
        )
    except subprocess.CalledProcessError as e:
        return {"error": f"git clone failed: {e.stderr.strip()[:300]}"}
    except subprocess.TimeoutExpired:
        return {"error": "git clone timed out (120s)"}
    except FileNotFoundError:
        return {"error": "git is not installed"}

    _CACHE[source] = dest
    return {"path": dest, "origin": "github", "name": _repo_name(source)}


def _repo_name(url: str) -> str:
    m = re.search(r"github\.com[:/]+[^/]+/([^/.\s]+)", url)
    return m.group(1) if m else "repo"
