#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="$repo_root/experiment/configs/followup_cpu_rapid_20m_6s.yaml"
run_dir="$repo_root/experiment/runs/followup_cpu_rapid_20m_6s"
meta_path="$repo_root/experiment/data/tinystories_smoke/meta.json"

target_tokens=20000000

if [[ -x "$repo_root/.venv/bin/python" ]]; then
  python_cmd="$repo_root/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  python_cmd="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  python_cmd="$(command -v python)"
else
  echo "No Python interpreter found. Install Python or create .venv first." >&2
  exit 1
fi

if ! "$python_cmd" -c "import tinystories_xpu" >/dev/null 2>&1; then
  export PYTHONPATH="$repo_root/experiment/src${PYTHONPATH:+:$PYTHONPATH}"
fi

need_install=0
if ! "$python_cmd" -c "import tinystories_xpu" >/dev/null 2>&1; then
  need_install=1
fi
if ! "$python_cmd" -c "import tokenizers, datasets, yaml" >/dev/null 2>&1; then
  need_install=1
fi

if [[ "$need_install" -eq 1 ]]; then
  if [[ -f "$repo_root/experiment/pyproject.toml" ]]; then
    echo "Installing experiment package and dependencies..."
    "$python_cmd" -m pip install -e "$repo_root/experiment"
  fi
fi

if ! "$python_cmd" -c "import torch" >/dev/null 2>&1; then
  echo "PyTorch not found. Installing CPU PyTorch wheel..."
  "$python_cmd" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

if ! "$python_cmd" -c "import tinystories_xpu" >/dev/null 2>&1; then
  echo "Unable to import tinystories_xpu after bootstrap. Ensure dependencies are installed." >&2
  exit 1
fi
if ! "$python_cmd" -c "import tokenizers, datasets, yaml, torch" >/dev/null 2>&1; then
  echo "Missing runtime dependencies after bootstrap (expected: tokenizers, datasets, pyyaml, torch)." >&2
  exit 1
fi

if [[ ! -f "$meta_path" ]]; then
  echo "Prepared data cache missing. Running prepare stage first..."
  "$python_cmd" -m tinystories_xpu.prepare_data --config "$config"
fi

mapfile -t resume_seeds < <("$python_cmd" - "$run_dir" "$target_tokens" <<'PY'
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
target_tokens = int(sys.argv[2])
seeds = [101, 202, 303, 404, 505, 606]

for seed in seeds:
    seed_dir = run_dir / f"seed_{seed}"
    status_path = seed_dir / "status.json"

    status_stage = ""
    status_tokens = 0
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status_stage = str(status.get("stage", ""))
            status_tokens = int(status.get("tokens_seen", 0) or 0)
        except Exception:
            status_stage = ""
            status_tokens = 0

    checkpoint_tokens = 0
    checkpoint_dir = seed_dir / "checkpoints"
    if checkpoint_dir.exists():
        for path in checkpoint_dir.glob(f"seed_{seed}_tokens_*.pt"):
            match = re.search(r"_tokens_(\d+)\.pt$", path.name)
            if match:
                checkpoint_tokens = max(checkpoint_tokens, int(match.group(1)))

    best_tokens = max(status_tokens, checkpoint_tokens)
    done = status_stage == "done" and best_tokens >= target_tokens
    print(f"seed={seed} stage={status_stage} tokens={best_tokens} done={done}", file=sys.stderr)

    if not done:
        print(seed)
PY
)

if [[ ${#resume_seeds[@]} -eq 0 ]]; then
  echo "All seeds already reached target tokens. Rebuilding telemetry + analysis only..."
else
  echo "Resuming incomplete seeds: ${resume_seeds[*]}"
  "$python_cmd" -m tinystories_xpu.train --config "$config" --seeds "${resume_seeds[@]}"
fi

"$python_cmd" -m tinystories_xpu.telemetry --run-dir "$run_dir"
"$python_cmd" -m tinystories_xpu.analyze --run-dir "$run_dir"

echo "Done. See: $run_dir"
