# TinyStories XPU Experiment

This experiment keeps the dissertation design intact while moving the implementation to
native Windows 11 PyTorch with Intel GPU acceleration through `torch.xpu`.

The scaffold provides:

- TinyStories download, tokenizer training, and fixed token-cache preparation
- a configurable Llama-style decoder-only language model with RoPE, RMSNorm, and SwiGLU
- repeated-seed training runs with fixed token checkpoints
- CSV telemetry for training loss, validation loss, perplexity, throughput, and epoch equivalents
- derived stability metrics for volatility, spike rate, backslides, and between-seed variance
- repeated-measures ANOVA, pairwise comparisons, and a mixed-effects robustness check

## Hardware and Software Target

The Windows setup targets Intel Arc through native PyTorch XPU wheels. PyTorch documents
Windows 11 and Intel Arc A-Series GPUs as a supported Intel client GPU path, installed with
the XPU wheel index:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

Intel Extension for PyTorch is not used by default because Intel has announced that IPEX
is being retired and that Intel GPU support has moved upstream into PyTorch.

## Install on Windows 11

Run PowerShell from the repository root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows_xpu.ps1
```

If the stable PyTorch XPU wheel has a regression on your machine, try the nightly path:

```powershell
.\scripts\setup_windows_xpu.ps1 -NightlyTorch
```

The setup script creates `.venv`, installs PyTorch XPU, installs this package in editable
mode, and runs an XPU matrix-multiply check.

If XPU verification fails (for example due to enterprise application-control policy), the
setup script now automatically reinstalls CPU-only PyTorch wheels so the rest of the
pipeline can still run.

## Smoke Test

The smoke config uses a tiny model and a tiny token budget. It is designed to verify the
whole pipeline before a long run:

```powershell
.\scripts\run_smoke_windows.ps1
```

That script runs:

```powershell
tinystories-prepare --config experiment\configs\smoke.yaml
tinystories-train --config experiment\configs\smoke.yaml
tinystories-telemetry --run-dir experiment\runs\smoke_xpu
tinystories-analyze --run-dir experiment\runs\smoke_xpu
```

## Follow-Up Study Pilot (CPU)

For follow-up runs on CPU fallback, use `experiment\configs\followup_cpu_pilot.yaml`.

This pilot profile keeps the full model shape but reduces the run budget for practical
iteration on CPU:

- `seeds: [101, 202, 303]`
- `target_tokens: 100,000,000`
- `checkpoint_tokens: 10,000,000`

Run the pilot experiment:

```powershell
.\.venv\Scripts\Activate.ps1
tinystories-prepare --config experiment\configs\followup_cpu_pilot.yaml
tinystories-train --config experiment\configs\followup_cpu_pilot.yaml
tinystories-telemetry --run-dir experiment\runs\followup_cpu_pilot_100m
tinystories-analyze --run-dir experiment\runs\followup_cpu_pilot_100m
```

## Rapid CPU Follow-Up (20M Tokens)

For much faster CPU iteration, use `experiment\configs\followup_cpu_rapid.yaml`.

This profile is intentionally small and practical for day-to-day iteration:

- `seeds: [101, 202]`
- `target_tokens: 20,000,000`
- model shape: `n_layer: 4`, `n_head: 4`, `n_embd: 256`

Run with the convenience script:

```powershell
.\scripts\run_followup_cpu_rapid_windows.ps1
```

Or run stages manually:

```powershell
.\.venv\Scripts\Activate.ps1
tinystories-prepare --config experiment\configs\followup_cpu_rapid.yaml
tinystories-train --config experiment\configs\followup_cpu_rapid.yaml
tinystories-telemetry --run-dir experiment\runs\followup_cpu_rapid_20m
tinystories-analyze --run-dir experiment\runs\followup_cpu_rapid_20m
```

## Full-Scale 2.5B Profile (Reference)

The full config is `experiment\configs\full_2_5b_xpu.yaml`.

It uses a roughly 125M-parameter Llama-style profile:

- `n_layer: 14`
- `n_head: 12`
- `n_embd: 768`
- `block_size: 512`
- `vocab_size: 32000`

The configured token target is `2,500,000,000`.

Training is checkpoint-resumable. If power is lost or the process is interrupted,
rerun the same train command and each seed will resume from its latest checkpoint.

Run the full experiment:

```powershell
.\.venv\Scripts\Activate.ps1
tinystories-prepare --config experiment\configs\full_2_5b_xpu.yaml
tinystories-train --config experiment\configs\full_2_5b_xpu.yaml
tinystories-telemetry --run-dir experiment\runs\tinystories_2_5b_xpu
tinystories-analyze --run-dir experiment\runs\tinystories_2_5b_xpu
```

If the full model runs out of memory, first reduce `training.micro_batch_size` to `1`
if it is not already there, then increase `training.gradient_accumulation_steps` to keep
the effective token count per optimizer step comparable.

## Outputs

Each seed writes:

```text
experiment\runs\<run_name>\seed_<seed>\metrics.csv
experiment\runs\<run_name>\seed_<seed>\run_meta.json
experiment\runs\<run_name>\seed_<seed>\checkpoints\
```

The combined run writes:

```text
experiment\runs\<run_name>\metrics_all.csv
experiment\runs\<run_name>\telemetry_derived.csv
experiment\runs\<run_name>\interval_summary.csv
experiment\runs\<run_name>\analysis_report.md
experiment\runs\<run_name>\plots\
```

`metrics_all.csv` is the primary repeated-measures dataset. `seed` is the subject,
`interval_index` is the within-subjects factor, and `tokens_seen` is the cumulative
training-token checkpoint.

## Main Dependent Variables

The raw logged variables are:

- `train_loss`
- `val_loss`
- `val_ppl`
- `tokens_per_second`

The derived stability variables are:

- `val_loss_delta`
- `rolling_volatility`
- `spike`
- `spike_rate`
- `backslide`
- `backslide_frequency`
- `between_seed_val_loss_variance`
- `stable_phase_candidate`

These map directly onto the paper's repeated-measures questions about training stability,
between-seed variability, non-monotonic behavior, and stable phase detection.
