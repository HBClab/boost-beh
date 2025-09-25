# Holistic QC Pipeline

A modular rewrite of the BOOST behavioral quality-control (QC) pipeline. The repo pulls raw JATOS exports, normalizes them into tidy data frames, applies construct-specific QC, and persists both participant-level artifacts and aggregate dashboards for downstream analysts.

## Highlights
- Single entrypoint (`code/main_handler.py`) that coordinates pulling raw studies, CSV conversion, QC, persistence, and plotting.
- Domain-specific QC modules for the core cognitive constructs: cognitive control (CC), psychomotor speed (PS), memory (MEM), and word learning (WL).
- Automatic artifact management: raw outputs land under `data/`, aggregated summaries in `meta/`, and generated plots in per-subject folders (with exemplar group views retained in `group/plots/`).
- Ready to automate: `python code/main_handler.py all` mirrors the GitHub Action and is safe to schedule.

## Repository Layout
```text
code/
  main_handler.py        # Orchestrates end-to-end QC for a task or the full battery
  data_processing/
    pull_handler.py      # Pulls fresh JATOS exports by study IDs
    utils.py             # Shared helpers (CSV normalization, accuracy/RT math, WL fuzzy matching)
    save_utils.py        # Writes subject artifacts (CSV + plots) into the data lake structure
    cc_qc.py             # CC task QC rules (AF/NF/NTS/ATS/NNB/VNB)
    ps_qc.py             # PS task QC rules (PC/LC/DSST)
    mem_qc.py            # Working memory QC rules (FN/SM)
    wl_qc.py             # Word learning QC rules (WL/DWL + delay reconciliation)
    plot_utils.py        # Matplotlib/seaborn helpers for construct-specific visualizations
  transfer/
    path_logic.py        # Optional helper to mirror generated outputs onto the BOOST file server

data/                   # Subject-level caches (obs/int sites, then subject/task/data|plot)
meta/                   # Auto-saved aggregate CSVs (master_acc, cc_master, ps_master, mem_master, wl_master)
group/plots/            # Example construct plots for quick reference
requirements.txt        # Python dependencies for QC + plotting
run.py                  # Flask placeholder (not yet active)
```

## Data & QC Flow
1. **Pull** – `Pull` in `pull_handler.py` requests study metadata + data blobs from JATOS for the study IDs defined in `Handler.IDs`. `days_ago` defaults to 127 but can be overridden when calling `load()`.
2. **Normalize** – `CONVERT_TO_CSV` flattens newline-delimited JSON into tidy Pandas frames ready for QC.
3. **QC & Metrics** – `Handler.choose_construct()` routes each task to its construct-specific QC class:
   - `CCqC` enforces max RT checks, per-condition accuracy thresholds, and task-switching rules.
   - `PS_QC` scores psychomotor speed blocks and tallies correct counts.
   - `MEM_QC` inspects FN/SM performance with RT + accuracy rollups.
   - `WL_QC` orchestrates fuzzy matching against version-specific keys, handling WL and DWL simultaneously.
4. **Visualize** – `plot_utils` generates construct-appropriate figures (per-condition counts, RT distributions, WL learning curves, etc.).
5. **Persist** – `SAVE_EVERYTHING` stores per-participant CSVs and plots under `data/<study>/<site>/<subject>/<task>/`. `Handler._persist_all_masters()` writes aggregate CSVs into `meta/` on every successful task run to keep analytics in sync.

## Supported Tasks
| Construct | Tasks | Notes |
|-----------|-------|-------|
| CC (Cognitive Control) | `AF`, `NF`, `ATS`, `NTS`, `NNB`, `VNB` | Shared QC thresholds at 50% accuracy, optional task-switching logic for ATS/NTS |
| PS (Psychomotor Speed) | `PC`, `LC`, `DSST` | Separate RT limits for LC/PC vs DSST; exports accuracy and correct-count masters |
| MEM (Face/Scene Memory) | `FN`, `SM` | Captures per-condition accuracy, mean RT, and counts into `mem_master.csv` |
| WL (Word Learning + Delayed) | `WL`, `DWL` | Combines learning/distraction/immediate blocks with delayed recall; masters upsert rows per subject/session |

To target a single task, run `python code/main_handler.py WL`. To mirror the nightly sweep, use `python code/main_handler.py all`.

## Setup
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. (Optional) If you are on Nix, `nix develop` provisions the toolchain.
3. Configure secrets:
   - `Handler.pull()` currently references a token inline. Replace with an environment variable (e.g., `JATOS_TOKEN`) and export it before running.
   - Proxy credentials (`tease`) should also come from the environment or an `.env` file that is not committed.

## Running QC Locally
```bash
# QC the full battery (mirrors CI)
python code/main_handler.py all

# QC a single construct
python code/main_handler.py AF
```

Outputs land under `data/` using the subject -> task folder pattern enforced by `SAVE_EVERYTHING`. Every run also refreshes the aggregated CSVs in `meta/`:
- `master_acc.csv`: high-level accuracy summaries for PS/MEM tasks.
- `cc_master.csv`: condition-level accuracy + mean RT for CC tasks.
- `ps_master.csv`: per-block correct counts for PS tasks.
- `mem_master.csv`: joined counts/RT/accuracy for FN/SM.
- `wl_master_wide.csv` & `wl_master.csv`: wide vs flattened WL summaries combining WL + DWL submissions.

## Visual Artifacts
- Participant plots are co-located with their data under `data/.../plot/`.
- Shared reference visuals live in `group/plots/` (e.g., `flanker.png`, `task_switching.png`) for quick distribution in slide decks.

## Transferring Data to the Server
`code/transfer/path_logic.py` discovers local subject folders and mirrors them to `/mnt/lss/Projects/BOOST` (observational vs intervention sites routed automatically). Use `PathLogic.copy_subjects_to_server(max_workers=?, dry_run=True)` inside a Python shell to preview the copy plan before executing.

## Development Workflow
- Lint with `python -m flake8 code` before committing.
- Run `pytest` (tests live under `tests/`) to cover threshold logic, expected artifact names, and any new utilities.
- Keep notebooks or ad-hoc experiments outside tracked directories, or convert them into reproducible scripts.

## Extending the Pipeline
1. Add the new task code and study IDs to `Handler.IDs`.
2. Implement construct logic under `code/data_processing/` (reuse helpers in `utils.py` when possible).
3. Register the new branch in `Handler.choose_construct()` and add persistence hooks for master CSVs.
4. Document the task behavior and update tests/fixtures to reflect the new data expectations.

## Troubleshooting
- **No data returned from JATOS**: confirm the study IDs in `Handler.IDs` and that your token has access; adjust the `days_ago` window if you are backfilling.
- **Missing session folders**: ensure input CSVs include `session` or `session_number`. `SAVE_EVERYTHING` uses those columns to label artifacts.
- **WL metrics look stale**: WL and DWL write to the same `wl_master` rows via `_upsert_wl_master`; make sure both tasks are run for each session to populate delay scores.

## License & Data Privacy
This repository processes sensitive participant responses. Keep tokens, raw exports, and downstream artifacts off public machines. Add new temp/output folders to `.gitignore` as needed to avoid leaking data.
