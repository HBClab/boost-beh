#!/usr/bin/env python3
"""Data session/task completeness checker for BOOST behavior data.

This script scans the ``data/`` tree (both ``int`` and ``obs`` sites),
counts how many subjects have a full set of task exports per session,
and surfaces any missing task/session combinations.

Outputs (written to ``meta/`` by default):
- ``dsmc_complete_counts.csv``: complete-subject counts per session and dataset.
- ``dsmc_missing_tasks.csv``: rows for every missing task per subject/session.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

# Default task roster (derived from README.md: CC + PS + MEM + WL constructs)
DEFAULT_TASKS: Set[str] = {
    "AF",
    "NF",
    "NTS",
    "ATS",
    "NNB",
    "VNB",
    "FN",
    "SM",
    "PC",
    "LC",
    "DSST",
    "WL",
    "DWL",
}

SESSION_PATTERN = re.compile(r"_ses-(?P<session>[^_]+)")


def normalize_session_label(session: str) -> str:
    """Strip common prefixes so labels are comparable (e.g., ses-1 -> 1)."""
    session = session.strip()
    return session[4:] if session.startswith("ses-") else session


def load_offloaded_pairs(path: Path) -> Set[Tuple[str, str, str]]:
    """
    Read offloaded task records.

    Returns a set of (task, subject, session) tuples using normalized session labels.
    """
    pairs: Set[Tuple[str, str, str]] = set()
    if not path.exists():
        return pairs

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row.get("task", "").strip().upper()
            subject = row.get("sub", "").strip()
            session = normalize_session_label(row.get("session", ""))
            if task and subject and session:
                pairs.add((task, subject, session))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count complete subjects per session and list missing tasks."
    )
    parser.add_argument(
        "--tasks",
        help="Comma-separated list of expected tasks. Defaults to the full BOOST battery.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Path to the data directory containing int/ and obs/ (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "meta",
        help="Directory to write output CSVs (default: %(default)s).",
    )
    return parser.parse_args()


def parse_session(filename: str) -> str | None:
    """Extract the session token from a CSV filename."""
    match = SESSION_PATTERN.search(filename)
    return match.group("session") if match else None


def session_sort_key(session: str) -> Tuple[int, float | str]:
    """Sort sessions numerically when possible."""
    try:
        return (0, float(session))
    except ValueError:
        return (1, session)


def format_session(session: str) -> str:
    """Standardize session labels in outputs."""
    return f"ses-{session}"


def discover_tasks(data_root: Path) -> Set[str]:
    """Infer task names from existing data files."""
    tasks: Set[str] = set()
    for csv_path in data_root.glob("*/*/*/*/data/*.csv"):
        tasks.add(csv_path.parent.parent.name)
    return tasks


def iter_subjects(dataset_root: Path) -> Iterable[Tuple[str, Path]]:
    """Yield (site, subject_path) pairs for a dataset root."""
    for site_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        for subject_dir in sorted(p for p in site_dir.iterdir() if p.is_dir()):
            yield site_dir.name, subject_dir


def collect_subject_sessions(subject_dir: Path) -> Dict[str, Set[str]]:
    """Map session -> set of tasks present for a subject."""
    sessions: Dict[str, Set[str]] = defaultdict(set)
    for csv_path in subject_dir.glob("*/data/*.csv"):
        session = parse_session(csv_path.name)
        if session is None or session.lower() == "nan" or session == "":
            continue
        task = csv_path.parent.parent.name
        sessions[session].add(task)
    return sessions


def gather_dataset_stats(
    dataset_root: Path,
    dataset_label: str,
    expected_tasks: Set[str],
    offloaded_pairs: Set[Tuple[str, str, str]],
) -> Tuple[Dict[str, int], List[Tuple[str, str, str, str]], Set[str]]:
    """
    Compute completeness counts and missing task rows for one dataset.

    Returns:
        complete_counts: session -> number of subjects with all expected tasks.
        missing_rows: list of (site, subject, session, missing_task).
        sessions_seen: set of all sessions observed in this dataset.
    """
    complete_counts: Dict[str, int] = defaultdict(int)
    missing_rows: List[Tuple[str, str, str, str]] = []
    sessions_seen: Set[str] = set()

    for site, subject_dir in iter_subjects(dataset_root):
        session_tasks = collect_subject_sessions(subject_dir)
        for session, tasks_present in session_tasks.items():
            sessions_seen.add(session)
            missing = expected_tasks - tasks_present

            # Exempt VNB/NNB sessions that were run offline (recorded in offloaded.csv).
            missing = {
                task
                for task in missing
                if not (
                    task.upper() in {"VNB", "NNB"}
                    and (
                        task.upper(),
                        subject_dir.name,
                        normalize_session_label(session),
                    )
                    in offloaded_pairs
                )
            }

            if not missing:
                complete_counts[session] += 1
            else:
                for task in sorted(missing):
                    missing_rows.append((site, subject_dir.name, session, task))

    return complete_counts, missing_rows, sessions_seen


def write_complete_counts(
    path: Path, rows: Sequence[Tuple[str, str, int]]
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "session", "complete_subjects"])
        for dataset, session, count in rows:
            writer.writerow([dataset, format_session(session), count])


def write_missing_rows(
    path: Path, rows: Sequence[Tuple[str, str, str, str, str]]
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "site", "subject", "session", "missing_task"])
        for dataset, site, subject, session, task in rows:
            writer.writerow([dataset, site, subject, format_session(session), task])


def main() -> None:
    args = parse_args()

    expected_tasks = (
        {t.strip() for t in args.tasks.split(",") if t.strip()}
        if args.tasks
        else DEFAULT_TASKS
    )

    data_root: Path = args.data_root
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    offloaded_pairs = load_offloaded_pairs(
        Path(__file__).resolve().parent / "offloaded.csv"
    )

    # If no explicit tasks were provided, include any additional tasks found on disk.
    if not args.tasks:
        expected_tasks = discover_tasks(data_root) or DEFAULT_TASKS

    complete_rows: List[Tuple[str, str, int]] = []
    missing_rows: List[Tuple[str, str, str, str, str]] = []

    for dataset_label in ("int", "obs"):
        dataset_path = data_root / dataset_label
        if not dataset_path.exists():
            continue

        counts, missing, sessions_seen = gather_dataset_stats(
            dataset_path, dataset_label, expected_tasks, offloaded_pairs
        )

        for session in sorted(sessions_seen, key=session_sort_key):
            complete_rows.append(
                (dataset_label, session, counts.get(session, 0))
            )

        for site, subject, session, task in missing:
            missing_rows.append((dataset_label, site, subject, session, task))

    write_complete_counts(output_dir / "dsmc_complete_counts.csv", complete_rows)
    write_missing_rows(output_dir / "dsmc_missing_tasks.csv", missing_rows)


if __name__ == "__main__":
    main()
