import importlib.util
import json
import sys
import sysconfig
import types
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd
import pytest


def _prepare_code_namespace() -> None:
    """
    Make sure the project package ``code`` wins import resolution without
    breaking libraries (pytest) that rely on the stdlib module of the same name.
    """
    stdlib_key = "code._stdlib_bootstrap"

    if stdlib_key not in sys.modules:
        stdlib_path = sysconfig.get_path("stdlib")
        if stdlib_path:
            candidate = Path(stdlib_path) / "code.py"
            if candidate.exists():
                spec = importlib.util.spec_from_file_location(stdlib_key, candidate)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    sys.modules[stdlib_key] = module

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_prepare_code_namespace()

try:
    import code.functions  # noqa: F401,E402
except ModuleNotFoundError:
    # Some developer environments may not ship the optional helpers yet; provide a stub.
    sys.modules.setdefault("code.functions", types.ModuleType("code.functions"))
    import code.functions  # noqa: F401,E402
from code.main_handler import Handler
from code.data_processing.pull_handler import Pull
from code.data_processing.utils import CONVERT_TO_CSV


def _base_record(task: str) -> Dict[str, object]:
    """Build a minimal trial record mimicking the JATOS payload."""
    return {
        "task": task,
        "subject_id": f"{task.lower()}-001",
        "session": 1,
        "condition": "A",
        "response_time": 350,
        "correct": 1,
    }


def _payload_templates() -> List[Callable[[Dict[str, object]], str]]:
    """Different JSON layouts that the converter should understand."""
    return [
        # Standard dict with a list under ``data``.
        lambda rec: json.dumps({"data": [rec, {**rec, "trial_index": 2}]}),
        # Dict with a single ``data`` dict.
        lambda rec: json.dumps({"data": rec}),
        # Flat list of trial dictionaries.
        lambda rec: json.dumps([rec, {**rec, "trial_index": 3}]),
        # Nested data structures inside a list.
        lambda rec: json.dumps(
            [{"data": [rec]}, {"data": {**rec, "trial_index": 4}}]
        ),
        # Newline-delimited JSON objects, wrapped in array markers by exporters.
        lambda rec: "\n".join(
            [
                "[",
                json.dumps({"data": rec}) + ",",
                json.dumps({"data": {**rec, "trial_index": 5}}),
                "]",
            ]
        ),
        # CamelCase wrappers with inconsistent key names.
        lambda rec: json.dumps(
            {
                "TrialData": {
                    "BlockName": "Test ",
                    "Condition": rec["condition"],
                    "Correct": 1,
                    "Session": rec["session"],
                    "Subject": rec["subject_id"],
                    "Extra": {"Correct": 0, "Block_Type": "Practice"},
                },
                "response": {"latency": 350},
            }
        ),
    ]


def _payload_for_task(task: str, index: int) -> str:
    templates = _payload_templates()
    template = templates[index % len(templates)]
    return template(_base_record(task))


def _txt_frames(payload: str):
    """Match the Pull.return_data contract (list of DataFrames w/ file_content)."""
    return [pd.DataFrame({"file_content": [payload]})]


@pytest.mark.parametrize("days_ago", [1, 7])
def test_convert_to_csv_handles_api_payloads(monkeypatch, days_ago):
    # Avoid registering cleanup handlers during the test run.
    monkeypatch.setattr("code.main_handler.atexit.register", lambda func: func)

    handler = Handler()

    # Cycle through JSON shapes so every task exercises the converter differently.
    tasks = sorted(handler.IDs.keys())
    template_index_by_task: Dict[str, int] = {
        task: idx for idx, task in enumerate(tasks)
    }

    def fake_load(self, days_ago: int = 1):
        payload = _payload_for_task(self.taskName, template_index_by_task[self.taskName])
        return _txt_frames(payload)

    monkeypatch.setattr(Pull, "load", fake_load, raising=False)

    errors: List[Tuple[str, Exception]] = []

    for task, task_ids in handler.IDs.items():
        pull = Pull(task_ids, tease="<tease>", token="<token>", taskName=task, proxy=False)

        try:
            txt_dfs = pull.load(days_ago=days_ago)
            converter = CONVERT_TO_CSV(task)
            csv_dfs = converter.convert_to_csv(txt_dfs)

            assert csv_dfs, f"No CSV data returned for task {task}"
            for df in csv_dfs:
                assert isinstance(df, pd.DataFrame)
                assert not df.empty, f"Empty dataframe returned for task {task}"
                assert "subject_id" in df.columns
                assert df["subject_id"].notna().all()

        except Exception as exc:  # pragma: no cover - gather failures for assertion
            errors.append((task, exc))

    assert not errors, "; ".join(f"{task}: {error!r}" for task, error in errors)
