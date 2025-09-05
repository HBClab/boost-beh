import pandas as pd
import pytest

from pathlib import Path

import code.main_handler as mh


@pytest.mark.parametrize(
    "method_name, dfs, task, expected_filename",
    [
        ("qc_cc_dfs", [], "AF", "meta_file.csv"),
        ("qc_ps_dfs", [], "PC", "meta_file.csv"),
        ("qc_mem_dfs", [], "FN", "meta_file.csv"),
    ],
)
def test_master_acc_saved_to_meta_file(tmp_path, monkeypatch, method_name, dfs, task, expected_filename):
    # Prepare a temporary project structure where code/main_handler.py resides
    project_root = tmp_path / "project_root"
    code_dir = project_root / "code"
    code_dir.mkdir(parents=True)
    # Monkeypatch the module's __file__ so that Path(__file__).parents[1] == project_root
    fake_file = code_dir / "main_handler.py"
    monkeypatch.setattr(mh, "__file__", str(fake_file))

    # Create handler and set a known master_acc DataFrame
    handler = mh.Handler()
    handler.master_acc = pd.DataFrame({"col": [1, 2, 3]})

    # Spy on the to_csv method of this DataFrame to capture call args
    called = {}
    def fake_to_csv(path, index=False, **kwargs):  # catch extra kwargs if any
        called['path'] = Path(path)
        called['index'] = index

    monkeypatch.setattr(handler.master_acc, "to_csv", fake_to_csv)

    # Invoke the QC method; it should trigger saving master_acc to CSV
    method = getattr(handler, method_name)
    result = method(dfs, task)

    # Ensure method returns categories and plots tuple
    assert isinstance(result, tuple) and len(result) == 2

    # Check that to_csv was called with correct path and index flag
    assert 'path' in called and 'index' in called
    expected_path = project_root / expected_filename
    assert called['path'] == expected_path
    assert called['index'] is False
