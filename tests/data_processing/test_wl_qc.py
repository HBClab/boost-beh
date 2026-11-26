"""Tests for WL/DWL QC categorisation, plot threshold, and meta NA handling."""
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

CODE_DIR = Path(__file__).resolve().parents[2] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_processing.meta import META_RECREATE, WL_SCORE_COLUMNS  # noqa: E402
from data_processing.plot_utils import MEM_PLOTS  # noqa: E402
from data_processing.utils import WL_MATCH_THRESHOLD  # noqa: E402
from data_processing.wl_qc import WL_QC  # noqa: E402

LIST_A1 = ["book", "flower", "train", "rug", "meadow"]
LIST_A2 = ["bowl", "dawn", "judge", "grant", "insect"]


def _type_words(words, condition, block_c, subject=7, session=1):
    """Keystroke rows for typing each word followed by Enter."""
    rows, t = [], 0.0
    for word in words:
        buf = ""
        for ch in word:
            buf += ch
            t += 0.2
            rows.append([condition, block_c, ch, buf, t])
        t += 0.2
        rows.append([condition, block_c, "enter", buf + "enter", t])
    df = pd.DataFrame(
        rows, columns=["condition", "block_c", "response", "multichar_response", "block_dur"]
    )
    df["subject_id"], df["session_number"], df["task_vers"] = subject, session, "A"
    return df


def _wl_session(immediate_words, subject=7, session=1):
    """A complete WL session. The first entry of each phase is unscored, so
    each phase starts with a lead-in word."""
    header = pd.DataFrame([{"response": "space", "multichar_response": ""}])
    parts = [header]
    for trial in range(1, 6):
        parts.append(_type_words(["x"] + LIST_A1 if trial == 1 else LIST_A1,
                                 "learn", trial, subject, session))
    parts.append(_type_words(["x"] + LIST_A2, "distr", 6, subject, session))
    parts.append(_type_words(["x"] + immediate_words, "immed", 7, subject, session))
    df = pd.concat(parts, ignore_index=True)
    df["subject_id"], df["session_number"], df["task_vers"] = subject, session, "A"
    return df


def _dwl_session(words, subject=7, session=1):
    header = {"condition": None, "response": "space", "multichar_response": ""}
    rows = [header] + [
        {"condition": "delay", "response": "enter", "multichar_response": w + "enter"}
        for w in words
    ]
    df = pd.DataFrame(rows)
    df["subject_id"], df["session_number"], df["task_vers"] = subject, session, "A"
    return df


def test_wl_low_mean_similarity_is_category_2():
    df_all, category = WL_QC().wl_qc(_wl_session(["zzzz", "qqqq", "xyxy"]), "A")
    assert category == 2


def test_wl_bad_first_entry_but_good_mean_is_category_1():
    _, category = WL_QC().wl_qc(_wl_session(["zzzz"] + LIST_A1), "A")
    assert category == 1


def test_dwl_low_mean_similarity_is_category_2():
    _, category = WL_QC().dwl_qc(_dwl_session(["zzzz", "qqqq", "xyxy"]), "A")
    assert category == 2


def test_dwl_bad_first_entry_but_good_mean_is_category_1():
    _, category = WL_QC().dwl_qc(_dwl_session(["zzzz"] + LIST_A1), "A")
    assert category == 1


def test_wl_plot_reference_line_matches_scoring_threshold():
    df_all, _ = WL_QC().wl_qc(_wl_session(LIST_A1), "A")
    ax = MEM_PLOTS().wl_plot(df_all)
    hlines = [ln for ln in ax.lines if len(set(ln.get_ydata())) == 1]
    assert any(ln.get_ydata()[0] == WL_MATCH_THRESHOLD for ln in hlines)


def _write(df, root, subject, session, task):
    out = root / "obs" / "UI" / str(subject) / task / "data"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{subject}_ses-{session}_cat-1.csv", index=False)


@pytest.fixture
def wl_master(tmp_path):
    data_root = tmp_path / "data"
    _write(_wl_session(LIST_A1, subject=101), data_root, 101, 1, "WL")      # WL only
    _write(_dwl_session(LIST_A1, subject=102), data_root, 102, 1, "DWL")    # DWL only
    _write(_wl_session(LIST_A1, subject=103), data_root, 103, 1, "WL")      # both
    _write(_dwl_session(LIST_A1, subject=103), data_root, 103, 1, "DWL")
    META_RECREATE(data_root=data_root, meta_root=tmp_path / "meta").recreate("wl")
    df = pd.read_csv(tmp_path / "meta" / "wl_master.csv")
    return df.set_index("subject_id")


def test_missing_dwl_gives_na_delay(wl_master):
    row = wl_master.loc[101]
    assert row["task"] == "WL"
    assert pd.isna(row["delay"])
    assert row["immediate"] == len(LIST_A1)


def test_missing_wl_gives_na_wl_columns(wl_master):
    row = wl_master.loc[102]
    assert row["task"] == "DWL"
    assert row["delay"] == len(LIST_A1)
    assert row[[c for c in WL_SCORE_COLUMNS if c != "delay"]].isna().all()


def test_both_files_fill_every_column(wl_master):
    row = wl_master.loc[103]
    assert row["task"] == "WL"
    assert row[WL_SCORE_COLUMNS].notna().all()
    assert row["delay"] == len(LIST_A1)
