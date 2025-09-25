from data_processing.pull_handler import Pull
from data_processing.cc_qc import CCqC
from data_processing.mem_qc import MEM_QC
from data_processing.ps_qc import PS_QC
from data_processing.utils import CONVERT_TO_CSV
from data_processing.utils import QC_UTILS
from data_processing.wl_qc import WL_QC
from data_processing.plot_utils import CC_PLOTS, MEM_PLOTS, PS_PLOTS
from data_processing.save_utils import SAVE_EVERYTHING
import pandas as pd
from pathlib import Path
import os, atexit


class Handler:

    def __init__(self):
        self.IDs = {
            "AF": [945, 960, 990, 898, 919, 932],
            "ATS": [947, 961, 984, 918, 920, 933],
            "DSST": [949, 975, 986, 901, 959, 935],
            "DWL": [948, 974, 985, 900, 921, 934],
            "FN": [950, 964, 987, 902, 923, 936],
            "LC": [951, 976, 988, 903, 924, 937],
            "NF": [980, 981, 982, 978, 979, 977],
            "NNB": [946, 967, 989, 905, 929, 939],
            "NTS": [953, 968, 991, 906, 930, 940],
            "PC": [954, 969, 992, 912, 925, 941],
            "SM": [955, 970, 993, 916, 926, 996],
            "VNB": [957, 971, 994, 915, 928, 943],
            "WL": [958, 972, 995, 910, 927, 944]
        }

        self.master_acc = pd.DataFrame(columns=['task','subject_id','session','condition','accuracy'])
        self.cc_master  = pd.DataFrame(columns=['task','subject_id','session','condition','accuracy','mean_rt'])
        self.ps_master  = pd.DataFrame(columns=['task','subject_id','session','condition','count_correct'])
        self.mem_master = pd.DataFrame(columns=['task','subject_id','session','condition','count_correct','mean_rt','accuracy'])
        self.wl_master  = pd.DataFrame(columns=['task','subject_id','session','block_1','block_2','block_3','block_4','block_5','distraction','immediate','delay'])

        # --- NEW: where to save masters ---
        self.base_dir = Path(__file__).parents[1]   # project root (adjust if needed)
        self.meta_dir = self.base_dir / "meta"      # keep masters together
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        # autosave on normal process exit
        atexit.register(self._persist_all_masters)

    def _atomic_to_csv(self, df: pd.DataFrame, path: Path, index: bool = False):
        """Write CSV atomically to avoid partial files."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(tmp, index=index)
        os.replace(tmp, path)  # atomic on POSIX

    def _persist_all_masters(self):
        # save each master; create both “wide” and flat for wl_master
        self._atomic_to_csv(self.master_acc, self.meta_dir / "master_acc.csv", index=False)
        self._atomic_to_csv(self.cc_master,  self.meta_dir / "cc_master.csv",  index=False)
        self._atomic_to_csv(self.ps_master,  self.meta_dir / "ps_master.csv",  index=False)
        self._atomic_to_csv(self.mem_master, self.meta_dir / "mem_master.csv", index=False)

        # WL: save as-is (wide) and also a flat version for easy joins
        wl_wide_path = self.meta_dir / "wl_master_wide.csv"
        self._atomic_to_csv(self.wl_master, wl_wide_path, index=False)

        wl_flat = self.wl_master.copy()
        # if you ever switch wl_master to a MultiIndex, this handles it
        if isinstance(wl_flat.index, pd.MultiIndex):
            wl_flat = wl_flat.reset_index()
        self._atomic_to_csv(wl_flat, self.meta_dir / "wl_master.csv", index=False)

    def pull(self, task):
        pull_instance = Pull(
            self.IDs[task],
            tease="WEEEEEEEEEEEEEE",
            token="jap_5ThOJ14yf7z1EPEUpAoZYMWoETZcmJk305719",
            taskName=task,
            proxy=False
        )

        txt_dfs = pull_instance.load(days_ago=127)
        return self.convert_to_csv(txt_dfs, task)

    def convert_to_csv(self, txt_dfs, task):
        csv_instance = CONVERT_TO_CSV(task)
        csv_dfs = csv_instance.convert_to_csv(txt_dfs)
        return csv_dfs, self.choose_construct(csv_dfs, task)


    def choose_construct(self, csv_dfs, task):

        if task in ['NF', 'AF', 'NTS', 'ATS', 'NNB', 'VNB']:
            return self.qc_cc_dfs(csv_dfs, task)
        elif task in ['FN', 'SM']:
            return self.qc_mem_dfs(csv_dfs, task)
        elif task in ['WL', 'DWL']:
            return self.qc_wl_dfs(csv_dfs, task)
        elif task in ['PC', 'LC', 'DSST']:
            return self.qc_ps_dfs(csv_dfs, task)
        else:
            return print("ERROR: TASK NAME NOT VALID")

    def qc_cc_dfs(self, dfs, task):
        categories, plots = [], []
        plot_instance = CC_PLOTS()
        utils = QC_UTILS()
        # pick the grouping column for accuracy/RT
        cond_col = "condition" if task in ["AF", "NF", "NNB", "VNB"] else "block_cond"

        # Configure QC with task-specific column names/symbols
        qc_instance = CCqC(
            task,
            MAXRT=1800,
            RT_COLUMN_NAME="response_time",
            ACC_COLUMN_NAME="correct",
            CORRECT_SYMBOL=1,
            INCORRECT_SYMBOL=0,
            COND_COLUMN_NAME=cond_col,
        )


        for df in dfs:
            subject = df["subject_id"].iloc[0]
            if "session" in df.columns:
                session = df["session"].iloc[0]
            elif "session_number" in df.columns:
                session = df["session_number"].iloc[0]
            else:
                session = None

            # --- Run QC + plots (kept as you had it) ---
            if task in ["AF", "NF"]:
                category, _ = qc_instance.cc_qc(df, threshold=0.5)
                plot = plot_instance.af_nf_plot(df)
            elif task in ["NNB", "VNB"]:
                category, _ = qc_instance.cc_qc(df, threshold=0.5)
                plot = plot_instance.nnb_vnb_plot(df)
            else:
                category = qc_instance.cc_qc(df, threshold=0.5, TS=True)
                plot = plot_instance.ats_nts_plot(df)

            categories.append([subject, category, df])
            plots.append([subject, plot])

            # --- Compute metrics by condition using your helpers ---
            # Use the column names from qc_instance so this is task-agnostic
            acc_by = utils.get_acc_by_block_cond(
                df,
                block_cond_column_name=qc_instance.COND_COLUMN_NAME,
                acc_column_name=qc_instance.ACC_COLUMN_NAME,
                correct_symbol=qc_instance.CORRECT_SYMBOL,
                incorrect_symbol=qc_instance.INCORRECT_SYMBOL,
            )
            rt_by = utils.get_avg_rt(
                df,
                rt_column_name=qc_instance.RT_COLUMN_NAME,
                conditon_column_name=qc_instance.COND_COLUMN_NAME,
            )

            # unify all conditions present in either dict
            all_conditions = sorted(set(acc_by.keys()) | set(rt_by.keys()), key=lambda x: str(x))

            # append one row per condition
            rows = []
            for cond in all_conditions:
                rows.append({
                    'task'      : task,
                    'subject_id': subject,
                    'session'   : session,
                    'condition' : cond,
                    'accuracy'  : float(acc_by.get(cond, 0.0)),
                    'mean_rt'   : float(rt_by.get(cond, float('nan'))),
                })

            # append to cc_master
            if rows:
                self.cc_master = pd.concat([self.cc_master, pd.DataFrame(rows)], ignore_index=True)

        # save artifacts (unchanged)
        save_instance = SAVE_EVERYTHING()
        save_instance.save_dfs(categories=categories, task=task)
        save_instance.save_plots(plots=plots, task=task)


        self._persist_all_masters()  # save after each task for safety
        return categories, plots


    def qc_ps_dfs(self, dfs, task):
        categories, plots = [], []
        plot_instance = PS_PLOTS()
        if task in ['PC', 'LC']:
            ps_instance = PS_QC('response_time', 'correct', 1, 0, 'block_c', 30000)
            for df in dfs:
                subject = df['subject_id'][1]
                category = ps_instance.ps_qc(df, threshold=0.6,)
                if task == 'PC':
                    plot = plot_instance.lc_plot(df)
                elif task == 'LC':
                    plot = plot_instance.lc_plot(df)
                categories.append([subject, category, df])
                plots.append([subject, plot])

        else:
            ps_instance = PS_QC('block_dur', 'correct', 1, 0, 'block_c', 125)
            for df in dfs:
                subject = df['subject_id'][1]
                category = ps_instance.ps_qc(df, threshold=0.6, DSST=True)
                plot = plot_instance.dsst_plot(df)
                categories.append([subject, category, df])
                plots.append([subject, plot])

        save_instance = SAVE_EVERYTHING()
        save_instance.save_dfs(categories=categories, task=task)
        save_instance.save_plots(plots=plots, task=task)

        # append accuracies by block/condition to master_acc
        master_rows = []
        from data_processing.utils import QC_UTILS
        qc_util = QC_UTILS()
        cond_col = 'block_c'
        for subject, _, df in categories:
            # preserve session information from available column
            if 'session' in df.columns:
                session = df['session'].iloc[0]
            elif 'session_number' in df.columns:
                session = df['session_number'].iloc[0]
            else:
                session = None
            acc_by = qc_util.get_acc_by_block_cond(
                df,
                block_cond_column_name=cond_col,
                acc_column_name='correct',
                correct_symbol=1,
                incorrect_symbol=0,
            )
            for cond, acc in acc_by.items():
                master_rows.append([task, subject, session, cond, float(acc)])
        if master_rows:
            self.master_acc = pd.concat(
                [self.master_acc,
                 pd.DataFrame(master_rows,
                              columns=['task','subject_id','session','condition','accuracy'])],
                ignore_index=True,
            )

        # append count of correct responses by block/condition to ps_master
        ps_rows = []
        for subject, _, df in categories:
            # preserve session information from available column
            if 'session' in df.columns:
                session = df['session'].iloc[0]
            elif 'session_number' in df.columns:
                session = df['session_number'].iloc[0]
            else:
                session = None
            count_by = qc_util.get_count_correct(
                df,
                block_cond_column_name=cond_col,
                acc_column_name='correct',
                correct_symbol=1,
            )
            for cond, count in count_by.items():
                ps_rows.append([task, subject, session, cond, int(count)])
        if ps_rows:
            self.ps_master = pd.concat(
                [self.ps_master,
                 pd.DataFrame(ps_rows,
                              columns=['task','subject_id','session','condition','count_correct'])],
                ignore_index=True,
            )

        self._persist_all_masters()  # save after each task for safety
        return categories, plots


    def qc_mem_dfs(self, dfs, task):
        plot_instance = MEM_PLOTS()
        categories, plots = [], []
        if task in ['FN']:
            mem_instance = MEM_QC('response_time', 'correct', 1, 0, 'block_c', 4000)
            for df in dfs:
                subject = df['subject_id'][1]
                category = mem_instance.fn_sm_qc(df, threshold=0.5)
                plot = plot_instance.fn_plot(df)
                categories.append([subject, category, df])
                plots.append([subject, plot])
        elif task in ['SM']:
            mem_instance = MEM_QC('response_time', 'correct', 1, 0, 'block_c', 2000)
            for df in dfs:
                subject = df['subject_id'][1]
                category = mem_instance.fn_sm_qc(df, threshold=0.5)
                plot = plot_instance.sm_plot(df)
                categories.append([subject, category, df])
                plots.append([subject, plot])
        save_instance = SAVE_EVERYTHING()
        save_instance.save_dfs(categories=categories, task=task)
        save_instance.save_plots(plots=plots, task=task)

        # append accuracies by block/condition to master_acc
        master_rows = []
        from data_processing.utils import QC_UTILS
        qc_util = QC_UTILS()
        cond_col = 'block_c'
        for subject, _, df in categories:
            # preserve session information from available column
            if 'session' in df.columns:
                session = df['session'].iloc[0]
            elif 'session_number' in df.columns:
                session = df['session_number'].iloc[0]
            else:
                session = None
            acc_by = qc_util.get_acc_by_block_cond(
                df,
                block_cond_column_name=cond_col,
                acc_column_name='correct',
                correct_symbol=1,
                incorrect_symbol=0,
            )
            for cond, acc in acc_by.items():
                master_rows.append([task, subject, session, cond, float(acc)])
        if master_rows:
            self.master_acc = pd.concat(
                [self.master_acc,
                 pd.DataFrame(master_rows,
                              columns=['task','subject_id','session','condition','accuracy'])],
                ignore_index=True,
            )

        # Append richer MEM metrics (count, RT, accuracy) to mem_master
        mem_rows = []
        count_by = {}
        rt_by = {}
        for subject, _, df in categories:
            if 'session' in df.columns:
                session = df['session'].iloc[0]
            elif 'session_number' in df.columns:
                session = df['session_number'].iloc[0]
            else:
                session = None

            # reuse cond column and symbol config
            count_by = qc_util.get_count_correct(
                df,
                block_cond_column_name=cond_col,
                acc_column_name='correct',
                correct_symbol=1,
            )
            acc_by = qc_util.get_acc_by_block_cond(
                df,
                block_cond_column_name=cond_col,
                acc_column_name='correct',
                correct_symbol=1,
                incorrect_symbol=0,
            )
            rt_by = qc_util.get_avg_rt(
                df,
                rt_column_name='response_time',
                conditon_column_name=cond_col,
            )

            all_conditions = sorted(set(count_by.keys()) | set(acc_by.keys()) | set(rt_by.keys()), key=lambda x: str(x))
            for cond in all_conditions:
                mem_rows.append({
                    'task': task,
                    'subject_id': subject,
                    'session': session,
                    'condition': cond,
                    'count_correct': int(count_by.get(cond, 0)),
                    'mean_rt': float(rt_by.get(cond, float('nan'))),
                    'accuracy': float(acc_by.get(cond, 0.0)),
                })

        if mem_rows:
            self.mem_master = pd.concat([self.mem_master, pd.DataFrame(mem_rows)], ignore_index=True)


        self._persist_all_masters()  # save after each task for safety

        return categories, plots


    def qc_wl_dfs(self, dfs, task):
        categories, plots = [], []
        plot_instance = MEM_PLOTS()

        if task == 'WL':
            for df in dfs:
                subject = df['subject_id'].iloc[1]
                version = df['task_vers'].iloc[1]
                session = (df['session_number'].iloc[1] if 'session_number' in df.columns
                           else (df['session'].iloc[1] if 'session' in df.columns else None))

                wl_instance = WL_QC()
                df_all, category = wl_instance.wl_qc(df, version)
                plot = plot_instance.wl_plot(df_all)

                counts = wl_instance.wl_count_correct(df_all)  # wide: learn_1..learn_5, distraction, immediate
                upd = {
                    'block_1'   : counts['learn_1'].iat[0],
                    'block_2'   : counts['learn_2'].iat[0],
                    'block_3'   : counts['learn_3'].iat[0],
                    'block_4'   : counts['learn_4'].iat[0],
                    'block_5'   : counts['learn_5'].iat[0],
                    'distraction': counts['distraction'].iat[0],
                    'immediate' : counts['immediate'].iat[0],
                    # do NOT set delay here; leave whatever exists
                }
                self._upsert_wl_master(subject, session, upd)

                categories.append([subject, category, df])
                plots.append([subject, plot])

        elif task == 'DWL':
            for df in dfs:
                subject = df['subject_id'].iloc[1]
                version = df['task_vers'].iloc[1]
                session = (df['session_number'].iloc[1] if 'session_number' in df.columns
                           else (df['session'].iloc[1] if 'session' in df.columns else None))

                dwl_instance = WL_QC()
                df_all, category = dwl_instance.dwl_qc(df, version)
                plot = plot_instance.dwl_plot(df_all)

                counts_delay = dwl_instance.dwl_count_correct(df_all)  # wide: just 'delay'
                upd = {'delay': counts_delay['delay'].iat[0]}
                self._upsert_wl_master(subject, session, upd)

                categories.append([subject, category, df])
                plots.append([subject, plot])

        # maybe: materialize wl_master back to columns if you prefer
        # wl_master_out = self.wl_master.reset_index()

        save_instance = SAVE_EVERYTHING()
        save_instance.save_dfs(categories=categories, task=task)
        save_instance.save_plots(plots=plots, task=task)

        self._persist_all_masters()  # save after each task for safety
        return categories, plots


    def _upsert_wl_master(self, subject, session, upd: dict):
        import pandas as pd

        # --- ensure identifier columns exist ---
        for col in ['subject_id', 'session', 'task']:
            if col not in self.wl_master.columns:
                self.wl_master[col] = pd.NA

        # --- normalize id types a bit ---
        def _to_int_if_possible(v):
            try:
                return int(v)
            except Exception:
                return v
        subj = _to_int_if_possible(subject)
        sess = _to_int_if_possible(session)

        # Infer task label if caller didn’t include it
        # (WL writes learn/distraction/immediate; DWL writes only delay)
        inferred_task = ('DWL' if set(upd.keys()) == {'delay'} else 'WL')

        # If you want the caller to decide, let 'task' in upd override our guess:
        task_val = upd.get('task', inferred_task)

        # Build the full row to upsert
        row = {'subject_id': subj, 'session': sess, 'task': task_val, **upd}

        idx = self.wl_master.index
        is_multi = isinstance(idx, pd.MultiIndex)

        # ---------- Preferred path: upsert by columns (robust, index-agnostic) ----------
        # If we already keep subject/session as columns, do a boolean mask upsert.
        if {'subject_id', 'session'}.issubset(self.wl_master.columns):
            mask = (
                (self.wl_master['subject_id'] == subj) &
                (self.wl_master['session'] == sess)
            )
            if mask.any():
                for k, v in row.items():
                    self.wl_master.loc[mask, k] = v
            else:
                self.wl_master = pd.concat([self.wl_master, pd.DataFrame([row])], ignore_index=True)
            return

        # ---------- Fallback: handle MultiIndex (subject, session) ----------
        # If you truly rely on a MultiIndex instead of columns, keep this path.
        def _coerce(val, dtype):
            try:
                if pd.api.types.is_integer_dtype(dtype): return int(val)
                if pd.api.types.is_float_dtype(dtype):   return float(val)
                return str(val)
            except Exception:
                return val

        if is_multi:
            # Expect two levels: subject, session
            level0_dtype = idx.levels[0].dtype
            level1_dtype = idx.levels[1].dtype
            key = (_coerce(subj, level0_dtype), _coerce(sess, level1_dtype))

            if key not in idx:
                new_index = pd.MultiIndex.from_tuples([key], names=idx.names)
                empty_row = pd.DataFrame(index=new_index, columns=self.wl_master.columns)
                self.wl_master = pd.concat([self.wl_master, empty_row], axis=0)

            # Write the fields we know about
            # If subject/session columns exist, keep them in sync too
            if 'subject_id' in self.wl_master.columns:
                self.wl_master.loc[key, 'subject_id'] = subj
            if 'session' in self.wl_master.columns:
                self.wl_master.loc[key, 'session'] = sess
            if 'task' in self.wl_master.columns:
                self.wl_master.loc[key, 'task'] = task_val

            for col, val in upd.items():
                self.wl_master.loc[key, col] = val
            return

        # ---------- Last resort: single index that is not subject/session ----------
        # Reindex by the subject (not ideal). Still write id columns so they’re visible.
        dtype = idx.dtype
        key = _coerce(subj, dtype)
        if key not in idx:
            self.wl_master = self.wl_master.reindex(list(self.wl_master.index) + [key])

        # Keep id columns updated
        self.wl_master.loc[key, 'subject_id'] = subj
        self.wl_master.loc[key, 'session']    = sess
        self.wl_master.loc[key, 'task']       = task_val
        for col, val in upd.items():
            self.wl_master.loc[key, col] = val

if __name__ == '__main__':
    import os
    import sys

    task_list = ['AF', 'NF', 'NTS', 'ATS', 'NNB', 'VNB', 'WL', 'DWL', 'FN', 'SM', 'PC', 'LC', 'DSST']
    if sys.argv[1] == 'all':
        instance = Handler()
        for task in task_list:
            csv_dfs = instance.pull(task=task)
    elif sys.argv[1] in task_list:
        instance = Handler()
        csv_dfs = instance.pull(task=sys.argv[1])
