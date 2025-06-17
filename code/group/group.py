import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io
from PIL import Image

class Group:

    def __init__(self) -> None:
        pass

    def group(self):
        self.flanker()
        self.task_switching()
        self.n_back()
        self.ps_mem()
        return None

    def flanker(self):

        columns_to_extract = ["subject_id", "response_time", "correct", "condition", "task_vers"]

        grouped_data = {}  # store grouped RT per task

        for task in ['AF', 'NF']:
            df = self.load_task_session_data(task, columns_to_extract)
            df = self.add_study_and_site_columns(df)
            df = df[df['correct'] == 1]
            grouped = df.groupby(['condition', 'subjectID', 'session', 'task_vers'])['response_time'].mean().reset_index()
            grouped_data[task] = grouped
                    
        self.plot_flanker_rt_by_study_site_session_version(
            grouped_data["AF"],
            grouped_data["NF"]
        )
        return None

    def task_switching(self):
        columns_to_extract = ['block', 'block_cond', 'correct', 'response_time', 'con_img', 'task_vers']        # === Load and prepare raw task-switching data ===
        ats_ses = self.add_study_and_site_columns(self.load_task_session_data("ATS", columns_to_extract))
        nts_ses = self.add_study_and_site_columns(self.load_task_session_data("NTS", columns_to_extract))

        # Filter for test blocks only
        ats_ses = ats_ses[ats_ses['block'] == 'test']
        nts_ses = nts_ses[nts_ses['block'] == 'test']

        # Compute cost metrics (e.g., switch vs repeat cost)
        ats_ses_ts = self.compute_task_switch_costs(ats_ses)
        nts_ses_ts = self.compute_task_switch_costs(nts_ses)

        # Convert subject to string once
        ats_ses_ts['subject'] = ats_ses_ts['subject'].astype(str)
        nts_ses_ts['subject'] = nts_ses_ts['subject'].astype(str)

        # === Create group subsets based on subject ID prefix ===
        def group_subset(df, prefixes):
            return df[df['subject'].str.startswith(tuple(prefixes))]

        group_map = {
            'obs': ['7'],
            'int': ['8', '9'],
            'ui': ['8'],
            'ne': ['9']
        }

        ats_groups = {k: group_subset(ats_ses_ts, v) for k, v in group_map.items()}
        nts_groups = {k: group_subset(nts_ses_ts, v) for k, v in group_map.items()}

        # === Compute wide-format summary stats for each group ===
        summaries = {}
        for group in group_map:
            summaries[f'{group}_ats'] = self.compute_wide_cost_stats_by_task_vers_session(
                ats_groups[group], cost_cols=["single", "repeat", "switching"]
            )
            summaries[f'{group}_nts'] = self.compute_wide_cost_stats_by_task_vers_session(
                nts_groups[group], cost_cols=["single", "repeat", "switching"]
            )

        # === Generate individual FacetGrid plots for each group ===
        grids = [
            self.plot_affective_cost_by_task_vers_and_session(summaries['obs_ats'], summaries['obs_nts'], crit="Observational (Baseline)"),
            self.plot_affective_cost_by_task_vers_and_session(summaries['int_ats'], summaries['int_nts'], crit="Intervention (Baseline)"),
            self.plot_affective_cost_by_task_vers_and_session(summaries['ui_ats'],  summaries['ui_nts'],  crit="UI Site"),
            self.plot_affective_cost_by_task_vers_and_session(summaries['ne_ats'],  summaries['ne_nts'],  crit="NE Site")
        ]

        # === Render each FacetGrid to memory with consistent sizing ===
        dpi = 300
        base_height = 4
        base_aspect = 1.2
        buffers = []

        for g in grids:
            nrows, ncols = g.axes.shape
            width_in = ncols * (base_height * base_aspect)
            height_in = nrows * base_height
            g.fig.set_size_inches(width_in, height_in)
            g.fig.tight_layout()
            buf = io.BytesIO()
            g.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=1)
            plt.close()
            buf.seek(0)
            buffers.append(Image.open(buf))

        # === Stack all plots vertically ===
        fig, axes = plt.subplots(
            nrows=len(buffers),
            ncols=1,
            figsize=(width_in, sum(img.height for img in buffers) / dpi),
            dpi=dpi,
            constrained_layout=True
        )

        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        for ax, img in zip(axes, buffers):
            ax.imshow(img)
            ax.axis("off")

        # === Save final composite plot ===
        save_dir = "./group/plots"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "task_switching.png")

        fig.savefig(save_path, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)

        return None


    def n_back(self):
        columns_to_extract = ['block', 'condition', 'correct', 'response_time', 'task_vers']
        nnb_ses_df = self.load_task_session_data("NNB", columns_to_extract)
        vnb_ses_df = self.load_task_session_data("VNB", columns_to_extract)
        nnb_ses_df = self.add_study_and_site_columns(nnb_ses_df)
        vnb_ses_df = self.add_study_and_site_columns(vnb_ses_df)
        nnb_ses_df = nnb_ses_df[nnb_ses_df['block']=='test']
        vnb_ses_df = vnb_ses_df[vnb_ses_df['block']=='test']
        nnb_ses_df.dropna(subset=['correct'], inplace=True)
        vnb_ses_df.dropna(subset=['correct'], inplace=True)

        all_grids = []
        all_grids.append(
            self.plot_proportion_correct_by_session_rows(
                nnb_ses_df[nnb_ses_df['study'] == 'obs'],
                vnb_ses_df[vnb_ses_df['study'] == 'obs'],
                crit="Observational (Baseline)"
            )
        )
        all_grids.append(
            self.plot_proportion_correct_by_session_rows(
                nnb_ses_df[nnb_ses_df['study'] == 'int'],
                vnb_ses_df[vnb_ses_df['study'] == 'int'],
                crit="Intervention (Baseline)"
            )
        )
        all_grids.append(
            self.plot_proportion_correct_by_session_rows(
                nnb_ses_df[nnb_ses_df['site'] == 'UI'],
                vnb_ses_df[vnb_ses_df['site'] == 'UI'],
                crit="UI Site"
            )
        )
        all_grids.append(
            self.plot_proportion_correct_by_session_rows(
                nnb_ses_df[nnb_ses_df['site'] == 'NE'],
                vnb_ses_df[vnb_ses_df['site'] == 'NE'],
                crit="NE Site"
            )
        )
        buffers = []
        dpi = 300
        base_height = 4
        base_aspect = 1.2

        for g in all_grids:
            nrows, ncols = g.axes.shape
            w = ncols * base_height * base_aspect
            h = nrows * base_height
            g.fig.set_size_inches(w, h)
            g.fig.tight_layout()

            buf = io.BytesIO()
            g.fig.set_size_inches(w, h)
            g.fig.tight_layout(rect=[0, 0, 1, 0.95])  # Add some headroom
            g.savefig(buf, format="png", dpi=dpi, bbox_inches=None, pad_inches=1)
            buf.seek(0)
            buffers.append(Image.open(buf))

        fig, axes = plt.subplots(
            nrows=len(buffers),
            ncols=1,
            figsize=(w, sum(im.height for im in buffers) / dpi),
            dpi=dpi,
            constrained_layout=True
        )
        if not hasattr(axes, "__iter__"):
            axes = [axes]

        for ax, im in zip(axes, buffers):
            ax.imshow(im)
            ax.axis("off")

        os.makedirs("./group/plots", exist_ok=True)
        fig.savefig("./group/plots/n_back.png",
                    dpi=dpi, bbox_inches=None, pad_inches=0.3)
        plt.close(fig)
        return None

    def ps_mem(self):
        # ─── Pattern Comparison ───────────────────────────────────────────────────────
        pc_cols = ['subject_id','session_number','task_vers','correct','condition']
        pc_df   = self.load_and_tag("PC", pc_cols)
        pc_summ = self.summarize_task(pc_df, filter_col='condition', filter_val='test')

        # Access:
        #   pc_summ['total'], pc_summ['obs'], pc_summ['int'], pc_summ['UI'], pc_summ['NE']


        # ─── Letter Comparison ────────────────────────────────────────────────────────
        lc_cols = ['subject_id','session_number','task_vers','correct','condition']
        lc_df   = self.load_and_tag("LC", lc_cols)
        lc_summ = self.summarize_task(lc_df, filter_col='condition', filter_val='test')


        # ─── Spatial Memory ──────────────────────────────────────────────────────────
        sm_cols = ['subject_id','session_number','task_vers','correct','block']
        sm_df   = self.load_and_tag("SM", sm_cols)
        sm_summ = self.summarize_task(sm_df, filter_col='block', filter_val='test')


        # ─── Digit Symbol Substitution ───────────────────────────────────────────────
        dsst_cols = ['subject_id','session_number','task_vers','correct','condition']
        dsst_df   = self.load_and_tag("DSST", dsst_cols)
        dsst_summ = self.summarize_task(dsst_df, filter_col='condition', filter_val='test')


        # ─── Face Name ───────────────────────────────────────────────────────────────
        fn_cols = ['subject_id','session_number','task_vers','correct','block']
        fn_df   = self.load_and_tag("FN", fn_cols)
        fn_summ = self.summarize_task(fn_df, filter_col='block', filter_val='test')
        pc_summary = pc_summ['total']
        lc_summary = lc_summ['total']
        sm_summary = sm_summ['total']
        dsst_summary = dsst_summ['total']
        fn_summary = fn_summ['total']
        # --- assume these are your raw summary DataFrames: 
        #     pc_summary, lc_summary, sm_summary, dsst_summary, fn_summary

        # 1) Observational study
        pc_obs   = pc_summary[  pc_summary['study']=='obs']
        lc_obs   = lc_summary[  lc_summary['study']=='obs']
        sm_obs   = sm_summary[  sm_summary['study']=='obs']
        dsst_obs = dsst_summary[dsst_summary['study']=='obs']
        fn_obs   = fn_summary[  fn_summary['study']=='obs']

        # rename to match the plotting function
        pc_obs_df   = pc_obs.  rename(columns={'subject_id':'subjectID', 'n_correct':'correct'})
        lc_obs_df   = lc_obs.  rename(columns={'subject_id':'subjectID', 'n_correct':'correct'})
        sm_obs_df   = sm_obs.  rename(columns={'subject_id':'subjectID', 'prop_correct':'proportion_correct'})
        dsst_obs_df = dsst_obs.rename(columns={'subject_id':'subjectID', 'n_correct':'correct'})
        fn_obs_df   = fn_obs.  rename(columns={'subject_id':'subjectID', 'prop_correct':'proportion_correct'})

        fig_obs = self.plot_task_performance_by_session(
            pc_obs_df, lc_obs_df, sm_obs_df, dsst_obs_df, fn_obs_df,
            crit='Observational Study'
        )


        # 2) Intervention study
        pc_int   = pc_summary[  pc_summary['study']=='int']
        lc_int   = lc_summary[  lc_summary['study']=='int']
        sm_int   = sm_summary[  sm_summary['study']=='int']
        dsst_int = dsst_summary[dsst_summary['study']=='int']
        fn_int   = fn_summary[  fn_summary['study']=='int']

        pc_int_df   = pc_int.  rename(columns={'subject_id':'subjectID', 'n_correct':'correct'})
        lc_int_df   = lc_int.  rename(columns={'subject_id':'subjectID', 'n_correct':'correct'})
        sm_int_df   = sm_int.  rename(columns={'subject_id':'subjectID', 'prop_correct':'proportion_correct'})
        dsst_int_df = dsst_int.rename(columns={'subject_id':'subjectID', 'n_correct':'correct'})
        fn_int_df   = fn_int.  rename(columns={'subject_id':'subjectID', 'prop_correct':'proportion_correct'})

        fig_int = self.plot_task_performance_by_session(
            pc_int_df, lc_int_df, sm_int_df, dsst_int_df, fn_int_df,
            crit='Intervention Study'
        )


        # 3) UI site, intervention only
        mask_ui_int = (pc_summary['study']=='int') & (pc_summary['site']=='UI')
        pc_ui   = pc_summary[ mask_ui_int]
        lc_ui   = lc_summary[ mask_ui_int]
        sm_ui   = sm_summary[ mask_ui_int]
        dsst_ui = dsst_summary[mask_ui_int]
        fn_ui   = fn_summary[ mask_ui_int]

        pc_ui_df   = pc_ui.  rename(columns={'subject_id':'subjectID', 'n_correct':'correct'})
        lc_ui_df   = lc_ui.  rename(columns={'subject_id':'subjectID', 'n_correct':'correct'})
        sm_ui_df   = sm_ui.  rename(columns={'subject_id':'subjectID', 'prop_correct':'proportion_correct'})
        dsst_ui_df = dsst_ui.rename(columns={'subject_id':'subjectID', 'n_correct':'correct'})
        fn_ui_df   = fn_ui.  rename(columns={'subject_id':'subjectID', 'prop_correct':'proportion_correct'})

        fig_ui = self.plot_task_performance_by_session(
            pc_ui_df, lc_ui_df, sm_ui_df, dsst_ui_df, fn_ui_df,
            crit='UI Site (Intervention Only)'
        )


        # 4) NE site (all studies)
        mask_ne = pc_summary['site']=='NE'
        pc_ne   = pc_summary[  mask_ne]
        lc_ne   = lc_summary[  mask_ne]
        sm_ne   = sm_summary[  mask_ne]
        dsst_ne = dsst_summary[mask_ne]
        fn_ne   = fn_summary[ mask_ne]

        pc_ne_df   = pc_ne.  rename(columns={'subject_id':'subjectID', 'n_correct':'correct'})
        lc_ne_df   = lc_ne.  rename(columns={'subject_id':'subjectID', 'n_correct':'correct'})
        sm_ne_df   = sm_ne.  rename(columns={'subject_id':'subjectID', 'prop_correct':'proportion_correct'})
        dsst_ne_df = dsst_ne.rename(columns={'subject_id':'subjectID', 'n_correct':'correct'})
        fn_ne_df   = fn_ne.  rename(columns={'subject_id':'subjectID', 'prop_correct':'proportion_correct'})

        fig_ne = self.plot_task_performance_by_session(
            pc_ne_df, lc_ne_df, sm_ne_df, dsst_ne_df, fn_ne_df,
            crit='NE Site'
        )
        # Step 2: Insert them into the list
        all_grids = [fig_obs, fig_int, fig_ui, fig_ne]

        # Step 3: dump each Figure to a buffer and record its size in inches
        buffers = []
        dpi = 300
        sizes = []
        for fig in all_grids:
            # make sure any tight_layout calls inside the fig are applied
            fig.tight_layout(rect=[0, 0, 0.95, 0.93])

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=1)
            buf.seek(0)

            img = Image.open(buf)
            buffers.append(img)

            # record image size in inches for height_ratios later
            w_in = img.width  / dpi
            h_in = img.height / dpi
            sizes.append((w_in, h_in))

            plt.close(fig)   # close this sub-figure

        # Step 4: stitch them into one tall figure, preserving each original height
        max_w_in = max(w for w, h in sizes)
        total_h_in = sum(h for w, h in sizes)

        # create a new figure with a GridSpec whose row heights match each sub-figure
        fig = plt.figure(figsize=(max_w_in, total_h_in), dpi=dpi)
        gs  = fig.add_gridspec(nrows=len(buffers), ncols=1,
                            height_ratios=[h for w, h in sizes])

        for idx, img in enumerate(buffers):
            ax = fig.add_subplot(gs[idx, 0])
            ax.imshow(img)
            ax.axis("off")

        os.makedirs("./group/plots", exist_ok=True)
        fig.savefig("./group/plots/ps_mem.png",
                    dpi=dpi, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
    # ================================================================
    # PLOT FUNCTIONS
    # These functions create the relevant plots
    # Moved to the bottom to not clog space
    # ================================================================
    def plot_task_performance_by_session(self, pc_df, lc_df, sm_df, dsst_df, fn_df, crit="Summary"):
        """
        Multi‐panel summary of task performance by session.
        Expects each df to have columns:
        - subjectID
        - session
        - task_vers
        - one metric column: 'correct' or 'proportion_correct'
        """
        task_info = [
            (pc_df,   'correct',            'Pattern Comparison'),
            (lc_df,   'correct',            'Letter Comparison'),
            (sm_df,   'proportion_correct', 'Spatial Memory'),
            (dsst_df, 'correct',            'Digit Symbol Substitution'),
            (fn_df,   'proportion_correct', 'Face Name'),
        ]

        # figure out how many sessions we have
        sessions = sorted(pc_df['session'].unique())
        n_sessions = len(sessions)

        fig, axes = plt.subplots(n_sessions, 5,
                                figsize=(22, 5 * n_sessions),
                                sharey=False)

        for i, session in enumerate(sessions):
            for j, (df, metric, title) in enumerate(task_info):
                # filter to this session
                df_sess = df[df['session'] == session]

                # pick the right Axes
                if n_sessions > 1:
                    ax = axes[i, j]
                else:
                    ax = axes[j]

                # box + strip
                sns.boxplot(
                    x='task_vers', y=metric, data=df_sess,
                    ax=ax, palette='pastel', fliersize=0, linewidth=1
                )
                sns.stripplot(
                    x='task_vers', y=metric, data=df_sess,
                    ax=ax, color='black',
                    dodge=True, size=4, edgecolor='gray', alpha=0.3
                )

                ax.set_title(f"{title} — Session {session}",
                            fontsize=12, fontweight='bold')
                ax.set_xlabel('Task Version')

                if metric == 'proportion_correct':
                    ax.set_ylabel('Proportion Correct')
                    ax.set_ylim(0, 1)
                else:
                    ax.set_ylabel('Correct Count')

                ax.grid(alpha=0.3, linestyle='--')

        sns.despine(trim=True)
        plt.suptitle(f'Processing Speed Task Performance – {crit}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.95, 0.93])
        return fig


    def plot_flanker_rt_by_study_site_session_version(
        self,
        af_ses_grouped_rt: pd.DataFrame,
        nf_ses_grouped_rt: pd.DataFrame
    ):
        """
        Line‐plots of Flanker RT by condition & affect, faceted by session (rows),
        task version (columns), and split by study (Observational/Intervention),
        site (UI/NE/All), and session.
        Expects columns: ['condition','subjectID','response_time','task_vers','session'].
        """
        def _assign_study_site(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df['subject_str'] = df['subjectID'].astype(str)
            df['study'] = df['subject_str'].apply(
                lambda s: 'Observational' if s.startswith('7')
                        else ('Intervention' if s.startswith(('8','9')) else None)
            )
            df['site'] = df['subject_str'].apply(
                lambda s: 'UI' if s.startswith('8')
                        else ('NE' if s.startswith('9') else None)
            )
            return df[df['study'].notna()].drop(columns=['subject_str'])

        # 1) Map study/site and label affect
        af = _assign_study_site(af_ses_grouped_rt);  af['affect'] = 'Affective'
        nf = _assign_study_site(nf_ses_grouped_rt);  nf['affect'] = 'Neutral'

        # 2) Combine raw data
        combined = pd.concat([af, nf], ignore_index=True)

        # 3) Set up categoricals
        combined['session'] = pd.Categorical(
            combined['session'],
            categories=sorted(combined['session'].unique()),
            ordered=True
        )
        combined['task_vers'] = pd.Categorical(
            combined['task_vers'],
            categories=sorted(combined['task_vers'].unique()),
            ordered=True
        )
        combined['condition'] = pd.Categorical(
            combined['condition'],
            categories=['con','inc'],
            ordered=True
        )
        combined['affect'] = pd.Categorical(
            combined['affect'],
            categories=['Affective','Neutral'],
            ordered=True
        )

        # 4) Build full grid index of all combos (3×3×2×2 = 36)
        sess = combined['session'].cat.categories
        vers = combined['task_vers'].cat.categories
        conds = combined['condition'].cat.categories
        affs  = combined['affect'].cat.categories

        full_idx = pd.MultiIndex.from_product(
            [sess, vers, conds, affs],
            names=['session','task_vers','condition','affect']
        )

        # 5) Compute Intervention means and reindex onto full grid
        inter = combined[combined['study']=='Intervention']
        grp = inter.groupby(['session','task_vers','condition','affect'])['response_time'].mean()
        grp_full = grp.reindex(full_idx)

        agg = (
            grp_full
            .reset_index(name='response_time')
            .assign(study='Intervention', site='All')
        )

        # before faceting, create an "All" site copy of the raw data
        all_inter = combined[combined['study']=='Intervention'].copy()
        all_inter['site'] = 'All'
        all_inter['facet_row'] = all_inter.apply(
            lambda r: f"Intervention - All - S{r['session']}", axis=1
        )
        # then concat with the rest (including UI, NE, Observational)
        plot_df = pd.concat([combined, all_inter], ignore_index=True)

        # 7) Build facet row label
        plot_df['facet_row'] = plot_df.apply(
            lambda r: 'Observational'
                    if r['study']=='Observational'
                    else f"{r['study']} - {r['site']} - S{r['session']}",
            axis=1
        )

        # 8) Define row order
        ses_order = list(sess)
        row_order = (
            ['Observational'] +
            [f"Intervention - All - S{ses}" for ses in ses_order] +
            [f"Intervention - UI - S{ses}" for ses in ses_order] +
            [f"Intervention - NE - S{ses}" for ses in ses_order]
        )

        # 9) Plot
        sns.set(style='whitegrid', font_scale=1.1)
        g = sns.FacetGrid(
            plot_df,
            row='facet_row',
            col='task_vers',
            row_order=row_order,
            col_order=vers,
            sharey=True,
            height=3.5,
            aspect=1.2
        )
        g.map_dataframe(
            sns.lineplot,
            x='condition',
            y='response_time',
            hue='affect',
            marker='o',
            # show min/max
            errorbar='sd',
            dashes=False
        )
        g.add_legend(title='Affect', adjust_subtitles=True)
        g.set_axis_labels('Condition','Reaction Time (ms)')
        g.set_titles(row_template='{row_name}', col_template='Version {col_name}')
        plt.subplots_adjust(top=0.95)
        g.fig.suptitle('Flanker RT by Condition, Affect, Study, Site, Session & Version')
        os.makedirs("./group/plots", exist_ok=True)
        plt.savefig("./group/plots/flanker.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_affective_cost_by_task_vers_and_session(
        ats_summary: pd.DataFrame,
        nts_summary: pd.DataFrame,
        crit: str = "Comparison by Task Version and Session"
    ):
        """
        Facet‐grid of cost condition means ±95% CI by session (rows) and task version
        (columns), with separate lines for Affective vs Neutral.

        Parameters:
        -----------
        ats_summary : pd.DataFrame
            Summary for Affective trials; must have columns
            ['task_vers','session','condition','mean','ci'].
        nts_summary : pd.DataFrame
            Summary for Neutral trials; same format.
        crit : str
            Title for the figure.
        """
        # 1) Combine and label
        df = pd.concat([
            ats_summary.assign(affect="Affective"),
            nts_summary.assign(affect="Neutral")
        ], ignore_index=True)

        # 2) Categorical ordering
        df["session"] = pd.Categorical(
            df["session"],
            categories=sorted(df["session"].unique()),
            ordered=True
        )
        df["task_vers"] = pd.Categorical(
            df["task_vers"],
            categories=sorted(df["task_vers"].unique()),
            ordered=True
        )
        df["condition"] = pd.Categorical(
            df["condition"],
            categories=["single", "repeat", "switching"],
            ordered=True
        )

        # 3) Color palette
        palette = {
            "Affective": "#A28BD4",
            "Neutral":    "#EBAF65"
        }

        # 4) Build the FacetGrid
        g = sns.FacetGrid(
            df,
            row="session",
            col="task_vers",
            row_order=df["session"].cat.categories,
            col_order=df["task_vers"].cat.categories,
            sharey=True,
            height=4,
            aspect=1.2
        )

        # 5) Drawing function: one error‐bar line per affect
        def _draw(data, **kwargs):
            ax = plt.gca()
            for aff, grp in data.groupby("affect"):
                ax.errorbar(
                    grp["condition"],
                    grp["mean"],
                    yerr=grp["ci"],
                    fmt="o-",
                    label=aff,
                    color=palette[aff],
                    ecolor=palette[aff],
                    capsize=5,
                    markerfacecolor="white",
                    markeredgecolor=palette[aff]
                )

        g.map_dataframe(_draw)

        # 6) Final touches
        g.add_legend(title="Affect")
        g.set_axis_labels("Cost Condition", "Reaction Time (ms)")
        g.set_titles(row_template="Session {row_name}", col_template="Version {col_name}")
        plt.subplots_adjust(top=0.85)
        g.fig.suptitle(f"{crit} Task Switching", fontsize=16, fontweight="bold", style="italic")
        return g

    @staticmethod
    def plot_proportion_correct_by_session_rows(
        nnb_df: pd.DataFrame,
        vnb_df: pd.DataFrame,
        crit: str = "Proportion Correct by Task Version and Session"
    ):
        """
        Creates one plot where rows = sessions and columns = task versions.
        Each line = affect (Neutral/Affective), linestyle varies by session.
        """
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Label affect and combine
        nnb = nnb_df.assign(affect="Neutral")
        vnb = vnb_df.assign(affect="Affective")
        df = pd.concat([nnb, vnb], ignore_index=True).dropna(subset=['correct'])

        # Subject-level means
        subj = (
            df
            .groupby(['subjectID','session','task_vers','condition','affect'], as_index=False)['correct']
            .mean()
            .rename(columns={'correct': 'mean'})
        )

        # Group-level mean ± SEM
        summary = (
            subj
            .groupby(['session','task_vers','condition','affect'], as_index=False)
            .agg(
                mean=('mean', 'mean'),
                sem=('mean', lambda x: x.std(ddof=1) / np.sqrt(len(x)))
            )
        )

        # Categorical ordering
        summary['session'] = pd.Categorical(summary['session'], ordered=True)
        summary['task_vers'] = pd.Categorical(summary['task_vers'], ordered=True)
        summary['condition'] = pd.Categorical(summary['condition'], ordered=True)

        # Color palette
        palette = {"Affective": "#A28BD4", "Neutral": "#EBAF65"}

        # Plot: session (row) × task_vers (col)
        g = sns.FacetGrid(
            summary,
            row="session",
            col="task_vers",
            sharey=True,
            height=4,
            aspect=1.2,
            row_order=sorted(summary['session'].unique()),
            col_order=sorted(summary['task_vers'].unique())
        )

        def _draw(data, **kwargs):
            ax = plt.gca()
            for aff, group_df in data.groupby("affect"):
                ax.errorbar(
                    group_df["condition"],
                    group_df["mean"],
                    yerr=group_df["sem"],
                    fmt="o-",
                    label=aff,
                    color=palette[aff],
                    ecolor=palette[aff],
                    capsize=5,
                    markerfacecolor="white",
                    markeredgecolor=palette[aff]
                )

        g.map_dataframe(_draw)

        g.set_axis_labels("Condition", "Proportion Correct")
        g.fig.suptitle(crit, y=1.02, fontsize=16, fontweight="bold")

        # Put legend in top-right panel
        top_right = g.axes_dict[(g.row_names[0], g.col_names[-1])]
        top_right.legend(title="Affect", loc="upper right")

        plt.tight_layout()
        return g
    # ================================================================
    # HELPER FUNCTIONS
    # These functions support data loading and preprocessing operations
    # ================================================================

    @staticmethod
    def load_task_session_data(task_name, relevant_columns, root_dir="./data"):
        data_frames = []

        for study in os.listdir(root_dir):
            study_path = os.path.join(root_dir, study)
            if not os.path.isdir(study_path):
                continue

            for site in os.listdir(study_path):
                site_path = os.path.join(study_path, site)
                if not os.path.isdir(site_path):
                    continue

                for subject in os.listdir(site_path):
                    subject_path = os.path.join(site_path, subject)
                    if not os.path.isdir(subject_path):
                        continue

                    task_path = os.path.join(subject_path, task_name, "data")
                    if not os.path.isdir(task_path):
                        continue

                    for file in os.listdir(task_path):
                        if not file.endswith(".csv"):
                            continue

                        try:
                            # Validate session string (expects format like "ses-1")
                            session_str = file.split('_')[1]
                            if not session_str.startswith("ses-") or not session_str[4:].isdigit():
                                print(f"Skipping {file}: invalid session string format")
                                continue
                            session_num = int(session_str.split('-')[1])
                            
                            # Load and filter the file
                            csv_path = os.path.join(task_path, file)
                            temp_df = pd.read_csv(csv_path)
                            filtered_df = temp_df[relevant_columns].copy()
                            filtered_df.insert(0, "session", session_num)
                            filtered_df.insert(0, "subjectID", subject)
                            data_frames.append(filtered_df)
                            print(f"✅ Processed {file} for subject {subject}, session {session_num}")

                        except Exception as e:
                            print(f"❌ Error processing {file}: {e}")
                            continue

        return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame(columns=['subjectID', 'session'] + relevant_columns)


    @staticmethod
    def add_study_and_site_columns(df):
        """
        Adds 'study' and 'site' columns based on the prefix of each subject ID.
        
        Study rules:
            - subjectID starts with '7' → 'obs'
            - otherwise → 'int'

        Site rules:
            - subjectID starts with '7' or '8' → 'UI'
            - otherwise → 'NE'

        Parameters:
            df (pd.DataFrame): Input DataFrame with a 'subjectID' column.
        
        Returns:
            pd.DataFrame: Updated DataFrame with 'study' and 'site' columns.
        """

        # Derive 'study' based on subjectID prefix
        df['study'] = df['subjectID'].astype(str).str.startswith('7').map({
            True: 'obs',
            False: 'int'
        })

        # Derive 'site' based on the first digit of subjectID
        df['site'] = df['subjectID'].astype(str).str[0].map(
            lambda x: 'UI' if x in ['7', '8'] else 'NE'
        )

        return df

    @staticmethod
    def compute_task_switch_costs(ats: pd.DataFrame) -> pd.DataFrame:
        """
        Compute task switch costs (single, repeat, switching) for each subject-session.
        Returns one row per (subjectID, session).
        """
        results = []
        # group by subject AND session
        for (subject, session), sub_df in ats.groupby(["subjectID","session"]):
            task_vers = sub_df["task_vers"].iloc[0] if "task_vers" in sub_df.columns else None

            # single cost
            ab = sub_df[sub_df["block_cond"].isin(["A","B"])]
            single = ab["response_time"].mean() if len(ab)>0 else np.nan

            # repeat/switching from C
            c = sub_df[sub_df["block_cond"]=="C"].sort_index()
            if len(c) < 2:
                repeat = switching = np.nan
            else:
                c = c.assign(prev_con_img=c["con_img"].shift())
                valid = c.iloc[1:]
                repeat    = valid.loc[ valid["con_img"]==valid["prev_con_img"], "response_time"].mean()
                switching = valid.loc[ valid["con_img"]!=valid["prev_con_img"], "response_time"].mean()

            results.append({
                "subject": subject,
                "session": session,
                "task_vers": task_vers,
                "single": single,
                "repeat": repeat,
                "switching": switching
            })

        return pd.DataFrame(results)

    @staticmethod
    def compute_wide_cost_stats_by_task_vers_session(
        df: pd.DataFrame,
        cost_cols: list = ["single", "repeat", "switching"],
        group_cols: list = ["task_vers", "session"]
    ) -> pd.DataFrame:
        """
        Computes the mean and 95% CI for each cost column, grouped by task version and session.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the grouping columns and cost metric columns.
        cost_cols : list of str
            Column names corresponding to each cost metric.
        group_cols : list of str
            Column names to group by (e.g., ["task_vers","session"]).

        Returns:
        --------
        pd.DataFrame
            A summary DataFrame with columns:
            - one column per name in group_cols
            - condition : the name of the cost column
            - mean      : the mean of that cost column
            - ci        : half-width of the 95% confidence interval
        """
        results = []
        grouped = df.groupby(group_cols)

        for group_vals, group_df in grouped:
            # ensure group_vals is a tuple
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)
            # map each grouping column to its value
            group_dict = dict(zip(group_cols, group_vals))

            for col in cost_cols:
                col_data = group_df[col].dropna()
                col_mean = col_data.mean()
                n = col_data.count()
                if n > 1:
                    std = col_data.std(ddof=1)
                else:
                    std = np.nan

                results.append({
                    **group_dict,
                    "condition": col,
                    "mean": col_mean,
                    "ci": std
                })

        return pd.DataFrame(results)

    @staticmethod
    def summarize_task(
        df: pd.DataFrame,
        filter_col: str = None,
        filter_val=None
    ) -> dict[str, pd.DataFrame]:
        """
        Summarize performance for a single task DataFrame by session and version,
        then return summary splits by study/site.

        df must already include 'study' and 'site' columns.
        """
        if filter_col:
            df = df[df[filter_col] == filter_val]

        # Ensure IDs are consistent
        group_keys = ['subject_id', 'session_number', 'task_vers']

        # Preserve unique study/site info per subject-session
        meta = df[group_keys + ['study', 'site']].drop_duplicates()

        counts = (
            df.groupby(group_keys, as_index=False)['correct']
            .sum()
            .rename(columns={'correct': 'n_correct'})
        )
        totals = (
            df.groupby(group_keys, as_index=False)['correct']
            .count()
            .rename(columns={'correct': 'n_total'})
        )

        summary = pd.merge(counts, totals, on=group_keys)
        summary = pd.merge(summary, meta, on=group_keys)
        summary['prop_correct'] = summary['n_correct'] / summary['n_total']
        summary = summary.rename(columns={'session_number': 'session'})
        return {
            'total': summary,
            'obs':   summary[summary['study'] == 'obs'],
            'int':   summary[summary['study'] == 'int'],
            'UI':    summary[summary['site']  == 'UI'],
            'NE':    summary[summary['site']  == 'NE'],
        }
    
    def load_and_tag(self, task_code, columns, filter_col=None):
        df = self.load_task_session_data(task_code, columns)
        df = self.add_study_and_site_columns(df)
