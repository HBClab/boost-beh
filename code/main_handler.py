from data_processing.pull_handler import Pull
from data_processing.cc_qc import CCqC
from data_processing.mem_qc import MEM_QC
from data_processing.ps_qc import PS_QC
from data_processing.utils import CONVERT_TO_CSV
from data_processing.wl_qc import WL_QC
from data_processing.plot_utils import CC_PLOTS, MEM_PLOTS, PS_PLOTS
from data_processing.save_utils import SAVE_EVERYTHING
import pandas as pd


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


    def pull(self, task):
        pull_instance = Pull(
            self.IDs[task],
            tease="WEEEEEEEEEEEEEE",
            token="jap_5ThOJ14yf7z1EPEUpAoZYMWoETZcmJk305719",
            taskName=task,
            proxy=False
        )

        txt_dfs = pull_instance.load(days_ago=70)
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
        print(task)
        if task in ['AF', 'NF', 'NNB', 'VNB']:
            print("running AF version")
            qc_instance = CCqC(task,
                               MAXRT=1800,
                               RT_COLUMN_NAME='response_time',
                               ACC_COLUMN_NAME='correct',
                               CORRECT_SYMBOL=1,
                               INCORRECT_SYMBOL=0,
                               COND_COLUMN_NAME='condition')
            for df in dfs:
                subject = df['subject_id'][1]
                print(f"qcing {subject}")
                category = qc_instance.cc_qc(df, threshold=0.5)
                if task in ['AF', 'NF']:
                    plot = plot_instance.af_nf_plot(df)
                elif task in ['NNB', 'VNB']:
                    plot = plot_instance.nnb_vnb_plot(df)
                print(f"Category = {category}")
                categories.append([subject, category, df])
                plots.append([subject, plot])

        else:
            qc_instance = CCqC(task,
                               MAXRT=1800,
                               RT_COLUMN_NAME='response_time',
                               ACC_COLUMN_NAME='correct',
                               CORRECT_SYMBOL=1,
                               INCORRECT_SYMBOL=0,
                               COND_COLUMN_NAME='block_cond')
            for df in dfs:
                subject = df['subject_id'][1]
                print(f"qcing {subject}")
                category = qc_instance.cc_qc(df, threshold=0.5, TS=True)
                plot = plot_instance.ats_nts_plot(df)
                print(f"Category = {category}")
                categories.append([subject, category, df])
                plots.append([subject, plot])
        save_instance = SAVE_EVERYTHING()
        save_instance.save_dfs(categories=categories,
                                task=task)
        save_instance.save_plots(plots=plots,
                                     task=task)
        return categories, plots


    def qc_ps_dfs(self, dfs, task):
        categories, plots = [], []
        plot_instance = PS_PLOTS()
        print(task)
        if task in ['PC', 'LC']:
            ps_instance = PS_QC('response_time', 'correct', 1, 0, 'block_c', 30000)
            for df in dfs:
                subject = df['subject_id'][1]
                print(f"qcing {subject}")
                category = ps_instance.ps_qc(df, threshold=0.6,)
                if task == 'PC':
                    plot = plot_instance.lc_plot(df)
                elif task == 'LC':
                    plot = plot_instance.lc_plot(df)
                print(f"Category = {category}")
                categories.append([subject, category, df])
                plots.append([subject, plot])

        else:
            ps_instance = PS_QC('block_dur', 'correct', 1, 0, 'block_c', 125)
            for df in dfs:
                subject = df['subject_id'][1]
                print(f"qcing {subject}")
                category = ps_instance.ps_qc(df, threshold=0.6, DSST=True)
                plot = plot_instance.dsst_plot(df)
                print(f"Category = {category}")
                categories.append([subject, category, df])
                plots.append([subject, plot])

        save_instance = SAVE_EVERYTHING()
        save_instance.save_dfs(categories=categories,
                                task=task)
        save_instance.save_plots(plots=plots,
                                    task=task)

        return categories, plots


    def qc_mem_dfs(self, dfs, task):
        plot_instance = MEM_PLOTS()
        categories, plots = [], []
        print(task)
        if task in ['FN']:
            mem_instance = MEM_QC('response_time', 'correct', 1, 0, 'block_c', 4000)
            for df in dfs:
                subject = df['subject_id'][1]
                print(f"qcing {subject}")
                category = mem_instance.fn_sm_qc(df, threshold=0.5)
                plot = plot_instance.fn_plot(df)
                print(f"Category = {category}")
                categories.append([subject, category, df])
                plots.append([subject, plot])
        elif task in ['SM']:
            mem_instance = MEM_QC('response_time', 'correct', 1, 0, 'block_c', 2000)
            for df in dfs:
                subject = df['subject_id'][1]
                print(f"qcing {subject}")
                category = mem_instance.fn_sm_qc(df, threshold=0.5)
                plot = plot_instance.sm_plot(df)
                print(f"Category = {category}")
                categories.append([subject, category, df])
                plots.append([subject, plot])
        save_instance = SAVE_EVERYTHING()
        save_instance.save_dfs(categories=categories,
                                task=task)
        save_instance.save_plots(plots=plots,
                                    task=task)
        return categories, plots

    def qc_wl_dfs(self, dfs, task):
        categories, plots = [], []
        plot_instance = MEM_PLOTS()
        print(task)
        if task in ['WL']:
            for df in dfs:
                subject = df['subject_id'][1]
                version = df['task_vers'][1]
                wl_instance = WL_QC()
                print(f"qcing {subject}")
                df_all, category = wl_instance.wl_qc(df, version)
                plot = plot_instance.wl_plot(df_all)
                print(f"Category = {category}")
                categories.append([subject, category, df])
                plots.append([subject, plot])
        elif task in ['DWL']:
            for df in dfs:
                subject = df['subject_id'][1]
                version = df['task_vers'][1]
                dwl_instance = WL_QC()
                print(f"qcing {subject}")
                df_all, category = dwl_instance.dwl_qc(df, version)
                plot = plot_instance.dwl_plot(df_all)
                print(f"Category = {category}")
                categories.append([subject, category, df])
                plots.append([subject, plot])
        save_instance = SAVE_EVERYTHING()
        save_instance.save_dfs(categories=categories,
                                task=task)
        save_instance.save_plots(plots=plots,
                                    task=task)
        return categories, plots


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


