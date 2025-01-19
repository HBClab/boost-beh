import os
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import cprint

class SAVE_EVERYTHING:
    def __init__(self):
        self.datadir = './data'
        self.sessions = {}  # Dictionary to track session numbers for each subjectID

    def _get_folder(self, subjID):
        if 7000 <= int(subjID) < 8000:
            return 'obs', 'UI'
        elif 8000 <= int(subjID) < 9000:
            return 'int', 'UI'
        else:
            return 'int', 'NE'

    def save_dfs(self, categories, task):
        for subjectID, category, df in categories:
            cprint(f"saving {subjectID}...", "green")
            folder1, folder2 = self._get_folder(subjectID)
            outdir = os.path.join(self.datadir, folder1, folder2, str(subjectID), task, 'data')
            session = df['session_number'][2]
            os.makedirs(outdir, exist_ok=True)
            csv_path = os.path.join(outdir, f"{subjectID}_ses-{session}_cat-{category}.csv")
            df.to_csv(csv_path, index=False)

            # Update the sessions dictionary
            if subjectID not in self.sessions:
                self.sessions[subjectID] = set()
            self.sessions[subjectID].add(session)

    def save_plots(self, plots, task):
        # Validate 'plots' for NoneType objects
        if plots is None or any(item is None for item in plots):
            raise ValueError("The 'plots' list contains NoneType objects, which are not allowed.")

        for subjectID, plot_obj in plots:
            # Ensure subjectID and plot_obj are valid
            if subjectID is None or plot_obj is None:
                raise ValueError(f"Invalid data in plots: subjectID={subjectID}, plot_obj={plot_obj}")

            if subjectID not in self.sessions:
                raise ValueError(f"No session information found for subjectID {subjectID}.")

            folder1, folder2 = self._get_folder(subjectID)
            outdir = os.path.join(self.datadir, folder1, folder2, str(subjectID), task, 'plot')
            os.makedirs(outdir, exist_ok=True)

            for session in self.sessions[subjectID]:
                if isinstance(plot_obj, tuple):  # Handle multiple plots
                    for i, individual_plot in enumerate(plot_obj):
                        plot_path = os.path.join(outdir, f"{subjectID}_ses-{session}_plot{i+1}.png")
                        individual_plot.figure.savefig(plot_path)
                        plt.close(individual_plot.figure)
                else:  # Handle a single plot
                    plot_path = os.path.join(outdir, f"{subjectID}_ses-{session}.png")
                    plot_obj.figure.savefig(plot_path)
                    plt.close(plot_obj.figure)


"""

7000s- UI Observational
8000s- UI Intervention
9000s- NE Intervention

folder structure =
int -> UI/NE -> subID -> task -> data/plot
obs -> UI/NE -> subID -> task -> data/plot

"""
