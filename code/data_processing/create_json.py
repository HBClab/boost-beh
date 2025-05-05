import os
import pandas as pd
import json

def create_json(data_folder, out_file='data.json'):
    """
    Constructs a master list of all tasks grouped by Subject ID at application startup,
    with each session saved as a separate task entry (keyed as TASKNAME_ses-SESSION).
    """
    directories = ['int', 'obs']
    master_data = {}

    for directory in directories:
        dir_path = os.path.join(data_folder, directory)

        for site in os.listdir(dir_path):
            site_path = os.path.join(dir_path, site)
            if not os.path.isdir(site_path): continue

            for subject_id in os.listdir(site_path):
                subject_path = os.path.join(site_path, subject_id)
                if not os.path.isdir(subject_path): continue

                master_data.setdefault(subject_id, {
                    'site': site,
                    'project': directory,
                    'tasks': {}
                })

                for task_name in os.listdir(subject_path):
                    task_path = os.path.join(subject_path, task_name)
                    if not os.path.isdir(task_path): continue

                    plots_path = os.path.join(task_path, 'plot')
                    data_path  = os.path.join(task_path, 'data')

                    # find all CSVs (one per session)
                    if not os.path.isdir(data_path):
                        continue
                    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

                    for csv_filename in csv_files:
                        # parse session, category, and date
                        parts = csv_filename.split('_')
                        session_value  = parts[-2].replace('ses-', '')
                        category_value = parts[-1].replace('.csv', '').replace('cat-', '')

                        df = pd.read_csv(os.path.join(data_path, csv_filename))
                        date_value = df['datetime'].iloc[0] if 'datetime' in df.columns else None
                        del df

                        # build a unique task key for this session
                        task_key = f"{task_name}_ses-{session_value}"

                        # collect only PNGs for this session
                        png_list = []
                        if os.path.isdir(plots_path):
                            for png in os.listdir(plots_path):
                                if png.endswith('.png') and f"ses-{session_value}" in png:
                                    png_list.append(os.path.join(plots_path, png))

                        # assign
                        master_data[subject_id]['tasks'][task_key] = {
                            'date': date_value,
                            'category': category_value,
                            'png_paths': sorted(png_list),
                            'session': session_value
                        }

    # write out
    with open(out_file, 'w') as f:
        json.dump(master_data, f, indent=2)

    return master_data
