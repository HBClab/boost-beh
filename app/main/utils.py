from flask import current_app, jsonify
import os
import pandas as pd

def construct_master_list(data_folder):
    """
    Constructs a master list of all tasks grouped by Subject ID at application startup.

    Output:
        Dictionary:
            subject_id: {
                site: str,
                project: str,  # 'int' or 'obs'
                tasks: {
                    task_name: {
                        date: str,
                        category: str,
                        png_paths: list
                    }
                }
            }
    """
    directories = ['int', 'obs']
    master_data = {}

    for directory in directories:
        dir_path = os.path.join(data_folder, directory)

        for site in os.listdir(dir_path):  # Iterate over site folders (e.g., UI, NE)
            site_path = os.path.join(dir_path, site)

            if not os.path.isdir(site_path):
                continue

            for subject_id in os.listdir(site_path):  # Iterate over subject folders (e.g., 8006, 9002)
                subject_path = os.path.join(site_path, subject_id)

                if not os.path.isdir(subject_path):
                    continue

                # Initialize subject entry if not already in master_data
                if subject_id not in master_data:
                    master_data[subject_id] = {
                        'site': site,
                        'project': directory,
                        'tasks': {}
                    }

                for task_name in os.listdir(subject_path):  # Iterate over task folders (e.g., AF, DSST)
                    task_path = os.path.join(subject_path, task_name)

                    if not os.path.isdir(task_path):
                        continue

                    plots_path = os.path.join(task_path, 'plot')
                    data_path = os.path.join(task_path, 'data')

                    # Initialize task entry if not already in tasks
                    if task_name not in master_data[subject_id]['tasks']:
                        master_data[subject_id]['tasks'][task_name] = {
                            'date': None,
                            'category': None,
                            'png_paths': [],
                            'session': None
                        }

                    # Extract date and category from CSV in data directory
                    csv_file = [
                        file for file in os.listdir(data_path)
                        if file.endswith('.csv')
                    ]
                    if csv_file:
                        csv_filename = csv_file[0]

                        # Load the CSV into a DataFrame
                        df = pd.read_csv(os.path.join(data_path, csv_filename))

                        # Validate and extract the 'Date' column
                        if 'datetime' in df.columns:
                            date_value = df['datetime'].iloc[0]  # Extract the first value in the 'Date' column
                        else:
                            date_value = None  # Set to None or handle it as needed

                        # Extract the category from the filename
                        category_value = csv_filename.split('_')[-1].replace('.csv', '').replace('cat-', '')
                        session_value = csv_filename.split('_')[-2].replace('ses-', '')

                        # Update master_data
                        master_data[subject_id]['tasks'][task_name]['date'] = date_value
                        master_data[subject_id]['tasks'][task_name]['category'] = category_value
                        master_data[subject_id]['tasks'][task_name]['session'] = session_value

                        # Remove the DataFrame from memory
                        del df

                    # Collect PNG file paths from plot directory
                    if os.path.exists(plots_path):
                        png_files = [
                            os.path.join(plots_path, png)
                            for png in os.listdir(plots_path)
                            if png.endswith('.png')
                        ]
                        master_data[subject_id]['tasks'][task_name]['png_paths'].extend(png_files)

    return master_data



def filter_master_list_by_key(master_list, key, value):
    """
    Filters the master list for the given key (site, task_name, category, or subject) and value.

    Args:
        master_list (dict): The original master list stored in app.config['MASTER_LIST'].
        key (str): The key to filter by ('site', 'task', 'category', or 'subject').
        value (str): The value to match for the specified key.

    Returns:
        dict: A new dictionary filtered based on the key and value.
    """
    filtered_dict = {}

    for subject_id, subject_data in master_list.items():
        # Filter by 'subject'
        if key == "subject" and subject_id == value:
            filtered_dict[subject_id] = subject_data
            continue

        # Filter by 'site'
        if key == "site" and subject_data.get("site") == value:
            filtered_dict[subject_id] = subject_data
            continue

        # Filter by 'task' or 'category'
        if key in ["task", "category"]:
            tasks = subject_data.get("tasks", {})
            filtered_tasks = {}

            # Filter tasks based on the key and value
            for task_name, task_data in tasks.items():
                if key == "task" and task_name == value:
                    filtered_tasks[task_name] = task_data
                elif key == "category" and task_data.get("category") == value:
                    filtered_tasks[task_name] = task_data

            # Only add subjects with matching tasks
            if filtered_tasks:
                filtered_dict[subject_id] = {
                    "site": subject_data.get("site"),
                    "tasks": filtered_tasks
                }

    return filtered_dict
