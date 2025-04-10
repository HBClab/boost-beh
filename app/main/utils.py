from flask import current_app, jsonify
import os
import pandas as pd
import logging
import re

class DatabaseReading:
    def __init__(self, pool):
        """
        Initializes the DatabaseReading class with a connection pool.
        
        :param pool: A connection pool instance (should have a method to acquire a connection).
        """
        self.pool = pool

    def get_results(self, table, columns="*", joins=None, conditions=None, order_by=None):
        """
        Retrieves rows from the specified table with optional joins, conditions, and ordering.

        :param table: The main table name (e.g., "site").
        :param columns: The columns to select (default is "*").
        :param joins: A list of JOIN clauses as strings (e.g., ["INNER JOIN orders ON site.id = orders.site_id"]).
        :param conditions: A string containing the WHERE clause conditions (e.g., "site.status = 'active'").
        :param order_by: A string specifying the ORDER BY clause (e.g., "orders.date DESC").
        :return: A list of rows from the table.
        """
        query = f"SELECT {columns} FROM {table}"

        if joins:
            query += " " + " ".join(joins)

        if conditions:
            query += f" WHERE {conditions}"

        if order_by:
            query += f" ORDER BY {order_by}"

        query += ";"  # end properly

        results = []
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                results = cur.fetchall()

        return results

    def create_data_dict(self):
        """
        Creates a nested dictionary organized by subject_id, with each subject containing its site,
        project (study name), and tasks. Each task dictionary contains a list of session details
        (date, category, png_paths, and session number).
        
        Dictionary structure:
            {
                subject_id: {
                    "site": <site_name>,
                    "project": <project_name>,  # e.g. 'int' or 'obs'
                    "tasks": {
                        task_name: [
                            {
                                "session": <session_number as int>,
                                "date": <date as string>,
                                "category": <category as string>,
                                "png_paths": <list of png paths>
                            },
                            ...
                        ]
                    }
                },
                ...
            }
        
        :return: The constructed dictionary.
        """
        data_dict = {}
        query = """
            SELECT subject.id AS subject_id,
                   site.name AS site,
                   study.name AS project,
                   task.name AS task_name,
                   session.session_name AS session_name, -- Stores 'ses-#'
                   session.date AS date,
                   session.category AS category,
                   session.plot_paths AS png_paths
            FROM subject
            JOIN site ON subject.site_id = site.id
            JOIN study ON site.study_id = study.id
            JOIN task ON task.subject_id = subject.id
            JOIN session ON session.task_id = task.id
            ORDER BY subject.id, task.name, session.date;
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
                for row in rows:
                    subject_id, site, project, task_name, session_name, date, category, png_paths = row
                    
                    # Extract session number (integer) from 'ses-#' format
                    session_number = self._extract_session_number(session_name)

                    # Ensure subject entry exists
                    if subject_id not in data_dict:
                        data_dict[subject_id] = {
                            "site": site,
                            "project": project,
                            "tasks": {}
                        }
                    
                    # Ensure task entry exists for this subject
                    if task_name not in data_dict[subject_id]["tasks"]:
                        data_dict[subject_id]["tasks"][task_name] = []

                    # Append session details to the task list
                    data_dict[subject_id]["tasks"][task_name].append({
                        "session": session_number,
                        "date": date if date is not None else "",
                        "category": str(category) if category is not None else "",
                        "png_paths": png_paths if png_paths is not None else []
                    })
        return data_dict

    def _extract_session_number(self, session_name):
        """
        Extracts the session number from a session_name string in 'ses-#' format.
        
        :param session_name: The session string (e.g., "ses-1")
        :return: The session number as an integer, or None if not found.
        """
        match = re.search(r"ses-(\d+)", session_name)
        return int(match.group(1)) if match else None

        
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
