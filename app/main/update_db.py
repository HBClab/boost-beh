import os
import logging
import psycopg  # PostgreSQL database adapter
import pandas as pd
from datetime import datetime

class DatabaseUtils:
    def __init__(self, connection, data_folder):
        """
        Initializes the DatabaseUtils class with a database connection and a data folder path.
        
        :param connection: PostgreSQL database connection object.
        :param data_folder: Path to the directory containing study data.
        """
        self.connection = connection
        self.data_folder = data_folder

    def update_database(self):
        """
        Iterates through the directory structure and updates the database with study, site, subject, 
        task, and session information. Commits changes at the end of processing each study.
        """
        logging.info("Starting database update.")

        # Loop through each study folder in the data directory
        for study_name in os.listdir(self.data_folder):
            study_path = os.path.join(self.data_folder, study_name)
            if not os.path.isdir(study_path):
                logging.warning(f"Skipping non-directory: {study_path}")
                continue

            # Add or retrieve the study ID from the database
            study_id = self._add_or_get_id("study", {"name": study_name})

            # Loop through each site folder within the study
            for site_name in os.listdir(study_path):
                site_path = os.path.join(study_path, site_name)
                if not os.path.isdir(site_path):
                    logging.warning(f"Skipping non-directory: {site_path}")
                    continue

                # Add or retrieve the site ID from the database
                site_id = self._add_or_get_id("site", {"name": site_name, "study_id": study_id})

                # Loop through each subject folder within the site
                for subject_name in os.listdir(site_path):
                    subject_path = os.path.join(site_path, subject_name)
                    if not os.path.isdir(subject_path):
                        logging.warning(f"Skipping non-directory: {subject_path}")
                        continue

                    # Since subject names are always four-digit numbers, format them accordingly.
                    try:
                        # Ensure the subject folder name is numeric and pad with leading zeros if necessary.
                        int(subject_name)
                        subject_name = subject_name.zfill(4)
                    except ValueError:
                        logging.warning(f"Subject name {subject_name} is not numeric; saving as-is.")

                    # Add or retrieve the subject ID from the database
                    subject_id = self._add_or_get_id("subject", {"name": subject_name, "site_id": site_id})

                    # Loop through each task folder within the subject
                    for task_name in os.listdir(subject_path):
                        task_path = os.path.join(subject_path, task_name)
                        if not os.path.isdir(task_path):
                            logging.warning(f"Skipping non-directory: {task_path}")
                            continue

                        # Add or retrieve the task ID from the database
                        task_id = self._add_or_get_id("task", {"name": task_name, "subject_id": subject_id})

                        # Process data files within the task folder
                        self._process_data_folder(task_path, task_id)
                        # Process plot images within the task folder
                        self._process_plot_folder(task_path, task_id)

            # Commit all changes for the current study
            self.connection.commit()
            logging.info("Database committed.")

        logging.info("Database update complete.")

    def _add_or_get_id(self, table, values):
        """
        Adds a new entry to the specified table or retrieves the existing entry's ID.

        :param table: The name of the database table.
        :param values: A dictionary containing column names and their values.
        :return: The ID of the existing or newly inserted row.
        """
        # Build a WHERE clause for checking existing rows
        placeholders = ' AND '.join([f"{key} = %s" for key in values.keys()])
        columns = ', '.join(values.keys())
        values_list = list(values.values())

        # SQL query to insert a new record, avoiding conflicts
        query = f"""
            INSERT INTO {table} ({columns}) 
            VALUES ({', '.join(['%s'] * len(values))}) 
            ON CONFLICT DO NOTHING RETURNING id;
        """

        with self.connection.cursor() as cursor:
            cursor.execute(query, values_list)
            result = cursor.fetchone()
            if result:
                return int(result[0])

            # If no ID is returned, retrieve the existing record's ID.
            select_query = f"SELECT id FROM {table} WHERE {placeholders};"
            cursor.execute(select_query, values_list)
            return int(cursor.fetchone()[0])

    def _process_data_folder(self, task_path, task_id):
        """
        Processes CSV files in the "data" folder within a task directory and inserts session records into the database.

        :param task_path: Path to the task directory.
        :param task_id: ID of the corresponding task in the database.
        """
        data_folder_path = os.path.join(task_path, "data")
        if os.path.exists(data_folder_path):
            for file in os.listdir(data_folder_path):
                if file.endswith(".csv"):
                    logging.debug(f"Processing file: {file}")
                    try:
                        # Extract session and category information from the filename
                        parts = file.split("_")
                        if len(parts) < 3:
                            raise ValueError(f"Unexpected file format: {file}")

                        session_name = parts[1].split("-")[1]  # Extract session name
                        category = int(parts[2].split("-")[1].split(".")[0])  # Extract category
                        csv_path = os.path.join(data_folder_path, file)

                        # Extract and clean the date from the CSV if it contains a 'datetime' column
                        date = None
                        df = pd.read_csv(csv_path)
                        if 'datetime' in df.columns:
                            raw_date = str(df['datetime'].iloc[0])
                            date = self._clean_date(raw_date)
                        del df  # Free up memory

                        # Insert session data into the database
                        with self.connection.cursor() as cursor:
                            cursor.execute(
                                """
                                INSERT INTO session (session_name, category, csv_path, task_id, date)
                                VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT DO NOTHING;
                                """,
                                (session_name, category, csv_path, task_id, date)
                            )
                        logging.debug(f"Session added: {session_name}, Category: {category}, Path: {csv_path}, Date: {date}")
                    except (IndexError, ValueError, pd.errors.EmptyDataError) as e:
                        logging.error(f"Error processing file {file}: {e}")

    def _process_plot_folder(self, task_path, task_id):
        """
        Processes PNG image files in the "plot" folder and updates the session record with plot file paths.

        :param task_path: Path to the task directory.
        :param task_id: ID of the corresponding task in the database.
        """
        plot_folder_path = os.path.join(task_path, "plot")
        if os.path.exists(plot_folder_path):
            plots = [os.path.join(plot_folder_path, f) for f in os.listdir(plot_folder_path) if f.endswith(".png")]
            if plots:
                with self.connection.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE session
                        SET plot_paths = %s
                        WHERE task_id = %s;
                        """,
                        (plots, task_id)
                    )
                logging.debug(f"Plots updated for task {task_id}: {plots}")

    def _clean_date(self, raw_date):
        """
        Converts a raw date string into a standardized format.

        :param raw_date: Date string extracted from a CSV file.
        :return: Standardized date string or None if parsing fails.
        """
        import re
        try:
            # Remove timezone information enclosed in parentheses
            cleaned_raw_date = re.sub(r"\s\(.*?\)", "", raw_date)
            # Parse the cleaned date string into a datetime object
            clean_date = datetime.strptime(cleaned_raw_date, "%a %b %d %Y %H:%M:%S %Z%z")
            # Convert to SQL-compatible format
            return clean_date.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            logging.error(f"Error parsing date: {raw_date} - {e}")
            return None
