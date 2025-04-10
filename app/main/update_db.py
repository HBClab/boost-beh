import os
import logging
import psycopg
import pandas as pd
from datetime import datetime
import re

class DatabaseUtils:
    def __init__(self, db_name, user, password, data_folder):
        """
        Initializes the DatabaseUtils class with database credentials and a data folder path.
        
        :param db_name: Name of the PostgreSQL database.
        :param user: Username for the database.
        :param password: Password for the database.
        :param data_folder: Path to the directory containing study data.
        """
        self.data_folder = data_folder
        self.db_name = db_name
        self.user = user
        self.password = password

    def connect(self, host="localhost", port=5432):
        """
        Connects to the PostgreSQL database using provided credentials.
        """
        self.connection = psycopg.connect(
            dbname=self.db_name,
            host=host,
            port=port
        )
        return self.connection

    def update_database(self):
        """
        Walks through the directory structure organized as:
            data_folder -> study -> site -> subject -> task
        Each task folder is expected to contain:
            - A "data" folder with CSV files (naming: <subject>_ses-<session>_cat-<category>.csv)
            - A "plot" folder with PNG files (naming: <subject>_ses-<session>_plot*.png)
        A unique task record is created per task folder (per subject). For each CSV file in the
        task's data folder, a session record is inserted that references that unique task record.
        """
        logging.info("Starting database update.")

        # Loop through each study folder
        for study_name in sorted(os.listdir(self.data_folder)):
            study_path = os.path.join(self.data_folder, study_name)
            if not os.path.isdir(study_path):
                logging.warning(f"Skipping non-directory: {study_path}")
                continue

            study_id = self._add_or_get_id("study", {"name": study_name})
            logging.info(f"Processing study '{study_name}' (ID: {study_id}).")

            # Loop through each site folder within the study
            for site_name in sorted(os.listdir(study_path)):
                site_path = os.path.join(study_path, site_name)
                if not os.path.isdir(site_path):
                    logging.warning(f"Skipping non-directory: {site_path}")
                    continue

                site_id = self._add_or_get_id("site", {"name": site_name, "study_id": study_id})
                logging.info(f"Processing site '{site_name}' (ID: {site_id}).")

                # Loop through each subject folder within the site
                for subject_name in sorted(os.listdir(site_path)):
                    subject_path = os.path.join(site_path, subject_name)
                    if not os.path.isdir(subject_path):
                        logging.warning(f"Skipping non-directory: {subject_path}")
                        continue

                    # Format subject name if numeric (pad to 4 digits)
                    try:
                        int(subject_name)
                        subject_name = subject_name.zfill(4)
                    except ValueError:
                        logging.warning(f"Subject name {subject_name} is not numeric; saving as-is.")

                    subject_id = self._add_or_get_id("subject", {"name": subject_name, "site_id": site_id})
                    logging.info(f"Processing subject '{subject_name}' (ID: {subject_id}).")

                    # Loop through each task folder within the subject
                    for task_name in sorted(os.listdir(subject_path)):
                        task_path = os.path.join(subject_path, task_name)
                        if not os.path.isdir(task_path):
                            logging.warning(f"Skipping non-directory: {task_path}")
                            continue

                        # Create or retrieve a unique task record for the subject.
                        task_id = self._add_or_get_id("task", {"name": task_name, "subject_id": subject_id})
                        logging.info(f"Processing task '{task_name}' (ID: {task_id}).")

                        # Process sessions within this task folder (each session from a CSV file)
                        data_folder_path = os.path.join(task_path, "data")
                        if os.path.exists(data_folder_path):
                            for file in sorted(os.listdir(data_folder_path)):
                                if file.endswith(".csv"):
                                    logging.debug(f"Processing CSV file: {file}")
                                    try:
                                        # Expected naming: <subject>_ses-<session>_cat-<category>.csv
                                        parts = file.split("_")
                                        if len(parts) < 3:
                                            raise ValueError(f"Unexpected file format: {file}")

                                        # Extract session identifier (e.g., "ses-1")
                                        session_identifier = parts[1].strip()
                                        # Extract category (e.g., from "cat-1.csv")
                                        cat_part = parts[2]
                                        category = int(cat_part.split("-")[1].split(".")[0])

                                        # Build the CSV relative path (starting with "./data")
                                        csv_relative_path = os.path.join(".", "data", file)
                                        csv_full_path = os.path.join(data_folder_path, file)
                                        date = None
                                        try:
                                            df = pd.read_csv(csv_full_path)
                                            if "datetime" in df.columns and not df.empty:
                                                raw_date = str(df["datetime"].iloc[0])
                                                date = self._clean_date(raw_date)
                                        except Exception as e:
                                            logging.error(f"Error reading CSV {csv_full_path}: {e}")

                                        # Process the "plot" folder for matching PNG files
                                        plot_folder_path = os.path.join(task_path, "plot")
                                        plot_relative_paths = []
                                        if os.path.exists(plot_folder_path):
                                            for pfile in sorted(os.listdir(plot_folder_path)):
                                                if pfile.endswith(".png") and session_identifier in pfile:
                                                    # Build relative plot path (starting with "./plot")
                                                    plot_relative = os.path.join(".", "plot", pfile)
                                                    plot_relative_paths.append(plot_relative)

                                        # Insert session record that references the unique task
                                        with self.connection.cursor() as cursor:
                                            cursor.execute(
                                                """
                                                INSERT INTO session (session_name, category, csv_path, task_id, date, plot_paths)
                                                VALUES (%s, %s, %s, %s, %s, %s)
                                                ON CONFLICT DO NOTHING;
                                                """,
                                                (session_identifier, category, csv_relative_path, task_id, date, plot_relative_paths)
                                            )
                                        logging.debug(f"Inserted session '{session_identifier}' with CSV {csv_relative_path} and plots {plot_relative_paths}.")
                                    except Exception as e:
                                        logging.error(f"Error processing file {file}: {e}")
                        else:
                            logging.warning(f"No 'data' folder found in task directory: {task_path}")

            # Commit changes after processing each study
            self.connection.commit()
            logging.info(f"Committed changes for study '{study_name}'.")

        logging.info("Database update complete.")

    def _add_or_get_id(self, table, values):
        """
        Inserts a new record into the specified table or returns the existing record's ID.
        
        :param table: The name of the table.
        :param values: A dictionary of column names and values.
        :return: The ID of the record.
        """
        placeholders = " AND ".join([f"{key} = %s" for key in values.keys()])
        columns = ", ".join(values.keys())
        values_list = list(values.values())

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
            # If no new row was created, fetch the existing ID.
            select_query = f"SELECT id FROM {table} WHERE {placeholders};"
            cursor.execute(select_query, values_list)
            return int(cursor.fetchone()[0])

    def _clean_date(self, raw_date):
        """
        Cleans and converts a raw date string into SQL-compatible format.
        
        :param raw_date: Date string extracted from a CSV.
        :return: Formatted date string or None if parsing fails.
        """
        try:
            # Remove timezone information enclosed in parentheses, if any.
            cleaned_raw_date = re.sub(r"\s\(.*?\)", "", raw_date)
            # Parse date assuming format like "Wed Mar 03 2021 12:34:56 GMT+0000"
            clean_date = datetime.strptime(cleaned_raw_date, "%a %b %d %Y %H:%M:%S %Z%z")
            return clean_date.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            logging.error(f"Error parsing date '{raw_date}': {e}")
            return None

def create_init_db():
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Define PostgreSQL credentials and data folder path
    DB_NAME = "boost-beh-test"
    DB_USER = "your_username"
    DB_PASSWORD = "your_password"
    DATA_FOLDER = "../../data"  # Change this to the actual data folder path

    # Initialize the database utility
    db_utils = DatabaseUtils(DB_NAME, DB_USER, DB_PASSWORD, DATA_FOLDER)

    # Connect to the database
    db_utils.connect()

    # Run the update script
    db_utils.update_database()

    # Close the database connection when done
    db_utils.connection.close()


create_init_db()
