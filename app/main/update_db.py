import os
import logging
import psycopg
import pandas as pd
from datetime import datetime

class DatabaseUtils:
    def __init__(self, connection, data_folder):
        self.connection = connection
        self.data_folder = data_folder

    def update_database(self):
        logging.info("Starting database update.")

        for study_name in os.listdir(self.data_folder):
            study_path = os.path.join(self.data_folder, study_name)
            if not os.path.isdir(study_path):
                logging.warning(f"Skipping non-directory: {study_path}")
                continue

            study_id = self._add_or_get_id("study", {"name": study_name})

            for site_name in os.listdir(study_path):
                site_path = os.path.join(study_path, site_name)
                if not os.path.isdir(site_path):
                    logging.warning(f"Skipping non-directory: {site_path}")
                    continue

                site_id = self._add_or_get_id("site", {"name": site_name, "study_id": study_id})

                for subject_name in os.listdir(site_path):
                    subject_path = os.path.join(site_path, subject_name)
                    if not os.path.isdir(subject_path):
                        logging.warning(f"Skipping non-directory: {subject_path}")
                        continue

                    subject_id = self._add_or_get_id("subject", {"name": subject_name, "site_id": site_id})

                    for task_name in os.listdir(subject_path):
                        task_path = os.path.join(subject_path, task_name)
                        if not os.path.isdir(task_path):
                            logging.warning(f"Skipping non-directory: {task_path}")
                            continue

                        task_id = self._add_or_get_id("task", {"name": task_name, "subject_id": subject_id})

                        self._process_data_folder(task_path, task_id)
                        self._process_plot_folder(task_path, task_id)

            self.connection.commit()
            logging.info("Database committed.")

        logging.info("Database update complete.")

    def _add_or_get_id(self, table, values):
        placeholders = ', '.join([f"{key} = %s" for key in values.keys()])
        columns = ', '.join(values.keys())
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

            select_query = f"SELECT id FROM {table} WHERE {placeholders};"
            cursor.execute(select_query, values_list)
            return int(cursor.fetchone()[0])

    def _process_data_folder(self, task_path, task_id):
        data_folder_path = os.path.join(task_path, "data")
        if os.path.exists(data_folder_path):
            for file in os.listdir(data_folder_path):
                if file.endswith(".csv"):
                    logging.debug(f"Processing file: {file}")
                    try:
                        parts = file.split("_")
                        if len(parts) < 3:
                            raise ValueError(f"Unexpected file format: {file}")

                        session_name = parts[1].split("-")[1]  # Ensure split works correctly
                        category = int(parts[2].split("-")[1].split(".")[0])
                        csv_path = os.path.join(data_folder_path, file)

                        # Extract and clean date from CSV if column 'datetime' exists
                        date = None
                        df = pd.read_csv(csv_path)
                        if 'datetime' in df.columns:
                            raw_date = str(df['datetime'].iloc[0])
                            date = self._clean_date(raw_date)
                        del df

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
        import re
        """Converts raw date strings into a standardized format."""
        try:
            # Remove timezone information in parentheses, if any
            cleaned_raw_date = re.sub(r"\s\(.*?\)", "", raw_date)
            # Parse the cleaned date string
            clean_date = datetime.strptime(cleaned_raw_date, "%a %b %d %Y %H:%M:%S %Z%z")
            # Standardize to SQL-compatible format
            return clean_date.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            logging.error(f"Error parsing date: {raw_date} - {e}")
            return None
