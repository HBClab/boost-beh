import os
import psycopg
from psycopg import sql
import logging
from main.update_db import DatabaseUtils

# Database connection setup
def connect_to_db(db_name, user, password, host="localhost", port=5432):
    return psycopg.connect(dbname=db_name, user=user, password=password, host=host, port=port)

# Initialize database schema
def initialize_schema(connection):
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                -- Drop existing tables in reverse dependency order.
                DROP TABLE IF EXISTS session CASCADE;
                DROP TABLE IF EXISTS task CASCADE;
                DROP TABLE IF EXISTS subject CASCADE;
                DROP TABLE IF EXISTS site CASCADE;
                DROP TABLE IF EXISTS study CASCADE;

                -- Create table "study"
                CREATE TABLE study (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE
                );

                -- Create table "site"
                CREATE TABLE site (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    study_id INTEGER NOT NULL,
                    UNIQUE (name, study_id),
                    FOREIGN KEY (study_id) REFERENCES study(id) ON DELETE CASCADE
                );

                -- Create table "subject"
                CREATE TABLE subject (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    site_id INTEGER NOT NULL,
                    UNIQUE (name, site_id),
                    FOREIGN KEY (site_id) REFERENCES site(id) ON DELETE CASCADE
                );

                -- Create table "task"
                CREATE TABLE task (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    subject_id INTEGER NOT NULL,
                    UNIQUE (name, subject_id),
                    FOREIGN KEY (subject_id) REFERENCES subject(id) ON DELETE CASCADE
                );

                -- Create table "session"
                CREATE TABLE session (
                    id SERIAL PRIMARY KEY,
                    session_name TEXT NOT NULL,
                    category INTEGER NOT NULL,
                    csv_path TEXT NOT NULL,
                    task_id INTEGER NOT NULL,
                    date TIMESTAMP,
                    plot_paths TEXT[],
                    FOREIGN KEY (task_id) REFERENCES task(id) ON DELETE CASCADE,
                    UNIQUE (session_name, category, csv_path, task_id)
                );
            """)
            connection.commit()
    except Exception as e:
        logging.error(f"Error initializing schema: {e}")
        connection.rollback()

    finally:
        if connection:
            connection.close()

# Populate the database from the folder structure
def populate_database(connection, data_folder):
    for study_name in os.listdir(data_folder):
        study_path = os.path.join(data_folder, study_name)
        if not os.path.isdir(study_path):
            continue

        with connection.cursor() as cursor:
            cursor.execute("INSERT INTO study (name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING id;", (study_name,))
            study_id = cursor.fetchone() or (cursor.execute("SELECT id FROM study WHERE name = %s;", (study_name,)), cursor.fetchone()[0])

        for site_name in os.listdir(study_path):
            site_path = os.path.join(study_path, site_name)
            if not os.path.isdir(site_path):
                continue

            with connection.cursor() as cursor:
                cursor.execute("INSERT INTO site (name, study_id) VALUES (%s, %s) ON CONFLICT DO NOTHING RETURNING id;", (site_name, study_id))
                site_id = cursor.fetchone() or (cursor.execute("SELECT id FROM site WHERE name = %s AND study_id = %s;", (site_name, study_id)), cursor.fetchone()[0])

            for subject_name in os.listdir(site_path):
                subject_path = os.path.join(site_path, subject_name)
                if not os.path.isdir(subject_path):
                    continue

                with connection.cursor() as cursor:
                    cursor.execute("INSERT INTO subject (name, site_id) VALUES (%s, %s) ON CONFLICT DO NOTHING RETURNING id;", (subject_name, site_id))
                    subject_id = cursor.fetchone() or (cursor.execute("SELECT id FROM subject WHERE name = %s AND site_id = %s;", (subject_name, site_id)), cursor.fetchone()[0])

                for task_name in os.listdir(subject_path):
                    task_path = os.path.join(subject_path, task_name)
                    if not os.path.isdir(task_path):
                        continue

                    with connection.cursor() as cursor:
                        cursor.execute("INSERT INTO task (name, subject_id) VALUES (%s, %s) ON CONFLICT DO NOTHING RETURNING id;", (task_name, subject_id))
                        task_id = cursor.fetchone() or (cursor.execute("SELECT id FROM task WHERE name = %s AND subject_id = %s;", (task_name, subject_id)), cursor.fetchone()[0])

                    for folder in ["data", "plot"]:
                        folder_path = os.path.join(task_path, folder)
                        if not os.path.exists(folder_path):
                            continue

                        if folder == "data":
                            for file in os.listdir(folder_path):
                                if file.endswith(".csv"):
                                    parts = file.split("_")
                                    session_name = parts[1].split("-")[1]
                                    category = int(parts[2].split("-")[1].split(".")[0])

                                    with connection.cursor() as cursor:
                                        cursor.execute("""
                                        INSERT INTO session (session_name, category, csv_path, task_id)
                                        VALUES (%s, %s, %s, %s)
                                        ON CONFLICT DO NOTHING;
                                        """, (session_name, category, os.path.join(folder_path, file), task_id))

                        elif folder == "plot":
                            plots = []
                            for file in os.listdir(folder_path):
                                if file.endswith(".png"):
                                    plots.append(os.path.join(folder_path, file))

                            with connection.cursor() as cursor:
                                cursor.execute("""
                                UPDATE session
                                SET plot_paths = %s
                                WHERE task_id = %s;
                                """, (plots, task_id))
        connection.commit()
import psycopg
from psycopg import sql


# Main entry point
if __name__ == "__main__":
    db_name = "boost-beh"
    user = "zakg04"
    password = "*mIloisfAT23*123*"
    data_folder = "../data"
    connection = connect_to_db(db_name, user, password)
    try:
        initialize_schema(connection)
    finally:
        connection.close()
'''
    util_instance = DatabaseUtils(connection, data_folder)
    util_instance.update_database()

'''
