import os
import psycopg
from psycopg import sql

# Database connection setup
def connect_to_db(db_name, user, password, host="localhost", port=5432):
    return psycopg.connect(dbname=db_name, user=user, password=password, host=host, port=port)

# Initialize database schema
def initialize_schema(connection):
    with connection.cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS study (
            id SERIAL PRIMARY KEY,
            name VARCHAR(50) UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS site (
            id SERIAL PRIMARY KEY,
            name VARCHAR(50) NOT NULL,
            study_id INT REFERENCES study(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS subject (
            id SERIAL PRIMARY KEY,
            name VARCHAR(50) NOT NULL,
            site_id INT REFERENCES site(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS task (
            id SERIAL PRIMARY KEY,
            name VARCHAR(50) NOT NULL,
            subject_id INT REFERENCES subject(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS session (
            id SERIAL PRIMARY KEY,
            session_name VARCHAR(50) NOT NULL,
            category INT NOT NULL,
            csv_path TEXT,
            plot_paths TEXT[],
            task_id INT REFERENCES task(id) ON DELETE CASCADE
        );
        """)
        connection.commit()

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

def initialize_postgres_db(host, user, password, port, db_name):
    try:
        # Connect to PostgreSQL server (default database is 'postgres')
        connection = psycopg.connect(
            host=host,
            user=user,
            password=password,
            port=port,
            dbname="postgres"  # Connect to the default database
        )
        connection.autocommit = True  # To allow database creation outside transactions
        cursor = connection.cursor()

        # Create the new database
        cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
        print(f"Database {db_name} created successfully.")

        # Close the connection to 'postgres'
        cursor.close()
        connection.close()

        # Connect to the new database
        connection = psycopg.connect(
            host=host,
            user=user,
            password=password,
            port=port,
            dbname=db_name
        )
        cursor = connection.cursor()

        # Create a sample table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        print("Table 'users' created successfully.")

        # Commit and close
        connection.commit()
        cursor.close()
        connection.close()

    except psycopg.Error as e:
        print(f"An error occurred: {e}")
    finally:
        if connection:
            connection.close()

# Main entry point
if __name__ == "__main__":
    db_name = "main_db"
    user = "zgdev"
    password = "*mIloisfAT23*123*"
    data_folder = "../data"
    # Example usage
    initialize_postgres_db(
        host="localhost",
        user="zgdev",
        password="*mIloisfAT23*123*",
        port=5432,
        db_name="main_db"
    )

    """conn = connect_to_db(db_name, user, password)
    try:
        initialize_schema(conn)
        populate_database(conn, data_folder)
        print("Database initialized and populated successfully.")
    finally:
        conn.close()"""
