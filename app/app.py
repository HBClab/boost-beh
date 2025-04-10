import os
from flask import Flask, send_from_directory
from main.utils import construct_master_list, DatabaseReading
from psycopg_pool import ConnectionPool

def update_png_paths_and_create_serve_function(app):
    """
    Updates the PNG file paths in MASTER_LIST to use the new served directory structure
    and creates a Flask route to serve the files.
    """
    master_list = app.config["MASTER_LIST"]
    data_folder = app.config["DATA_FOLDER"]

    # Create a new directory structure for serving
    for subject_id, subject_data in master_list.items():
        for task_name, task_data in subject_data.get("tasks", {}).items():
            new_png_paths = []
            for file_path in task_data.get("png_paths", []):
                # Extract relative path: 'subject/task/file'
                relative_path = os.path.relpath(file_path, data_folder)
                new_png_paths.append(f"data/{relative_path}")

            # Update the master list with the new paths
            task_data["png_paths"] = new_png_paths
    # Add a route to serve the updated files
    @app.route("/data/<path:subpath>")
    def serve_data_file(subpath):
        """
        Serve files from the data directory using the new structure.
        """
        file_path = os.path.join(data_folder, subpath)
        if not os.path.exists(file_path):
            return f"File not found: {file_path}", 404

        directory, filename = os.path.split(file_path)
        return send_from_directory(directory, filename)


def create_app():
    app = Flask(__name__)
    app.config['DATA_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    app.config['ALLOWED_EXTENSIONS'] = {'csv', 'txt', 'png'}

    # Ensure the data folder exists
    if not os.path.exists(app.config['DATA_FOLDER']):
        raise FileNotFoundError(f"Data folder not found at {app.config['DATA_FOLDER']}")

    # Construct the master list and store it in the app config
    app.config['MASTER_LIST'] = construct_master_list(app.config['DATA_FOLDER'])
    pool = ConnectionPool(conninfo="dbname=boost-beh host=localhost port=5432")
    db_utils = DatabaseReading(db_name="boost-beh", user="user", password="pass", data_folder="/path/to/data", pool=pool)    # Update paths in the master list and add the serve route
    
    with app.app_context():
        update_png_paths_and_create_serve_function(app)

    # Register blueprints
    from feed_blueprint import feed_print
    from home_blueprint import home_blueprint
    app.register_blueprint(feed_print)
    app.register_blueprint(home_blueprint)

    return app


if __name__ == '__main__':
    # Initialize and run the Flask app
    app = create_app()
    app.run(debug=True)
