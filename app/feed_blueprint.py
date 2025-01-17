from flask import Blueprint, current_app, render_template, url_for, redirect
from .main.utils import filter_master_list_by_key

feed_print = Blueprint("feed", __name__)

@feed_print.route("/results/<string:key>/<string:value>")
def results_data(key=None, value=None):
    """
    The route that filters the MASTER_LIST by a given key/value,
    generates cards, and then renders them within the results_page.html template.
    """
    master_list = current_app.config['MASTER_LIST']
    if key == "Subject":
        data = filter_master_list_by_key(master_list, key, value)
    # Filter the master list by the provided key and value
    data = filter_master_list_by_key(master_list, key, value)
    # Total completed is the total number of matching tasks
    total_completed = sum(
        len(subject_data.get("tasks", {}))
        for subject_id, subject_data in data.items()
    )

    # Create the card data to be displayed
    cards = []
    for subject_id, subject_data in data.items():
        for task_name, task_data in subject_data.get("tasks", {}).items():
            png_paths = task_data.get("png_paths", [])
            cards.append({
                "subject": f"{subject_id}",
                "task": f"{task_name}",
                "date": task_data["date"],
                "site": subject_data["site"],
                "category": task_data["category"],
                "session": task_data["session"],
                "image1": url_for('serve_data_file', subpath=png_paths[0][5:]) if len(png_paths) > 0 else None,
                "image2": url_for('serve_data_file', subpath=png_paths[1][5:]) if len(png_paths) > 1 else None
            })

    # Render the results_page.html template with the filtered data
    return render_template(
        'results_page.html',
        value=value,
        key=key.capitalize(),
        total_completed=total_completed,
        cards=cards
    )



from flask import request, redirect, url_for, current_app  # Ensure all necessary imports

def determine_search_type(query):
    """
    Determine the search type (task, subject, or category) based on the input query.

    Args:
        query (str): User's search query.

    Returns:
        tuple: (key, value) where key is 'task', 'subject', or 'category', and value is the query.
    """
    master_list = current_app.config["MASTER_LIST"]
    query = query.upper()
    # Check if the query matches a task
    for subject, subject_data in master_list.items():
        if query in subject_data.get("tasks", {}):
            return "task", query

    # Check if the query matches a subject (search in keys of master_list)
    if query in master_list.keys():
        return "subject", query

    # Check if the query matches a category
    for subject_data in master_list.values():
        for task_data in subject_data.get("tasks", {}).values():
            if task_data.get("category") == query:
                return "category", query

    # Return None if no match is found
    return None, None

@feed_print.route("/results/search", methods=["POST"])
def results_search():
    """
    Handle the search query on the results page.
    """
    # Get the search query from the form
    search_query = request.form.get("search_query", "").strip()

    # Determine the search type and redirect to the appropriate results
    key, value = determine_search_type(search_query)

    if key and value:
        return redirect(url_for("feed.results_data", key=key, value=value))
    else:
        return "No results found. Please refine your search.", 400

@feed_print.route("/results/waffle")
def waffle():
    """
    Handles the waffle button click (top-right of results page).
    You can define the redirection or action for this button here.
    """
    return "Waffle button pressed!"  # Replace this with the appropriate logic
from flask import request # Import request and other necessary functions

@feed_print.route('/image_redirect')
def image_redirect():
    """
    Redirect to the native URL of the given image path.
    """
    # Get the image path from the query parameter
    image_path = request.args.get('image_path')

    # Validate the image path
    if not image_path:
        return "Image path is missing", 400  # Return a 400 Bad Request if no image path is provided

    # Redirect directly to the provided image path
    return redirect(image_path)



@feed_print.route("/results/home")
def home_button_press():
    """
    Redirects to the home page when the home button (top-left) is pressed.
    """
    return redirect(url_for("home_blueprint.home"))
