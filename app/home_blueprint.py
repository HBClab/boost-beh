from flask import Blueprint, request, redirect, url_for, current_app, render_template

home_blueprint = Blueprint("home_blueprint", __name__)

@home_blueprint.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get user input from the search bar
        search_query = request.form.get("search_query", "").strip()

        # Determine the search type based on input
        key, value = determine_search_type(search_query)

        if key and value:
            # Redirect to the feed_blueprint route with the key and value
            return redirect(url_for("feed.results_data", key=key, value=value))
        else:
            # If no match, return an error message or reload the homepage
            return "No results found. Please refine your search.", 400

    # Render the home page
    return render_template("homepage.html")


def determine_search_type(query):
    """
    Determine the search type (task, subject, or category) based on the input query.

    Args:
        query (str): User's search query.

    Returns:
        tuple: (key, value) where key is 'task', 'subject', or 'category', and value is the query.
    """
    master_list = current_app.config["MASTER_LIST"]

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
            if task_data.get("site") == query:
                return "site", query
    # Return None if no match is found
    return None, None

# Example endpoints for waffle and search, if needed:
@home_blueprint.route("/waffle_function")
def waffle_function():
    # Do something for waffle
    return "Waffle function triggered"

