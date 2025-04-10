import os
import shutil
import psycopg
import pytest
from update_db import DatabaseUtils

# Test database credentials
TEST_DB_NAME = "boost-beh-test"
TEST_USER = "user"
TEST_PASSWORD = "password"
TEST_FOLDER = "../../data"

@pytest.fixture(scope="module")
def db_utils():
    """Fixture to initialize and clean up DatabaseUtils."""
    os.makedirs(TEST_FOLDER, exist_ok=True)
    db = DatabaseUtils(TEST_DB_NAME, TEST_USER, TEST_PASSWORD, TEST_FOLDER)
    db.connect()
    yield db
    shutil.rmtree(TEST_FOLDER)  # Cleanup test directory
    db.connection.close()  # Close connection

def test_add_or_get_id(db_utils):
    """Test inserting or retrieving IDs from the database."""
    study_id = db_utils._add_or_get_id("study", {"name": "TestStudy"})
    assert isinstance(study_id, int)

    site_id = db_utils._add_or_get_id("site", {"name": "TestSite", "study_id": study_id})
    assert isinstance(site_id, int)

def test_update_database(db_utils):
    """Test updating the database with directory structure."""
    os.makedirs(os.path.join(TEST_FOLDER, "StudyA", "Site1", "0001", "Task1", "data"), exist_ok=True)
    os.makedirs(os.path.join(TEST_FOLDER, "StudyA", "Site1", "0001", "Task1", "plot"), exist_ok=True)

    db_utils.update_database()
    db_utils.connection.commit()

    with db_utils.connection.cursor() as cursor:
        cursor.execute("SELECT * FROM study WHERE name = 'StudyA';")
        assert cursor.fetchone() is not None
