import pytest

try:
    from app import app
except ImportError as e:
    raise ImportError(f"Could not import 'app' from app.py. {e}")


@pytest.fixture
def client():
    """
    Pytest fixture to create a Flask test client.
    """
    with app.test_client() as client:
        yield client


def test_index_page(client):
    """
    Test that the home page is accessible (status code 200)
    and contains the text "SVM Classifier".
    """
    response = client.get('/')
    assert response.status_code == 200
    # Check if "SVM Classifier" is in the response HTML
    assert b"SVM Classifier" in response.data


def test_prediction_valid(client):
    """
    Test a valid prediction request (with numeric inputs).
    Expect "Predicted Class:" in response.
    """
    response = client.post('/', data={'x1': '15.55', 'x2': '28.65'})
    assert response.status_code == 200
    assert b"Predicted Class:" in response.data


def test_prediction_invalid(client):
    """
    Test an invalid prediction request (non-numeric input).
    Expect "Invalid input" message in response.
    """
    response = client.post('/', data={'x1': 'notANumber', 'x2': '28.65'})
    assert response.status_code == 200
    assert b"Invalid input. Please enter numeric values." in response.data


def test_page_not_found(client):
    """
    Test a non-existent route. Expect 404 status code.
    """
    response = client.get('/does_not_exist')
    assert response.status_code == 404
