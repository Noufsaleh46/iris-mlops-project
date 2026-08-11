import requests


def test_iris_prediction():
    url = "http://localhost:8000/predict"

    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = requests.post(url, json=data)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == 0
    assert result["class_name"] == "setosa"