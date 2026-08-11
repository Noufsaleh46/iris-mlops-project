from flask import Flask, request, jsonify, render_template

import mlflow
import mlflow.sklearn

app = Flask(__name__)

model = mlflow.sklearn.load_model("/app/model")


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = [[
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]]

    prediction = model.predict(features)

    class_names = ["setosa", "versicolor", "virginica"]
    predicted_class = class_names[int(prediction[0])]

    return jsonify({
        "prediction": int(prediction[0]),
        "class_name": predicted_class
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)