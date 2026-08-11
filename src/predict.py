import mlflow
import mlflow.sklearn

model = mlflow.sklearn.load_model("models:/iris-classifier@champion")

sample = [[5.1,3.5,1.4,0.2]]
prediction = model.predict(sample)
print("Prediction:",prediction)
print("prediction successful")