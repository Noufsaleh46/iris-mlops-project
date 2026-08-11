from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature

# 1.load the dataset
iris = load_iris()

X=iris.data
y=iris.target

#2. split data

X_train , X_test, y_train, y_test =train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

#3.create model

model = RandomForestClassifier(
n_estimators=100,
random_state=42
)


mlflow.set_experiment("iris-classification")

#4.train

with mlflow.start_run():
 
   model.fit(X_train,y_train) 

   #5.predict
   predictions = model.predict(X_test)   

   #6.evaluate

   accuracy = accuracy_score(y_test,predictions)
   
   print(f"Accuracy: {accuracy:.4f}")
   
   mlflow.log_param("n_estimators",100)
   mlflow.log_metric("accuracy",accuracy)
   signature = infer_signature(X_train,model.predict(X_train))

   mlflow.sklearn.log_model(sk_model=model,name="model",signature=signature)


#7.save model


print("Model saved successfully")