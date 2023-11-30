# import mlflow
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)


# Set our tracking server uri for logging
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
# mlflow.set_experiment("MLflow Quickstart")

# print("j")


# with mlflow.start_run():
#     mlflow.onnx.load_model("/home/eliza/passportdetect/models/model.onnx")
#     mlflow.log_params(params)
#     mlflow.log_metric("accuracy", accuracy)
#     mlflow.set_tag("Training Info", "Basic LR model for iris data")
#     signature = infer_signature(X_train, lr.predict(X_train))

#     model_info = mlflow.sklearn.log_model(
#         sk_model=lr,
#         artifact_path="iris_model",
#         signature=signature,
#         input_example=X_train,
#         registered_model_name="tracking-quickstart",
#     )

# from skl2onnx import convert_sklearn
# from skl2onnx.common.data_types import FloatTensorType

# initial_type = [("float_input", FloatTensorType([None, 4]))]
# onx = convert_sklearn(lr, initial_types=initial_type)
# with open("logreg_iris.onnx", "wb") as f:
#     f.write(onx.SerializeToString())


# class my_model(nn.Module):
#     def __init__(self, features, num_classes):
#         super(my_model, self).init()
#         self.linear_1 = nn.Linear(features, 256)
#         self.linear_2 = nn.Linear(256, num_classes)
#         self.act_1 = nn.ReLU()
#         self.batchnorm_1 = nn.BatchNorm1d(features)

#     def forward(self, x):
#         x = self.batchnorm_1(x)
#         x = self.linear_1(x)
#         x = self.act_1(x)
#         x = self.linear_2(x)
#         return F.softmax(x, dim=1)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = my_model(NUM_FEATURES, NUM_CLASSES).to(device)
