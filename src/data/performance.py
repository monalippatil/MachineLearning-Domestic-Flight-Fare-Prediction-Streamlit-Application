import pandas as pd
import numpy as np

# Below are functions for classifier, see functions for regression further down

class BaseLinePerformance_classifer:
    """
    Class to perform baseline performance calculation for classifier models
    ...

    Attributes
    ----------
    y: Numpy Array-like or Pandas DataFrame
        Target variable
    prediction_value: Float
        Prediction value
    preds: Numpy Array
        Prediction value in numpy array format

    Methods
    -------
    fit(y)
        Store the input target variable and calculate the prediction value
    transform(y)
        Transform predictions value into numpy array
    fit_transform(y)
        Performs fit and then transform
    """

    def __init__(self):
        self.y = None
        self.prediction_value = None
        self.preds = None

    def fit(self, y):
        self.y = y
        self.prediction_value = y.mode()

    def transform(self, y):
        self.preds = np.full(y.shape, self.prediction_value)
        return self.preds

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(self.y)
    


def print_performance_scores(y, y_prediction, y_proba, set_name=None):
    """Print AUROC, precision, recall and f1 scores

    Parameters
    ----------
    y: Numpy Array
        Actual target
    y_prediction: Numpy Array
        Predicted target
    y_proba: Numpy Array
        Prediction probability
    set_name: str
        Name of the set to be printed

    Returns
    -------
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score

    print(f'{set_name}:')
    if y_proba is not None:
        print(f"AUROC: {round(roc_auc_score(y, y_proba[:,1]), 4)}")
    print(f"Precision: {round(precision_score(y, y_prediction), 4)}")
    print(f"Recall: {round(recall_score(y, y_prediction), 4)}")
    print(f"F1_score: {round(f1_score(y, y_prediction), 4)}")

def compute_model_performance(model, features, target, set_name=''):
    """Save predicted values and then calculate and print its AUROC, RMSE and MAE scores

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        pre-trained model with pre-defined set of hyperparameters
    features: Pandas DataFrame
        Features
    target: Pandas DataFrame
        Target variable
    set_name : str
        Name of dataset to be printed

    Returns
    -------
    """
    preds = model.predict(features)
    proba = model.predict_proba(features)
    print_performance_scores(y=target, y_prediction=preds, y_proba=proba, set_name=set_name)

def train_evaluate_model(model, X_train, y_train, X_val, y_val):
    """Train a model, perform prediction and assess the model's performance by showing its AUROC, precision, recall and F1 scores on the training and validation set and then return trained model

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Instantiate model with pre-defined set of hyperparameters
    X_train: Pandas DataFrame
        Training dataset containing features/predictors
    y_train: Pandas DataFrame
        Training dataset containing target variable only
    X_val: Pandas DataFrame
        Validation dataset containing features/predictors
    y_val: Pandas DataFrame
        Validation dataset containing target variable only

    Returns
    sklearn.base.BaseEstimator
        Trained model
    -------
    """
    model.fit(X_train, y_train.values.ravel())
    compute_model_performance(model, X_train, y_train, set_name='Training')
    print('---------')
    compute_model_performance(model, X_val, y_val, set_name='Validation')
    return model


# Functions for regression


class BaseLinePerformance_regressor:
    """
    Class to perform baseline performance calculation for classifier models
    ...

    Attributes
    ----------
    y: Numpy Array-like or Pandas DataFrame
        Target variable
    prediction_value: Float
        Prediction value
    preds: Numpy Array
        Prediction value in numpy array format

    Methods
    -------
    fit(y)
        Store the input target variable and calculate the prediction value
    transform(y)
        Transform predictions value into numpy array
    fit_transform(y)
        Performs fit and then transform
    """

    def __init__(self):
        self.y = None
        self.prediction_value = None
        self.preds = None

    def fit(self, y):
        self.y = y
        self.prediction_value = y.mean()

    def transform(self, y):
        self.preds = np.full(y.shape, self.prediction_value)
        return self.preds

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(self.y)


def print_regression_performance_scores(y, y_prediction, set_name=None):
    """Print AUROC, precision, recall and f1 scores

    Parameters
    ----------
    y: Numpy Array
        Actual target
    y_prediction: Numpy Array
        Predicted target
    y_proba: Numpy Array
        Prediction probability
    set_name: str
        Name of the set to be printed

    Returns
    -------
    """
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae

    print(f'{set_name}:')
    print(f"RMSE: {round(mse(y, y_prediction, squared=False), 4)}")
    print(f"MAE: {round(mae(y, y_prediction), 4)}")


def compute_regression_model_performance(model, features, target, set_name=''):
    """Save predicted values and then calculate and print its RMSE score

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        pre-trained model with pre-defined set of hyperparameters
    features: Pandas DataFrame
        Features
    target: Pandas DataFrame
        Target variable
    set_name : str
        Name of dataset to be printed

    Returns
    -------
    """
    preds = model.predict(features)
    print_regression_performance_scores(y=target, y_prediction=preds, set_name=set_name)


def train_evaluate_regression_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train a model, perform prediction and assess the model's performance by showing its AUROC, precision, recall and F1 scores on the training and validation set and then return trained model

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Instantiate model with pre-defined set of hyperparameters
    X_train: Pandas DataFrame
        Training dataset containing features/predictors
    y_train: Pandas DataFrame
        Training dataset containing target variable only
    X_val: Pandas DataFrame
        Validation dataset containing features/predictors
    y_val: Pandas DataFrame
        Validation dataset containing target variable only

    Returns
    sklearn.base.BaseEstimator
        Trained model
    -------
    """
    model.fit(X_train, y_train.values.ravel())
    compute_regression_model_performance(model, X_train, y_train, set_name='Training')
    print('---------')
    compute_regression_model_performance(model, X_val, y_val, set_name='Validation')
    print('---------')
    compute_regression_model_performance(model, X_test, y_test, set_name='Testing')
    return model


# Other


class compute_transform_save:
    """ Class to perform prediction using provided model, reformat the result into 2 columns (player ID and result) and saving it as a csv file

    Attributes
    -------
    model: sklearn.base.BaseEstimator
        pre-trained model
    data: Pandas DataFrame
        data to be used for prediction
    column_name: str
        target variable name
    final_format_path: str
        path where final format is saved
    destination_path: str
        path where the transformed result will be saved

    Methods
    -------
    predict_result(model, data, column_name)
        Use provided model to perform prediction and save it to a variable
    transform_result(final_format, column_name)
        Transform result into two columns (player ID and result)
    save_result_as_csv(model, data, column_name, final_format, destination_path)
        Perform predict, transform and then save result to a csv file
    """

    import pandas as pd

    def __init__(self):
        self.model = None
        self.data = None
        self.column_name = None
        self.final_format_path = None
        self.destination_path = None

    def predict_result(self, model, data, column_name):
        y_test_preds = model.predict_proba(data)
        self.y_test_df = pd.DataFrame({column_name: y_test_preds[:,1]})

    def transform_result(self, final_format_path, column_name):
        self.final_result = pd.read_csv(f'{final_format_path}')
        self.final_result.drop([column_name], axis=1, inplace=True)
        self.final_result[column_name] = self.y_test_df[column_name]

    def save_result_as_csv(self, model, data, column_name, final_format_path, destination_path):
        self.predict_result(model, data, column_name)
        self.transform_result(final_format_path, column_name)
        self.final_result.to_csv(f'{destination_path}', index=False)
