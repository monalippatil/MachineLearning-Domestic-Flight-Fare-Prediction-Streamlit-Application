import pandas as pd

def extract_target(df, target_variable):
    """Extract target variable from dataframe

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe
    target_variable: str
        Name of the target variable

    Returns
    -------
    pd.DataFrame
        Pandas dataframe containing all features except target variable
    pd.Series
        Pandas dataframe containing target variable
    """

    df_copy = df.copy()
    target = df_copy.pop(target_variable)

    return df_copy, target

# def split_sets_random(features, target, test_ratio=0.2):
#     """Split sets randomly

#     Parameters
#     ----------
#     features : pd.DataFrame
#         Input dataframe
#     target : pd.Series
#         Target column
#     test_ratio : float
#         Ratio used for the validation and testing sets (default: 0.2)

#     Returns
#     -------
#     Numpy Array
#         Features for the training set
#     Numpy Array
#         Target for the training set
#     Numpy Array
#         Features for the validation set
#     Numpy Array
#         Target for the validation set
#     Numpy Array
#         Features for the testing set
#     Numpy Array
#         Target for the testing set
#     """
#     from sklearn.model_selection import train_test_split

#     val_ratio = test_ratio / (1 - test_ratio)
#     X_data, X_test, y_data, y_test = train_test_split(features, target, test_size=test_ratio, random_state=8)
#     X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=val_ratio, random_state=8)

#     return X_train, y_train, X_val, y_val, X_test, y_test

# def save_sets(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, path='../data/processed/'):
#     """Save the different sets locally

#     Parameters
#     ----------
#     X_train: Numpy Array
#         Features for the training set
#     y_train: Numpy Array
#         Target for the training set
#     X_val: Numpy Array
#         Features for the validation set
#     y_val: Numpy Array
#         Target for the validation set
#     X_test: Numpy Array
#         Features for the testing set
#     y_test: Numpy Array
#         Target for the testing set
#     path : str
#         Path to the folder where the sets will be saved (default: '../data/processed/')

#     Returns
#     -------
#     """
#     import numpy as np

#     if X_train is not None:
#       np.save(f'{path}X_train', X_train)
#     if X_val is not None:
#       np.save(f'{path}X_val',   X_val)
#     if X_test is not None:
#       np.save(f'{path}X_test',  X_test)
#     if y_train is not None:
#       np.save(f'{path}y_train', y_train)
#     if y_val is not None:
#       np.save(f'{path}y_val',   y_val)
#     if y_test is not None:
#       np.save(f'{path}y_test',  y_test)

# def load_sets(path='../data/processed/'):
#     """Load the different locally save sets

#     Parameters
#     ----------
#     path : str
#         Path to the folder where the sets are saved (default: '../data/processed/')

#     Returns
#     -------
#     Numpy Array
#         Features for the training set
#     Numpy Array
#         Target for the training set
#     Numpy Array
#         Features for the validation set
#     Numpy Array
#         Target for the validation set
#     Numpy Array
#         Features for the testing set
#     Numpy Array
#         Target for the testing set
#     """
#     import numpy as np
#     import os.path

#     X_train = np.load(f'{path}X_train.npy', allow_pickle=True) if os.path.isfile(f'{path}X_train.npy') else None
#     X_val   = np.load(f'{path}X_val.npy'  , allow_pickle=True) if os.path.isfile(f'{path}X_val.npy')   else None
#     X_test  = np.load(f'{path}X_test.npy' , allow_pickle=True) if os.path.isfile(f'{path}X_test.npy')  else None
#     y_train = np.load(f'{path}y_train.npy', allow_pickle=True) if os.path.isfile(f'{path}y_train.npy') else None
#     y_val   = np.load(f'{path}y_val.npy'  , allow_pickle=True) if os.path.isfile(f'{path}y_val.npy')   else None
#     y_test  = np.load(f'{path}y_test.npy' , allow_pickle=True) if os.path.isfile(f'{path}y_test.npy')  else None

#     return X_train, y_train, X_val, y_val, X_test, y_test

def save_to_csv(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, path='../data/processed/'):
    """Save data sets in csv format locally

    Parameters
    ----------
    X_train: CSV file
        Features for training set
    y_train: CSV file
        Target for training set
    X_val: CSV file
        Features for validation set
    y_val: CSV file
        Target for validation set
    X_test: CSV file
        Features for testing set
    y_test: CSV file
        Target for testing set
    path : str
        Path to the folder where csv files will be saved (default: '../data/processed/')

    Returns
    -------
    """
    import pandas as pd

    if X_train is not None:
      X_train.to_csv(f'{path}X_train.csv', index=False)
    if X_val is not None:
      X_val.to_csv(f'{path}X_val.csv', index=False)
    if X_test is not None:
      X_test.to_csv(f'{path}X_test.csv', index=False)
    if y_train is not None:
      y_train.to_csv(f'{path}y_train.csv', index=False)
    if y_val is not None:
      y_val.to_csv(f'{path}y_val.csv', index=False)
    if y_test is not None:
      y_test.to_csv(f'{path}y_test.csv', index=False)

def load_csv(path='../data/processed/'):
    """Load csv files

    Parameters
    ----------
    path : str
        Path to the folder where csv files are saved (default: '../data/processed/')

    Returns
    -------
    Pandas Dataframe
        Features for the training set
    Pandas Dataframe
        Target for the training set
    Pandas Dataframe
        Features for the validation set
    Pandas Dataframe
        Target for the validation set
    Pandas Dataframe
        Features for the testing set
    Pandas Dataframe
        Target for the testing set
    """
    import pandas as pd
    import os.path

    X_train = pd.read_csv(f'{path}X_train.csv') if os.path.isfile(f'{path}X_train.csv') else None
    X_val   = pd.read_csv(f'{path}X_val.csv') if os.path.isfile(f'{path}X_val.csv')     else None
    X_test  = pd.read_csv(f'{path}X_test.csv') if os.path.isfile(f'{path}X_test.csv')   else None
    y_train = pd.read_csv(f'{path}y_train.csv') if os.path.isfile(f'{path}y_train.csv') else None
    y_val   = pd.read_csv(f'{path}y_val.csv') if os.path.isfile(f'{path}y_val.csv')     else None

    return X_train, y_train, X_val, y_val, X_test