import pandas as pd
import os
from ucimlrepo import fetch_ucirepo
from typing import Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def download_data_from_uciml(id: int, data_folder: str = "../data"):
    """Fetch and download dataset from UC Irvine Machine Learning API"""
    
    # create the data_folder if it does not exist
    abs_data_folder = os.path.abspath(data_folder)
    if not os.path.exists(abs_data_folder):
        os.makedirs(abs_data_folder)
        
    # fetch data
    repo = fetch_ucirepo(id=id)
    
    # download data
    repo.data.features.to_csv(os.path.join(abs_data_folder, 'features.csv'), index=False)
    repo.data.targets.to_csv(os.path.join(abs_data_folder, 'target.csv'), index=False)
    
def load_data(data_folder: str = "../data") -> Tuple[pd.DataFrame, pd.Series]:
    
    abs_data_folder = os.path.abspath(data_folder)
    features = pd.read_csv(os.path.join(abs_data_folder, 'features.csv'))
    target_df = pd.read_csv(os.path.join(abs_data_folder, 'target.csv'))

    target = target_df.iloc[:, 0] if target_df.shape[1] == 1 else target_df
    return (features, target)

def preprocess_data(X_train, X_test=None):
    """
    Preprocess the dataset: Encode categorical variables, normalize numerical variables, and convert to PyTorch tensors.
    The preprocessor is fitted on training data and transforms both training and test data (if provided)

    Parameters:
    X_train (DataFrame): The feature train set.
    X_test (DataFrame): The feature test set.

    
    Returns:
    X_train (DataFrame): The preprocessed train features
    X_test (DataFrame): The preprocessed test features
    """
    categorical_features = X_train.select_dtypes(include=['object']).columns
    numeric_features = X_train.select_dtypes(include=['float64']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)])

    X_train = preprocessor.fit_transform(X_train)
    if X_test is not None:
        X_test = preprocessor.transform(X_test)
        return X_train, X_test
    else:
        return X_train

if __name__ == "__main__":
    download_data_from_uciml(id=544)