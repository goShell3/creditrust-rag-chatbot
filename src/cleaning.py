import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.impute import KNNImputer

class CleaningData:
    def __init__(self, df: DataFrame):
        self.df = df

    def remove_missing_values(self) -> DataFrame:
        """
        Removes all rows with missing values.
        Returns:
            Cleaned DataFrame
        """
        self.df = self.df.dropna()
        return self.df

    def insert_mean_value(self) -> DataFrame:
        """
        Fills missing values with column-wise mean.
        Returns:
            Cleaned DataFrame
        """
        self.df = self.df.fillna(self.df.mean(numeric_only=True))
        return self.df

    def predictive_impute(self, n_neighbors=3) -> DataFrame:
        """
        Fills missing values using KNN imputation.
        Args:
            n_neighbors: number of neighbors to use for imputation
        Returns:
            Cleaned DataFrame
        """
        imputer = KNNImputer(n_neighbors=n_neighbors)
        self.df = pd.DataFrame(imputer.fit_transform(self.df), columns=self.df.columns)
        return self.df
