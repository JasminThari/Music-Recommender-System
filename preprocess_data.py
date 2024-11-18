import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from geopy.geocoders import Nominatim
import time


class DataCleaner:
    """
    A class for cleaning the dataset by handling missing values and filtering users.
    """
    def drop_missing_values(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Drops rows with missing values in specified columns.

        Parameters:
            df (pd.DataFrame): The dataframe to clean.
            columns (list): List of column names to check for missing values.

        Returns:
            pd.DataFrame: Cleaned dataframe with missing values dropped.
        """
        cleaned_df = df.dropna(subset=columns).reset_index(drop=True)
        return cleaned_df

    def filter_users_with_min_songs(self, df_user: pd.DataFrame, df_songs_subs: pd.DataFrame, min_songs: int = 10) -> pd.DataFrame:
        """
        Filters users who have subscribed to at least a specified number of songs.

        Parameters:
            df_user (pd.DataFrame): User dataframe.
            df_songs_subs (pd.DataFrame): Subscribed songs dataframe.
            min_songs (int): Minimum number of songs a user must have subscribed to.

        Returns:
            pd.DataFrame: Filtered dataframe with users meeting the criteria.
        """
        merged_df = pd.merge(df_songs_subs[['song_id']], df_user, on='song_id', how='inner')
        user_subs_filtered = merged_df[merged_df.groupby('user_id')['user_id'].transform('count') > min_songs]
        return user_subs_filtered


class FeatureEngineer:
    """
    A class for engineering features such as artist frequency and hashed artist names.
    """
    
    def one_hot_encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        One-hot encodes a column in the dataframe.

        Parameters:
            df (pd.DataFrame): The dataframe containing the column to encode.
            column (str): The column name to one-hot encode.

        Returns:
            pd.DataFrame: The dataframe with the one-hot encoded column.
        """
        one_hot_encoded = pd.get_dummies(df[column], prefix=column)
        df = pd.concat([df, one_hot_encoded], axis=1)
        return df

    def calculate_frequency(self, df: pd.DataFrame, column: str) -> pd.Series:
        """
        Calculates the frequency of each category in the column in the dataset.

        Parameters:
            df (pd.DataFrame): The dataframe containing artist data.
            column (str): The column name to calculate frequency for.

        Returns:
            pd.Series: A series mapping the category to their frequencies.
        """
        freq = df[column].value_counts()
        return freq

    def hash_artist_features(self, df: pd.DataFrame, n_features, artist_column: str = 'artist_name') -> pd.DataFrame:
        """
        Applies feature hashing to the artist names.

        Parameters:
            df (pd.DataFrame): The dataframe containing artist data.
            artist_column (str): The column name for artist names.

        Returns:
            pd.DataFrame: A dataframe with hashed artist features.
        """
        hasher = FeatureHasher(n_features=n_features, input_type='dict', alternate_sign=False)
        feature_dicts = [{'artist_' + name: 1} for name in df[artist_column]]
        hashed_features = hasher.transform(feature_dicts)
        hashed_features_dense = hashed_features.toarray()
        hashed_df = pd.DataFrame(hashed_features_dense, 
                                  columns=[f'artist_hash_{i}' for i in range(n_features)])
        return hashed_df


class GeocoderService:
    """
    A class to handle geocoding of artist locations to obtain latitude and longitude.
    """

    def __init__(self, user_agent: str = "music_recommender"):
        """
        Initializes the GeocoderService with a specified user agent.

        Parameters:
            user_agent (str): User agent string for the Nominatim geocoder.
        """
        self.geolocator = Nominatim(user_agent=user_agent)

    def get_lat_lon(self, location: str) -> list:
        """
        Retrieves the latitude and longitude for a given location.

        Parameters:
            location (str): The location string to geocode.

        Returns:
            list: A list containing latitude and longitude. [latitude, longitude]
        """
        try:
            loc = self.geolocator.geocode(location)
            if loc:
                return [loc.latitude, loc.longitude]
            else:
                return [np.nan, np.nan]
        except Exception as e:
            print(f"Error geocoding '{location}': {e}")
            return [np.nan, np.nan]
        finally:
            time.sleep(1)  

class DataModelPreparer:
    """
    A class to prepare the processed data for modeling by splitting into training and testing sets and standardizing features.
    """
    def __init__(self, random_state: int = 42):
        """
        Initializes the DataModelPreparer with target column and split parameters.

        Parameters:
            random_state (int): Controls the shuffling applied to the data before applying the split.
        """
        self.random_state = random_state
        self.scaler = StandardScaler()

    def split_data_for_autoencoder(self, data: pd.DataFrame) -> tuple:
        """
        Splits the dataframe into training and testing sets.

        Parameters:
            df (pd.DataFrame): The dataframe to split.
            feature_columns (list, optional): List of columns to use as features. If None, all columns except target are used.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """        
        train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
        
        return train_data,val_data

    def fit_transform_scaler(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the scaler on the training data and transforms it.

        Parameters:
            X_train (pd.DataFrame): Training feature data.

        Returns:
            pd.DataFrame: Scaled training feature data.
        """
        scaled_train = self.scaler.fit_transform(X_train)
        scaled_train_df = pd.DataFrame(scaled_train, columns=X_train.columns, index=X_train.index)
        return scaled_train, scaled_train_df

    def transform_scaler(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the test data using the already fitted scaler.

        Parameters:
            X_test (pd.DataFrame): Testing feature data.

        Returns:
            pd.DataFrame: Scaled testing feature data.
        """
        scaled_test = self.scaler.transform(X_test)
        scaled_test_df = pd.DataFrame(scaled_test, columns=X_test.columns, index=X_test.index)
        return scaled_test, scaled_test_df

