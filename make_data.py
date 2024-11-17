import os
import h5py
import pandas as pd


class DataSetMaker:
    """
    A class to create and pre-process the datasets from HDF5 and other raw data files from Million Song Dataset (MSD).
    """

    def __init__(
        self,
        h5_file_path: str,
        unique_tracks_path: str,
        genres_paths: list,
        user_data_path: str,
        output_dir: str = "data",
    ):
        """
        Initializes the DataSetMaker with the necessary file paths.

        Parameters:
            h5_file_path (str): Path to the HDF5 file.
            unique_tracks_path (str): Path to the unique_tracks.txt file.
            genres_paths (list): List of paths to genres files (e.g., ['cd1', 'cd2', 'cd2c']).
            user_data_path (str): Path to the user_data.txt file.
            output_dir (str): Directory where output CSV files will be saved.
        """
        self.h5_file_path = h5_file_path
        self.unique_tracks_path = unique_tracks_path
        self.genres_paths = genres_paths  # List of genre file paths
        self.user_data_path = user_data_path
        self.output_dir = output_dir

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize dictionaries and DataFrames
        self.dfs = {}
        self.song_to_track_mapping = pd.DataFrame()
        self.merged_df = pd.DataFrame()
        self.df_genres_merged = pd.DataFrame()

    def extract_dfs_from_h5(self):
        """
        Extracts datasets from the HDF5 file and stores them as pandas DataFrames in a dictionary.
        """
        def extract_df_from_h5(file, indent=0, datasets=None):
            if datasets is None:
                datasets = {}
            for key in file:
                item = file[key]
                print("  " * indent + f"- {key}: {type(item)}")

                if isinstance(item, h5py.Group):
                    # Recursively extract from groups
                    extract_df_from_h5(item, indent + 1, datasets)
                elif isinstance(item, h5py.Dataset):
                    # Extract dataset into DataFrame
                    columns = list(item.dtype.names)
                    df = pd.DataFrame(item[:], columns=columns)

                    # Decode byte columns to strings if necessary
                    for col in df.columns:
                        if df[col].dtype == 'O' and isinstance(df[col].iloc[0], bytes):
                            df[col] = df[col].apply(
                                lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
                            )

                    # Store the DataFrame with the dataset's full path as the key
                    datasets[item.name] = df
            return datasets

        with h5py.File(self.h5_file_path, 'r') as h5_file:
            self.dfs = extract_df_from_h5(h5_file)
        print("Extraction from HDF5 completed.")

    def save_dfs_to_csv(self):
        """
        Saves each extracted DataFrame to a CSV file in the output directory.
        """
        for key, df in self.dfs.items():
            # Extract a meaningful name from the dataset path
            name = key.strip("/").replace("/", "_")
            csv_path = os.path.join(self.output_dir, f"{name}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved DataFrame to {csv_path}")
        print("All DataFrames have been saved to CSV.")

    def create_song_to_track_mapping(self):
        """
        Creates a mapping between song IDs and track IDs from the unique_tracks.txt file.
        """
        df_unique_tracks = pd.read_csv(
            self.unique_tracks_path,
            sep='<SEP>',
            header=None,
            engine='python',
            names=['track_id', 'song_id', 'artist', 'title']
        )
        self.song_to_track_mapping = df_unique_tracks[['song_id', 'track_id']].drop_duplicates(subset='song_id', keep='first')
        print("Song to track mapping created.")

    def merge_metadata_and_analysis(self):
        """
        Merges metadata and analysis DataFrames based on song_id and track_id.
        Also cleans the merged DataFrame by removing unnecessary columns.
        """
        # Load necessary DataFrames
        metadata_df = self.dfs.get('/metadata/songs')
        analysis_df = self.dfs.get('/analysis/songs')

        if metadata_df is None or analysis_df is None:
            raise ValueError("Required datasets '/metadata/songs' or '/analysis/songs' not found in HDF5 file.")

        # Drop duplicate song_ids
        metadata_df = metadata_df.drop_duplicates(subset='song_id', keep='first')

        # Merge with song_to_track_mapping
        merged_df = metadata_df.merge(self.song_to_track_mapping, on='song_id', how='inner')

        # Merge with analysis_df on track_id
        merged_df = merged_df.merge(analysis_df, on='track_id', how='inner')

        # Remove unnecessary columns
        columns_to_drop = [
            "artist_7digitalid",
            "artist_mbid",
            "artist_playmeid",
            "release_7digitalid",
            "track_7digitalid",
            "audio_md5",
        ]
        merged_df = merged_df.drop(columns=[col for col in columns_to_drop if col in merged_df.columns])

        # Remove columns with only one unique value
        merged_df = merged_df.loc[:, merged_df.nunique() > 1]

        self.merged_df = merged_df
        print("Metadata and analysis DataFrames merged and cleaned.")

    def process_genres(self):
        """
        Processes genre classification files and merges them into a single DataFrame.
        """
        genre_dfs = []
        for genre_path in self.genres_paths:
            df_genre = pd.read_csv(
                genre_path,
                sep="\t",
                comment='#',
                header=None,
                names=["trackId", "majority_genre", "minority_genre"],
                engine='python'
            )
            genre_dfs.append(df_genre)
            print(f"Loaded genres from {genre_path}")

        # Merge genres by adding new trackIds from subsequent files
        df_genres_cd1 = genre_dfs[0]
        df_genres_cd2 = genre_dfs[1]
        df_genres_cd2c = genre_dfs[2]

        # Add new trackIds from cd2 to cd1
        new_trackIds_cd2 = df_genres_cd2[~df_genres_cd2['trackId'].isin(df_genres_cd1['trackId'])]
        df_genres_cd1_updated = pd.concat([df_genres_cd1, new_trackIds_cd2], ignore_index=True)
        print("Merged genres from cd2 into cd1.")

        # Add new trackIds from cd2c to the updated cd1
        new_trackIds_cd2c = df_genres_cd2c[~df_genres_cd2c['trackId'].isin(df_genres_cd1_updated['trackId'])]
        df_genres_merged = pd.concat([df_genres_cd1_updated, new_trackIds_cd2c], ignore_index=True)
        print("Merged genres from cd2c into the updated cd1.")

        self.df_genres_merged = df_genres_merged
        print("All genre DataFrames have been merged.")
        
        self.df_genres_merged = self.df_genres_merged.drop_duplicates(subset='trackId', keep='first')
        
        # merge with the merged_df
        self.merged_df = self.merged_df.merge(self.df_genres_merged, left_on='track_id', right_on='trackId', how='left')
        self.merged_df = self.merged_df.drop(columns=['trackId'])


    def save_merged_df(self):
        """
        Saves the merged DataFrame to a CSV file.
        """
        # Sort columns alphabetically
        self.merged_df = self.merged_df.reindex(sorted(self.merged_df.columns), axis=1)

        # Define the output path
        merged_csv_path = os.path.join(self.output_dir, "all_songs.csv")
        self.merged_df.to_csv(merged_csv_path, index=False)
        print(f"Merged DataFrame saved to {merged_csv_path}")


    def process_user_data(self):
        """
        Processes user data to find and save songs that have been played by users.
        """
        # Load user data
        df_user_data = pd.read_csv(
            self.user_data_path,
            sep='\t',
            header=None,
            names=['user_id', 'song_id', 'play_count']
        )
        print("User data loaded.")

        # Find unique song_ids that have been played
        unique_song_ids = df_user_data['song_id'].unique()
        print(f"Found {len(unique_song_ids)} unique song IDs in user data.")

        # Load all songs DataFrame
        all_songs_csv_path = os.path.join(self.output_dir, "all_songs.csv")
        df_all_songs = pd.read_csv(all_songs_csv_path)
        print("All songs DataFrame loaded.")

        # Merge to find played songs
        df_played_songs = df_all_songs.merge(
            pd.DataFrame(unique_song_ids, columns=['song_id']),
            on='song_id',
            how='inner'
        )
        print(f"Found {len(df_played_songs)} songs that have been played by users.")

        # Save played songs to CSV
        played_songs_csv_path = os.path.join(self.output_dir, "played_songs.csv")
        df_played_songs.to_csv(played_songs_csv_path, index=False)
        print(f"Played songs DataFrame saved to {played_songs_csv_path}")

    def run_all(self):
        """
        Executes all processing steps in sequence.
        """
        print("Starting data extraction from HDF5...")
        self.extract_dfs_from_h5()

        print("Saving extracted DataFrames to CSV...")
        self.save_dfs_to_csv()

        print("Creating song to track mapping...")
        self.create_song_to_track_mapping()

        print("Merging metadata and analysis DataFrames...")
        self.merge_metadata_and_analysis()

        print("Processing genres...")
        self.process_genres()
        
        print("Saving merged DataFrame to CSV...")
        self.save_merged_df()

        print("Processing user data...")
        self.process_user_data()

        print("All processing steps completed successfully.")

class DataLoader:
    """
    A class to load and manage datasets related to songs and user interactions.
    """

    def load_song_data(self, data_path: str):
        """
        Loads the data_path file .csv file into a pandas DataFrame.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} does not exist.")
        df_songs = pd.read_csv(data_path)
        return df_songs
    
    def load_user_data(self, data_path: str):
        """
        Loads the data_path file .csv file into a pandas DataFrame.
        """
        df_user_data = pd.read_csv(
            data_path,
            sep='\t',
            header=None,
            names=['user_id', 'song_id', 'play_count']
        )
        return df_user_data
