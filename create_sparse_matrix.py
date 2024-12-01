from scipy.sparse import csr_matrix
import pandas as pd
def create_sparse_matrix(path_to_user_data: str = 'data/processed/user_data_cleaned_train.csv',
                         transaction_matrix: bool = True):
    """
    Create a sparse matrix from a csv file with user data with columns song_id, user_id, play_count    
    Input:
    path_to_user_data: str, path to the user data
    transaction_matrix: bool, if True the play_count is set to 1, else the play_count is used as is
    Returns:
    sparse_matrix: csr_matrix, 
    user_id_mapping: dict, mapping from the integer codes to the original user ids
    song_id_mapping: dict, mapping from the integer codes to the original song ids
    """
    df = pd.read_csv(path_to_user_data)
    print(f"Number of unique songs: {df['song_id'].nunique()}")
    print(f"Number of unique users: {df['user_id'].nunique()}")    

    # Convert user_id and song_id to categories
    # this the pandas way of mapping strings as integers    
    df["user_id"] = df["user_id"].astype("category")
    df["song_id"] = df["song_id"].astype("category")

    # Extract the integer codes from the categories
    user_codes = df["user_id"].cat.codes
    song_codes = df["song_id"].cat.codes

    # mapping from the integer codes to the original strings
    user_id_mapping = dict(enumerate(df["user_id"].cat.categories))
    song_id_mapping = dict(enumerate(df["song_id"].cat.categories))

    
    if transaction_matrix:
        df["play_count"] = 1  
        # create of type bool to save memory      
        sparse_matrix = csr_matrix(
            (df["play_count"], (user_codes, song_codes)),
            shape=(df["user_id"].cat.categories.size, df["song_id"].cat.categories.size),
            dtype="bool"
        )
    else:
        sparse_matrix = csr_matrix(
            (df["play_count"], (user_codes, song_codes)),
            shape=(df["user_id"].cat.categories.size, df["song_id"].cat.categories.size),
            dtype="int32"
        )
        
    print(f"Created sparse matrix wih shape: {sparse_matrix.shape}")
    print(f"Memory usage of sparse matrix: {sparse_matrix.data.nbytes / 1024:.2f} KB")
    return sparse_matrix , user_id_mapping, song_id_mapping