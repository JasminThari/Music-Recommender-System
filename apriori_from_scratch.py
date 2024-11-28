from scipy.sparse import csr_matrix, tril
from scipy.special import comb
import pandas as pd
import numpy as np
import json
import math


def find_train_users(path_to_user_data: str, path_to_train_users: str):
    # Read the data files
    listened_songs_all_users = pd.read_csv(path_to_user_data)
    train_users = pd.read_csv(path_to_train_users)["user_id"]

    # Filter the dataframe to include only users in the training set
    listened_songs_train_users = listened_songs_all_users[listened_songs_all_users["user_id"].isin(train_users)]

    # Save the filtered dataframe
    file_path = path_to_user_data.replace(".csv", "_train.csv") 
    listened_songs_train_users.to_csv(file_path, index=False)

    # Print the number of unique users
    print(f"Number of unique users in the training set: {listened_songs_train_users['user_id'].nunique()}")

    return file_path




def create_sparse_transaction_matrix(path_to_user_data: str,use_subset=False) -> csr_matrix:
    """
    Create a transaction matrix from the dataframe
    The df should have the same structure as the user_data_cleaned.csv
    That is just three columns: song_id, user_id play_count
    Input:
    path_to_user_data: str, path to the user data
    use_subset: bool, if True a subset of the data is used
    Returns:
    sparse_matrix: csr_matrix, the transaction matrix
    user_id_mapping: dict, mapping from the integer codes to the original user ids
    song_id_mapping: dict, mapping from the integer codes to the original song ids
    """
    df = pd.read_csv(path_to_user_data)
    print(f"Number of unique songs: {df['song_id'].nunique()}")
    print(f"Number of unique users: {df['user_id'].nunique()}")
    if use_subset:
        #df = df[:1000]
        df = df.sample(frac=0.2)
        print(f"Using a subset of the data")

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

    # Create a sparse matrix
    sparse_matrix = csr_matrix(
        (df["play_count"], (user_codes, song_codes)),
        shape=(df["user_id"].cat.categories.size, df["song_id"].cat.categories.size),
        dtype="bool"
    )
    print(f"Created sparse matrix wih shape: {sparse_matrix.shape}")
    print(f"Memory usage of sparse matrix: {sparse_matrix.data.nbytes / 1024:.2f} KB")
    return sparse_matrix , user_id_mapping, song_id_mapping

def apriori_algorithm(crs_matrix, min_support, singleton_path="data/support_singletons.csv", pair_path="data/support_pairs.csv"):
    """
    This function implements the apriori algorithm from scratch
    It takes a sparse matrix and returns the frequent itemsets
    Input:
    crs_matrix: sparse matrix
    min_support: float, the percentage of transactions that the itemset should occur in etc. 0.01
    singleton_path: str, path to the file where the singletons are saved
    pair_path: str, path to the file where the pairs are saved
    Returns:
    Saves the singletons and pairs to file
    """

    # convert the support to a int threshold
    support_threshold = int(math.ceil(crs_matrix.shape[0] * min_support))
    total_number_of_users = crs_matrix.shape[0]   


    # singleton items
    # find the single items which occur more than the threshold      
    single_item_occurance = np.array(crs_matrix.sum(axis=0)).flatten()
    relevant_single_item_indices = np.where(single_item_occurance >= support_threshold)[0]    
    print(f"Number of single items above the threshold: {len(relevant_single_item_indices)}")  
    
    # create a mapping from the original index to the new index
    index_mapping = {new_index: old_index for new_index, old_index in enumerate(relevant_single_item_indices)}

    # only keep in memory the items that are above the threshold
    crs_matrix = crs_matrix[:,relevant_single_item_indices]    
    print(f"Memory usage of the filtered matrix: {crs_matrix.data.nbytes / 1024:.2f} KB")

    # write the single items to file    
    with open(singleton_path, "w") as f:
        relevant_single_items = single_item_occurance[relevant_single_item_indices] / total_number_of_users
        for index, support in zip(relevant_single_item_indices, relevant_single_items):
            f.write(f"{index},{support}\n")


    # pairwise items    
    # we can find the pairs by multiplying the matrix with its transpose
    # NOTE if we do this on the boolean type matrix, we will only get true or false
    # as the elements of the new matrix. 
    # We are intrested in the number of transactions where the items are bought together
    # thus we must change the matrix to a type int when we multiply it with its transpose   
    pairwise_occurance_matrix = crs_matrix.astype(np.int32).T @ crs_matrix.astype(np.int32)

    # filter the pairs that are above the threshold
    # consider only the lower triangle (excluding the diagonal) since the matrix is symmetric
    pairwise_occurance_matrix = tril(pairwise_occurance_matrix, k=-1)
    # to do filtering we need the matrix in COO format
    coo = pairwise_occurance_matrix.tocoo()
    mask = coo.data >= support_threshold
    filtered_rows = coo.row[mask]
    filtered_cols = coo.col[mask]
    filtered_values = coo.data[mask]
    # get the original indices
    original_rows = [index_mapping[i] for i in filtered_rows]
    original_cols = [index_mapping[i] for i in filtered_cols]
    # write the pairs to file
    with open(pair_path, "w") as f:
        support_percentage = filtered_values / total_number_of_users
        for row, col, value in zip(original_rows, original_cols, support_percentage):
            f.write(f"{row},{col},{value}\n")
    print(f"Number of pairs above the threshold: {len(filtered_rows)}") 

    return 

    # optional (implement for k > 2)

def calc_confidence(song_id_mapping,   
                    path_singleton_support="data/support_singletons.csv",
                    path_pair_support="data/support_pairs.csv",
                    path_output="data/confidence_dict.json"):
    
    """
    Calculate the confidence of the pairs
    Input:
    path_singleton_support: str, path to the file with the singleton support
    path_pair_support: str, path to the file with the pair support
    path_output: str, path to the output file
    Returns:
    None
    """
    # read the files
    singletons = pd.read_csv(path_singleton_support, header=None, names=["item", "support"], index_col="item")
    pairs = pd.read_csv(path_pair_support, header=None, names=["item1", "item2", "support"])

    # Calculate confidence for each pair in both directions
    pairs['confidence_item1_to_item2'] = pairs.apply(lambda row: row['support'] / singletons.loc[row['item1'], 'support'], axis=1)
    pairs['confidence_item2_to_item1'] = pairs.apply(lambda row: row['support'] / singletons.loc[row['item2'], 'support'], axis=1)

    # Create the nested dictionary
    confidence_dict = {}

    for _, row in pairs.iterrows():
        item1, item2 = song_id_mapping[row['item1']], song_id_mapping[row['item2']]
        conf1_to_2 = row['confidence_item1_to_item2']
        conf2_to_1 = row['confidence_item2_to_item1']
        
        # Add item1 -> item2 confidence
        if item1 not in confidence_dict:
            confidence_dict[item1] = {}
        confidence_dict[item1][item2] = conf1_to_2
        
        # Add item2 -> item1 confidence
        if item2 not in confidence_dict:
            confidence_dict[item2] = {}
        confidence_dict[item2][item1] = conf2_to_1

    # Save to JSON file
    with open(path_output, "w") as json_file:
        json.dump(confidence_dict, json_file, indent=4)

    print(f"Confidence data saved to {path_output}")
    return

if __name__ == "__main__": 
        
    PATH_ALL_USER_DATA = "data/user_data_cleaned.csv"
    PATH_TRAIN_USERS = "data/train_users.csv"

    PATH_TRAIN_USER_DATA = find_train_users(PATH_ALL_USER_DATA, PATH_TRAIN_USERS)

    sparse_matrix, user_id_mapping, song_id_mapping = create_sparse_transaction_matrix(PATH_TRAIN_USER_DATA, use_subset=False)
        
    apriori_algorithm(sparse_matrix, 
                        min_support=0.0005)

    calc_confidence(song_id_mapping)
    
    
    

                      



