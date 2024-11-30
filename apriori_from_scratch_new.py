from scipy.sparse import csr_matrix, tril
from scipy.special import comb
import pandas as pd
import numpy as np
import json
import math
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict


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

    # set all the play counts to 1
    df["play_count"] = 1

    # Create a sparse matrix
    sparse_matrix = csr_matrix(
        (df["play_count"], (user_codes, song_codes)),
        shape=(df["user_id"].cat.categories.size, df["song_id"].cat.categories.size),
        dtype="bool"
    )
    print(f"Created sparse matrix wih shape: {sparse_matrix.shape}")
    print(f"Memory usage of sparse matrix: {sparse_matrix.data.nbytes / 1024:.2f} KB")
    return sparse_matrix , user_id_mapping, song_id_mapping

def apriori_algorithm(crs_matrix_initial, min_support, max_k=2, write_to_file=True):
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
    TOTAL_NUMBER_OF_USERS = crs_matrix_initial.shape[0]
    SUPPORT_THRESHOLD = int(math.ceil(TOTAL_NUMBER_OF_USERS * min_support))
    WRITE_TO_FILE = write_to_file
    MAX_K = max_k
    OUT_PATH = "data/frequent_itemsets.csv"

    
    def singletons(crs_matrix):
        """
        Finds the singletons above the threshold
        Input:
        crs_matrix: sparse matrix
        Returns:
        filter_crs_matrix: sparse matrix, the filtered matrix
        mapping: dict, mapping from the the new indices to the orginal indices
        """
        single_item_occurance = np.array(crs_matrix.sum(axis=0)).flatten()
        relevant_single_item_indices = np.where(single_item_occurance >= SUPPORT_THRESHOLD)[0]
        print(f"Number of single items above the threshold: {len(relevant_single_item_indices)}")          

        if WRITE_TO_FILE:              
            with open(OUT_PATH, "w") as f:
                relevant_single_items_support = single_item_occurance[relevant_single_item_indices] / TOTAL_NUMBER_OF_USERS
                for index, support in zip(relevant_single_item_indices, relevant_single_items_support):
                    f.write(f"{(index,)},{support}\n")
                
        filtered_csr_idx_to_org_item_idx = {new_index: old_index for new_index, old_index in enumerate(relevant_single_item_indices)}
        filtered_crs_matrix = crs_matrix[:,relevant_single_item_indices]

        return filtered_crs_matrix, filtered_csr_idx_to_org_item_idx
    
    def pairs(crs_matrix, filtered_csr_idx_to_org_item_idx):
        """
        Function to find the pairs above the threshold
        This made separately from k>2 because this step can be done fast using matrix multiplication
        Input:
        crs_matrix: sparse matrix
        filtered_csr_idx_to_org_item_idx: dict, mapping from the filtered singletons to the original indices
        Returns:
        accepted_candidates: dict, keys tuples of the indices of the pairs and values the support    
        """
        pairwise_occurance_matrix = crs_matrix.astype(np.int32).T @ crs_matrix.astype(np.int32)
        # only extract the lower triangle (excluding the diagonal) since the matrix is symmetric
        pairwise_occurance_matrix = tril(pairwise_occurance_matrix, k=-1)
        coo = pairwise_occurance_matrix.tocoo()
        mask = coo.data >= SUPPORT_THRESHOLD
        filtered_rows = coo.row[mask]
        filtered_cols = coo.col[mask]
        filtered_values = coo.data[mask]

        # create a new crs matrix with the bitwise_and of the pairs
        # matrix_i_j = crs_matrix[:, filtered_rows]
        # matrix_j_i = crs_matrix[:, filtered_cols]
        # bitwise_and_matrix = matrix_i_j.multiply(matrix_j_i)

        # save the accepted itemset to dict
        # also save the mapping from the indices of the bitwise_and_matrix to the itemsets      
        accepted_candidates  = {}
        item_set_to_index = {}
        for index, (row, col, value) in enumerate(zip(filtered_rows, filtered_cols, filtered_values)):
            item_set = tuple(sorted((row, col)))            
            
            item_set_to_index[item_set] = index           

            assert item_set not in accepted_candidates, f"Key {item_set} is already in the dictionary, should not happen"
            accepted_candidates[item_set] = value

        if WRITE_TO_FILE:
            with open(OUT_PATH, "a") as f:
                for itemset in accepted_candidates:
                    orginal_item_indices = [filtered_csr_idx_to_org_item_idx[i] for i in itemset]                    
                    support = accepted_candidates[itemset] / TOTAL_NUMBER_OF_USERS
                    f.write(f"{tuple(sorted(orginal_item_indices))},{support}\n")

        #return bitwise_and_matrix, item_set_to_index, accepted_candidates
        return accepted_candidates
    
    def k_plus_2(crs_matrix, previous_accepted_candidates, k):
        """
        Function to find the itemsets with k>2
        Input:
        crs_matrix: sparse matrix
        previous_accepted_candidates: dict, the previous accepted candidates
        k: int, the size of the itemsets to be created"""

        # Extract keys as sorted tuples for efficient comparison
        prev_candidates = {tuple(sorted(c)): support for c, support in previous_accepted_candidates.items()}

        accepted_candidates = {}

        # Group previous itemsets by the first k-1 elements
        group_by_prefix = defaultdict(list)
        
        for candidate in prev_candidates:
            # Create the first k-1 elements as the key for grouping
            prefix = tuple(candidate[:k-2])
            group_by_prefix[prefix].append(candidate)

        for group in tqdm(group_by_prefix.values(), desc="Generating candidates"):
            for set1, set2 in combinations(group, 2):
                # Check if the first k-2 elements match by intersecting sets
                if set1[:-1] == set2[:-1]: # redudant since we already grouped by the first k-2 elements
                    new_candidate_set = tuple(sorted(set(set1).union(set2)))

                    # Prune based on k-1 subset
                    k_minus_1_subsets = list(combinations(new_candidate_set, k - 1))
                    if all(subset in prev_candidates for subset in k_minus_1_subsets):
                        # Compute the support for new candidate set
                        column = crs_matrix[:, new_candidate_set[0]]
                        for item in new_candidate_set[1:]:
                            column = column.multiply(crs_matrix[:, item])

                        support = column.sum() / TOTAL_NUMBER_OF_USERS
                        if support >= SUPPORT_THRESHOLD:
                            accepted_candidates[new_candidate_set] = support

                        if WRITE_TO_FILE:
                            with open(OUT_PATH, "a") as f:
                                orginal_item_indices = [filtered_csr_idx_to_org_item_idx[i] for i in new_candidate_set]                    
                                f.write(f"{tuple(sorted(orginal_item_indices))},{support}\n")
        """
        if WRITE_TO_FILE:
            with open(OUT_PATH, "a") as f:
                for itemset in accepted_candidates:
                    orginal_item_indices = [filtered_csr_idx_to_org_item_idx[i] for i in itemset]                    
                    support = accepted_candidates[itemset] / TOTAL_NUMBER_OF_USERS
                    f.write(f"{tuple(sorted(orginal_item_indices))},{support}\n")
        """
        return accepted_candidates


    # Find the singletons
    filtered_crs_matrix, filtered_csr_idx_to_org_item_idx = singletons(crs_matrix_initial)
    # Find the pairs
    accepted_candidates = pairs(filtered_crs_matrix, filtered_csr_idx_to_org_item_idx)
    # Find the rest of the itemsets
    print("Starting with k>2")
    for k in range(3, MAX_K+1):
        accepted_candidates = k_plus_2(filtered_crs_matrix, accepted_candidates, k)
        if len(accepted_candidates) == 0:
            break
    
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
        
    apriori_algorithm(sparse_matrix, 0.003, write_to_file=True, max_k=6)
    
    
    

                      



