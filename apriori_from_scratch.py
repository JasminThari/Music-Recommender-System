from scipy.sparse import csr_matrix, tril
from scipy.special import comb
import pandas as pd
import numpy as np
import itertools
import sys
import math

PATH_USER_DATA = "data/user_data_cleaned.csv"

def create_sparse_transaction_matrix(path_to_user_data: str,use_subset=False) -> csr_matrix:
    """
    Create a transaction matrix from the dataframe
    The df should have the same structure as the user_data_cleaned.csv
    That is just three columns: song_id, user_id play_count
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

    # Create a sparse matrix
    sparse_matrix = csr_matrix(
        (df["play_count"], (user_codes, song_codes)),
        shape=(df["user_id"].cat.categories.size, df["song_id"].cat.categories.size),
        dtype="bool"
    )
    print(f"Created sparse matrix wih shape: {sparse_matrix.shape}")
    print(f"Memory usage of sparse matrix: {sparse_matrix.data.nbytes / 1024:.2f} KB")
    return sparse_matrix

def aprioni_algorithm(crs_matrix, min_support, max_k=2):
    """
    This function implements the apriori algorithm from scratch
    It takes a sparse matrix and returns the frequent itemsets
    Input:
    crs_matrix: sparse matrix
    min_support: float, the percentage of transactions that the itemset should occur in etc. 0.01
    max_k: int, the maximum size of the itemset    
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
    with open("data/single_items.csv", "w") as f:
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
    with open("data/pairs.csv", "w") as f:
        support_percentage = filtered_values / total_number_of_users
        for row, col, value in zip(original_rows, original_cols, support_percentage):
            f.write(f"{row},{col},{value}\n")
    print(f"Number of pairs above the threshold: {len(filtered_rows)}") 

    #TODO implement the rest of the algorithm for k > 2





def aprioni_algorithm_old(crs_matrix, min_support, k=1,max_k=2,first_call=True):
    """
    This function implements the apriori algorithm from scratch
    It takes a sparse matrix and returns the frequent itemsets
    Input:
    crs_matrix: sparse matrix
    """
    if first_call:
        assert k == 1, "The first call should be with k=1"
    
    if k > max_k:
        print("Reached the maximum k")
        return
    
    print(f"Starting calculation for k={k}")    
    # consider only items that have not been pruned in the previous step
    none_zero_columns = crs_matrix.sum(axis=0,dtype=np.int32).nonzero()[1].astype(np.int32)
    print(f"Number of items considered: {len(none_zero_columns)}")

    # create the itemsets of k size 
    # will be VERY memory intensive if the number of items is to large
    print(f"Total number of combinations: {comb(len(none_zero_columns),k)}")

    # calculate the memory usage int32     

    print(f"Memory usage of the itemsets (Without list overhead): {comb(len(none_zero_columns),k) * k * 4 / 1024:.2f} KB")
    
    item_sets = list(itertools.combinations(none_zero_columns, k))
    print(f"Memory usage of the itemsets (With list overhead): {sys.getsizeof(item_sets) / 1024:.2f} KB")  
    
    if k == 1:
        # find the support of single items
        item_sets_support = crs_matrix.sum(axis=0)/crs_matrix.shape[0]
        # zip the itemsets and the support
        zipped = zip(item_sets, item_sets_support)    
    
    else:
        # now only consider transactions
        pass

    #write zipped to file
    print("Writing to file")
    with open(f"data/support_k_{k}.csv", "w") as f:
        for itemset, support in zipped:
            f.write(f"{itemset},{support}\n")

    print("Done writing to file")
    
    k += 1
    aprioni_algorithm_old(crs_matrix, min_support, k=k, max_k=max_k, first_call=False)



if __name__ == "__main__":
    sparse_matrix = create_sparse_transaction_matrix(PATH_USER_DATA, use_subset=False)
    aprioni_algorithm(sparse_matrix, 
                        min_support=0.001,
                        max_k=1)
                      



