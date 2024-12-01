from scipy.sparse import csr_matrix, tril
from scipy.special import comb
import pandas as pd
import numpy as np
import json
import math

def apriori_algorithm_pairs(crs_matrix: csr_matrix, min_support: float, write_to_file: bool = True, out_path: str = "data/frequent_itemsets.csv"):
                            
    """
    Apriori algorithm for pairs using crs_matrix and matrix multiplication
    Input: 
    crs_matrix: sparse matrix, the transaction matrix
    min_support: float, the minimum support
    write_to_file: bool, if True the frequent itemsets are written to a file
    """
    TOTAL_NUMBER_OF_USERS = crs_matrix.shape[0]
    SUPPORT_THRESHOLD = int(math.ceil(TOTAL_NUMBER_OF_USERS * min_support))
    WRITE_TO_FILE = write_to_file
    OUT_PATH = out_path

    def write_to_file(frequent_itemsets, filtered_csr_idx_to_org_item_idx):
        """
        Input:
        frequent_itemsets: dict, keys are tuples of the indices of the pairs and values are the support
        filtered_csr_idx_to_org_item_idx: dict, mapping from the filtered singletons to the original indices
        """       
        with open(OUT_PATH, "a") as f:                
            for itemset in frequent_itemsets:
                # Ensure valid mapping of filtered indices to original item indices
                original_item_indices = tuple(sorted([filtered_csr_idx_to_org_item_idx[i] for i in itemset]))                   
                support = frequent_itemsets[itemset] / TOTAL_NUMBER_OF_USERS
                f.write(f"{original_item_indices},{support}\n")


    # singletons           
    single_item_occurance = np.array(crs_matrix_initial.sum(axis=0)).flatten()
    relevant_single_item_indices = np.where(single_item_occurance >= SUPPORT_THRESHOLD)[0]
    print(f"Number of single items above the threshold: {len(relevant_single_item_indices)}")
    filtered_csr_idx_to_org_item_idx = {new_index: old_index for new_index, old_index in enumerate(relevant_single_item_indices)}
    filtered_crs_matrix = crs_matrix_initial[:,relevant_single_item_indices]
    # remove old matrix to free up memory
    frequent_singletons = {relevant_single_item_indices[i]: single_item_occurance[i] for i in range(len(relevant_single_item_indices))}
    del crs_matrix_initial

    if WRITE_TO_FILE:
        write_to_file(frequent_singletons, filtered_csr_idx_to_org_item_idx)
    
    # pairs
    pairwise_occurance_matrix = filtered_crs_matrix.astype(np.int32).T @ filtered_crs_matrix.astype(np.int32)
    # only extract the lower triangle (excluding the diagonal) since the matrix is symmetric
    pairwise_occurance_matrix = tril(pairwise_occurance_matrix, k=-1)
    coo = pairwise_occurance_matrix.tocoo()
    mask = coo.data >= SUPPORT_THRESHOLD
    filtered_rows, filtered_cols, filtered_values = coo.row[mask], coo.col[mask], coo.data[mask]

    frequent_pairs  = {}
    for row, col, value in zip(filtered_rows, filtered_cols, filtered_values):
        item_set = tuple(sorted((row, col)))
        assert item_set not in frequent_pairs, f"Key {item_set} is already in the dictionary, should not happen"            
        frequent_pairs[item_set] = value

    if WRITE_TO_FILE:
        write_to_file(frequent_pairs, filtered_csr_idx_to_org_item_idx)

    return frequent_singletons, frequent_pairs
 