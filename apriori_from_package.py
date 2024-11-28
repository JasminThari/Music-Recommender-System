import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, apriori
from mlxtend.preprocessing import TransactionEncoder
import os
import psutil

PATH_USER_DATA = "data/user_data_cleaned.csv"

df = pd.read_csv(PATH_USER_DATA)

#use x percent of the data

#df = df.sample(frac=0.2)

#df = df[:10000]
print(f"Number of unique songs: {df['song_id'].nunique()}")
print(f"Number of unique users: {df['user_id'].nunique()}")

transactions = df.groupby("user_id")["song_id"].apply(list).reset_index(name="songs")

# we only care about the transactions, not which user it is
# potentially we need to make a mapping if we want to know which user it is
transactions = transactions["songs"].tolist()

# use the transaction encoder to transform the data into a one-hot encoded format
te = TransactionEncoder()
sparse_matrix = te.fit(transactions).transform(transactions, sparse=True)
print(f"Memory usage for the sparse matrix: {sparse_matrix.data.nbytes / 1024:.2f} KB")

sparse_df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, columns=te.columns_)
# change the column names to integers
#sparse_df.columns = range(len(df.columns))

print(f"Memory usage for the sparse df: {sparse_df.memory_usage().sum() / 1024:.2f} KB")

# check used memory

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
# remove the sparse matrix to free up memory
del sparse_matrix

# check used memory
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")


def from_number_to_support(number: int):
    return number / sparse_df.shape[0]

# find the support of single items
#support_one_item = sparse_df.sum() / sparse_df.shape[0]
frequent_itemsets = apriori(sparse_df, min_support=0.001, use_colnames=False, verbose=1,low_memory=True)
print("i made it")
# save the frequent itemsets
frequent_itemsets.to_csv("data/frequent_itemsets_n.csv", index=False)


