import pandas as pd
import logging as l
from pathlib import Path
from mlxtend.frequent_patterns import apriori, association_rules

l.basicConfig(level=l.DEBUG, format="%(message)s")

# Loads all csv files we have and adds them to a dataframe list
# 0: analysis df
# 1: played_songs df
# 2: all_songs df
# 3: metadata df
# 4: musicbrainz df
def load_dataframes():
    dfs = []
    for file in Path("data/").glob("*.csv"):
        df = pd.read_csv(file)
        dfs.append(df)

        l.debug(f"\033[94m{file} dataframe: \033[0m\n{df.head()}")
        l.info(f"\033[92mColumns: \033[0m\n{df.columns}\n")

    return dfs

# Creates basket data format to be used for the Apriori algorithm
def create_basket(dfs):
    played_songs_df = dfs[1]
    metadata_df = dfs[3]

    basket = []

    l.debug(f"Basket dataframe: \n{basket}\n")

    return basket

# Apply apriori algorithm to find frequent itemsets and association rules
def apply_apriori(b):
    frequent_itemsets = apriori(b, min_support=0.05, use_colnames=True)

    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0, num_itemsets=len(b))

    l.debug(f"Frequent itemsets: \n{frequent_itemsets.head()}\n")
    l.debug(f"Association rules: \n{rules.head()}\n")

    return rules

DFS = load_dataframes()
# B = create_basket(DFS)
# R = apply_apriori(B)