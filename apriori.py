import pandas as pd
import logging as l
from pathlib import Path
from mlxtend.frequent_patterns import apriori, association_rules

l.basicConfig(level=l.DEBUG, format="%(message)s")

# Load played_songs_cleaned into a dataframe
def load_dataframe(filename):
    file_path = Path(f"data/{filename}")

    df = pd.read_csv(file_path)

    l.debug(f"\033[94m{file_path} dataframe: \033[0m\n{df.head}")
    l.info(f"\033[92mColumns: \033[0m\n{df.columns}\n")
    
    return df

# Creates basckets for the Apriori algorithm
def create_baskets(df):
    baskets = []

    l.info(baskets[0].columns)

    for b in baskets:
        l.debug(f"Basket dataframe: \n{b.head}\n")

    return baskets

# Apply apriori algorithm to find frequent itemsets and association rules
def apply_apriori(b):
    frequent_itemsets = apriori(b, min_support=0.05, use_colnames=True)

    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0, num_itemsets=len(b))

    l.info(f"Frequent itemsets: \n{frequent_itemsets.head}\n")
    l.info(f"Association rules: \n{rules.head}\n")

    return rules

DF = load_dataframe("played_songs_cleaned.csv")
# BASCKETS = create_baskets(DF)
# R = apply_apriori(B)