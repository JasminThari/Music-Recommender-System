import pandas as pd
import logging as l
from pathlib import Path
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

l.basicConfig(level=l.DEBUG, format="%(message)s")

# Loads all csv files we have (5 at the moment) and adds them to a dataframe
def load_dataframes():
    dfs = []
    for file in Path("data/").glob("*.csv"):
        df = pd.read_csv(file)
        dfs.append(df)

        l.debug(file)
        l.debug(df.head())
        l.debug("\n")

    return dfs

DFS = load_dataframes()