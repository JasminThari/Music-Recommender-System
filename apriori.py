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

# Helper function to apply one hot encoding on baskets
def one_hot_encoding(x):
    if(x <= 0): 
        return 0
    if(x >= 1): 
        return 1

# Helper function to create a singualar basket
def create_basket(df, column_name):
    basket = (
        df[df[column_name] == 1] 
            .groupby(['song_id', 'genres'])['title'] 
            .sum().unstack().reset_index().fillna(0) 
            .set_index('artist_id')
    )

    return (
        df[df[column_name] == 1] 
            .groupby(['artist_id', 'song_id'])['title'] 
            .sum().unstack().reset_index().fillna(0) 
            .set_index('artist_id')
    )

# Creates basckets for the Apriori algorithm
def create_baskets(df):
    baskets = []
    # baskets.append(create_basket(df, 'genre_Blues'))
    # baskets.append(create_basket(df, 'genre_Country'))
    # baskets.append(create_basket(df, 'genre_Electronic'))
    # baskets.append(create_basket(df, 'genre_Folk'))
    # baskets.append(create_basket(df, 'genre_Jazz'))
    # baskets.append(create_basket(df, 'genre_Latin'))
    # baskets.append(create_basket(df, 'genre_Metal'))
    # baskets.append(create_basket(df, 'genre_New Age'))
    # baskets.append(create_basket(df, 'genre_Pop'))
    # baskets.append(create_basket(df, 'genre_Punk'))
    # baskets.append(create_basket(df, 'genre_Rap'))
    # baskets.append(create_basket(df, 'genre_Reggae'))
    # baskets.append(create_basket(df, 'genre_RnB'))
    # baskets.append(create_basket(df, 'genre_Rock'))
    # baskets.append(create_basket(df, 'genre_World'))

    l.info(baskets[0].columns)

    for bascket in baskets:
        l.debug(f"Basket dataframe: \n{bascket.head}\n")

    return bascket

# Apply apriori algorithm to find frequent itemsets and association rules
def apply_apriori(b):
    frequent_itemsets = apriori(b, min_support=0.05, use_colnames=True)

    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0, num_itemsets=len(b))

    l.info(f"Frequent itemsets: \n{frequent_itemsets.head}\n")
    l.info(f"Association rules: \n{rules.head}\n")

    return rules

DF = load_dataframe("played_songs_cleaned.csv")
BASCKETS = create_baskets(DF)
# R = apply_apriori(B)