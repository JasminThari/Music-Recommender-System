import pandas as pd
import logging as l
import seaborn as sns
import matplotlib.pyplot as plt
import random
from mlxtend.frequent_patterns import apriori, association_rules

l.basicConfig(level=l.DEBUG, format="%(message)s")

PATH_USER_DATA = "data/user_data_cleaned.csv"

PATH_TEST_DATA = "data/test_users.csv"
PATH_TRAIN_DATA = "data/train_users.csv"

LISTENED_FIXED_SPLIT_PATH = "data/listened_songs_fixed_split.csv"
LISTENED_FIXED_RATIO_PATH = "data/listened_songs_fixed_ratio.csv"

MASKED_FIXED_SPLIT_PATH = "data/masked_songs_fixed_split.csv"
MASKED_FIXED_RATIO_PATH = "data/masked_songs_fixed_ratio.csv"

def create_transactions():
    # load data
    df = pd.read_csv(PATH_USER_DATA)

    #TODO handle memory better
    df = df[:100000] 

    # Create transactions & hot encode
    transactions = (
        df.groupby(['user_id', 'song_id'])['play_count'] 
          .sum().unstack().reset_index().fillna(0) 
          .set_index('user_id')
        ).map(lambda x: 0 if x <= 0 else 1 if x >= 1 else x)

    # Investigate data
    exploratory_data_analysis(df, transactions)

    # Visualize data
    data_visualization(df)

    return transactions, df

def exploratory_data_analysis(df, t):
    blue = "\033[34m"
    green = "\033[32m"
    end = "\033[0m"

    # Dataframe info
    l.debug(f"{blue}Dataframe:{end}")
    l.debug(f"{green}Head:\n{df.head}\n{end}")
    l.debug(f"{green}Shape:\n{df.shape}\n{end}")
    l.debug(f"{green}Info:\n{df.info}\n{end}")
    l.debug(f"{green}Is null sum:\n{df.isnull().sum()}\n{end}")
    l.debug(f"{green}Describe:\n{df.describe().T}\n{end}")

    # Transactions info
    l.debug("{blue}Transactons:{end}")
    l.debug(f"{green}Head:\n{t.head}\n{end}")

def data_visualization(df):
    # Top 10 most listened to song ids
    song_count = df.groupby("song_id")["play_count"].sum().nlargest(10)
    song_count = song_count.reset_index()

    plt.figure(figsize=(12, 8))

    ax = sns.barplot(
        data = song_count,
        y = "song_id",
        x = "play_count",
        palette = "icefire")

    for i in ax.containers:
        ax.bar_label(i,)

    ax.set_title("Top 10 most listened to song ids")
    plt.xlabel("Total play count")
    plt.ylabel("Song ids")
    plt.tight_layout()
    plt.show()

def apply_apriori(b): 
    # Find frequent items
    frequent_items = apriori(
        df = b, 
        min_support = 0.003, 
        use_colnames = True
    ) 
    
    l.debug(f"{frequent_items.head}\n")

    # Create rules
    rules = association_rules(
        df = frequent_items, 
        metric = "lift", 
        min_threshold = 0.8, 
        num_itemsets = b.shape[0]
    ).sort_values(
        by = ['confidence', 'lift'], 
        ascending = [False, False]
    )

    l.debug(f"{rules.head}\n")

    return rules

def recommend_songs(rules):
    # Load test and train data
    test_data = pd.read_csv(LISTENED_FIXED_SPLIT_PATH)
    train_data = pd.read_csv(MASKED_FIXED_SPLIT_PATH) #TODO?

    # Pick a random user from the test data
    user_id = random.choice(test_data['user_id'].unique())

    # Get 10 random songs listened to by the user
    user_songs = test_data[test_data['user_id'] == user_id]['song_id'].sample(n=10, random_state=42).tolist()

    l.info(f"User: {user_id}, Initial Songs: {user_songs}")

    # Recommend songs using association rules
    recommendations = set()
    for song in user_songs:
        # Find rules where the test user's songs are in the antecedents
        matching_rules = rules[rules['antecedents'].apply(lambda x: song in x)]

        # Extract the consequents from the matching rules
        for _, rule in matching_rules.iterrows():
            recommendations.update(rule['consequents'])

    # Remove songs the user already knows
    recommendations = recommendations - set(user_songs)

    # Select up to 10 recommendations
    recommendations = list(recommendations)[:10]

    l.info(f"Recommended Songs for User {user_id}: {recommendations}")

    return recommendations

# BASKETS, SONGS = create_transactions()
# RULES = apply_apriori(BASKETS)
# recommend_songs(RULES)