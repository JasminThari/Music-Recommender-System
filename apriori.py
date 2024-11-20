import pandas as pd
import logging as l
import random
from pathlib import Path
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

    l.debug(f"{df.head}\n")

    #TODO handle memory better
    df = df[:100000] 

    # Create transactions & hot encode
    transactions = (
        df.groupby(['user_id', 'song_id'])['play_count'] 
          .sum().unstack().reset_index().fillna(0) 
          .set_index('user_id')
        ).map(lambda x: 0 if x <= 0 else 1 if x >= 1 else x)

    l.debug(f"{transactions.head}\n")

    return transactions, df

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

BASKETS, SONGS = create_transactions()
RULES = apply_apriori(BASKETS)
recommend_songs(RULES)