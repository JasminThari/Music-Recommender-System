from itertools import combinations
import json
import csv
import temp
import pandas as pd

def find_top_songs_single(sorted_c_scores: dict, k_songs: set, limit: int=10) -> list:
    # Find c scores that contain keys in k_songs
    found_sets = {}
    for key in k_songs:
        if key in sorted_c_scores:
            found_sets.update(sorted_c_scores[key])
    
    # Pick the top songs
    top_songs = []
    seen_songs = set()  # To avoid duplicates
    for song, conf_value in found_sets.items():
        # Ignore k_songs and seen songs
        if song not in seen_songs and song not in k_songs:
            top_songs.append((song, conf_value))
            seen_songs.add(song)
        if len(top_songs) == limit:
            break

    return top_songs

def find_top_songs_sets(sorted_c_scores: dict, k_songs: set, limit: int = 10) -> list:
    # Initialize a list to hold all found sets
    found_sets = []
    
    # Get only the k_songs that appear as keys in sorted_c_scores
    valid_k_songs = [song for song in k_songs if any(song in key for key in sorted_c_scores.keys())]

    valid_itemset_size = max(len(key) for key in sorted_c_scores.keys())

    # Try to find subsets starting from the largest to the smallest
    for subset_size in range(len(valid_k_songs), 0, -1):  # Start from largest subset size to 1
        print(subset_size)
        if subset_size > valid_itemset_size:
            continue
        # Generate all combinations of the current subset size
        for subset in combinations(valid_k_songs, subset_size):
            subset_set = frozenset(subset)  # Use frozenset instead of set
            
            # Check if the frozenset subset is a key in the dictionary
            if subset_set in sorted_c_scores:
                found_sets.append(sorted_c_scores[subset_set])
            else:
                # If the exact subset doesn't exist, we need to find subsets containing one or more of the songs
                for key in sorted_c_scores.keys():
                    # Check if key contains any song from the current subset
                    if not subset_set.isdisjoint(key):  # Check if there's any intersection
                        found_sets.append(sorted_c_scores[key])

    # Pick the top songs
    top_songs = []
    seen_songs = set()  # To avoid duplicates
    
    # Iterate through the found sets and pick songs until the limit is reached
    for current_set in found_sets:
        for song, conf_value in current_set.items():
            # Ignore k_songs and seen songs
            if song not in seen_songs and song not in k_songs:
                top_songs.append((song, conf_value))
                seen_songs.add(song)
            if len(top_songs) == limit:
                break
        if len(top_songs) == limit:
            break

    return top_songs

def recommend_songs_split(sorted_c_scores: dict, l_path: str, m_path: str, override_limit: bool=False) -> list:
    # Get listened and masked songs
    listened_df = pd.read_csv(l_path)
    masked_df = pd.read_csv(m_path)

    # Group songs per user into a dict
    listened_per_user = listened_df.groupby("user_id")["song_id"].apply(set).to_dict() 
    masked_per_user = masked_df.groupby("user_id")["song_id"].apply(set).to_dict() 

    recommended = []

    # Find top songs per user for recommendation
    for user_id, k_songs in listened_per_user.items():
        if override_limit:
            recommended.append((user_id, find_top_songs_single(sorted_c_scores, k_songs, 50)))
        else:
            rec_limit = len(masked_per_user.get(user_id))
            recommended.append((user_id, find_top_songs_single(sorted_c_scores, k_songs, rec_limit)))

    return recommended

def save_recommendations_to_csv(recommendations: list, output_path: str):
    with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(["user_id", "recommended_songs", "c_score"])
        
        # Write the recommendations
        for user_id, user_recs in recommendations:
            songs, scores = [], []
            for song, score in user_recs:
                songs.append(song)
                scores.append(score)
            writer.writerow([user_id, songs, scores])

def recommend_songs(c_score_path: str):
    confidence_level = c_score_path.split("_")[-1].split(".json")[0]

    print(f"Setup c_scores{confidence_level}...")
    with open(c_score_path, "r") as file:
        c_scores = json.load(file)
    print("  1/2 loading c_scores completed.")

    # Sort each inner dictionary by value in descending order
    sorted_inner_dicts = {
        outer_key: dict(
            sorted(inner_dict.items(), key=lambda x: x[1], reverse=True)
        )
        for outer_key, inner_dict in c_scores.items()
    }

    # Sort the outer dictionary keys by their length in descending order
    sorted_c_scores = {
        key: sorted_inner_dicts[key]
        for key in sorted(sorted_inner_dicts.keys(), key=len, reverse=True)
    }
    print("  2/2 sorting c_scores completed.")

    l_fixed_path = "data/listened_songs_fixed_split.csv"
    l_ratio_path = "data/listened_songs_ratio_split.csv"
    m_fixed_path = "data/masked_songs_fixed_split.csv"
    m_ratio_path = "data/masked_songs_ratio_split.csv"

    # Recommend songs based on listened song splits
    print("Recommending songs split...")
    fixed_recs1 = recommend_songs_split(sorted_c_scores, l_fixed_path, m_fixed_path)
    print("  1/3 fixed 10 completed.")
    ratio_recs1 = recommend_songs_split(sorted_c_scores, l_ratio_path, m_ratio_path)
    print("  2/3 ratio 10 completed.")
    ratio_recs2 = recommend_songs_split(sorted_c_scores, l_ratio_path, m_ratio_path, True)
    print("  3/3 ratio 50 completed.")

    # Save to CSV file
    print("Saving to csv...")
    save_recommendations_to_csv(
        fixed_recs1, f"data/apriori_fixed10_recommendations_c{confidence_level}.csv"
    )
    print("  1/3 fixed 10 completed.")
    save_recommendations_to_csv(
        ratio_recs1, f"data/apriori_ratio10_recommendations_c{confidence_level}.csv"
    )
    print("  2/3 ratio 10 completed.")
    save_recommendations_to_csv(
        ratio_recs2, f"data/apriori_ratio50_recommendations_c{confidence_level}.csv"
    )
    print("  3/3 ratio 50 completed.")

# recommend_songs("data/confidence_dict_0.001.json")
# recommend_songs("data/confidence_dict_0.0005.json")

recs_df = pd.read_csv("data/apriori_ratio50_recommendations_c0.0005.csv")
mask_df = pd.read_csv("data/masked_songs_ratio_split.csv")

k = 50
evaluation_df = temp.evaluate_recommendations(recs_df, mask_df, k)

mean_precision_at_k = evaluation_df['precision_at_k'].mean()
mean_recall_at_k = evaluation_df['recall_at_k'].dropna().mean() 
print(f"Mean Precision@{k}: {mean_precision_at_k:.4f}")
print(f"Mean Recall@{k}: {mean_recall_at_k:.4f}")

# Example data 1
# sorted_c_scores = {
#     "song_id_1": {"song_id_2": 0.85, "song_id_3": 0.75, "song_id_4": 0.50},
#     "song_id_2": {"song_id_1": 0.90, "song_id_3": 0.65},
#     "song_id_3": {"song_id_1": 0.55, "song_id_5": 0.95},
#     "song_id_4": {"song_id_3": 0.55, "song_id_2": 0.95},
#     "song_id_5": {"song_id_4": 0.55, "song_id_2": 0.95},
# }

# k_songs = ["song_id_1", "song_id_3"]

# find_top_songs(sorted_c_scores, k_songs)

# Example data 2
# sorted_c_scores = {
#     frozenset(["song_id1", "song_id3", "song_id4"]): {
#         "song_id2": 0.88
#     },
#     frozenset(["song_id1", "song_id5", "song_id2"]): {
#         "song_id10": 0.1
#     },
#     frozenset(["song_id1", "song_id2"]): {
#         "song_id3": 0.99,
#         "song_id4": 0.77,
#         "song_id17": 0.40
#     },
#     frozenset(["song_id3", "song_id5"]): {
#         "song_id1": 0.99,
#         "song_id6": 0.77,
#         "song_id8": 0.33,
#         "song_id4": 0.22
#     },
#     frozenset(["song_id5"]): {
#         "song_id1": 0.99,
#         "song_id2": 0.77,
#         "song_id10": 0.69,
#         "song_id3": 0.33,
#         "song_id4": 0.22
#     }
# }

# k_songs = ["song_id1", "song_id2", "song_id5"]

# # Find top songs
# top_songs = find_top_songs_sets(sorted_c_scores, k_songs, limit=3)
# print(top_songs)