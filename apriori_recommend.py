import json
import pandas as pd

def find_top_songs(sorted_c_scores: dict, k_songs: list, limit: int=10) -> list:
    # Find c scores that contain keys in k_songs
    found_sets = {}
    for key in k_songs:
        if key in sorted_c_scores:
            found_sets.update(sorted_c_scores[key])
    
    # print(found_sets)

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

    # print(top_songs)

    return top_songs

def recommend_songs_split(sorted_c_scores: dict, l_path: str, m_path: str, override_limit: bool=False) -> list:
    # Get listened and masked songs
    listened_df = pd.read_csv(l_path)
    masked_df = pd.read_csv(m_path)

    # Group songs per user into a dict
    listened_per_user = listened_df.groupby("user_id")["song_id"].apply(list).to_dict() 
    masked_per_user = masked_df.groupby("user_id")["song_id"].apply(list).to_dict() 

    recommended = []

    # Find top songs per user for recommendation
    for user_id, k_songs in listened_per_user.items():
        if override_limit:
            recommended.append((user_id, find_top_songs(sorted_c_scores, k_songs, 50)))
        else:
            rec_limit = len(masked_per_user.get(user_id))
            recommended.append((user_id, find_top_songs(sorted_c_scores, k_songs, rec_limit)))

    #print(recommended)

    return recommended

def recommend_songs(c_score_path: str):
    with open(c_score_path, "r") as file:
        c_scores = json.load(file)

    # Sort each inner dictionary by value in descending order
    sorted_c_scores = {
        outer_key: dict(
            sorted(inner_dict.items(), key=lambda x: x[1], reverse=True)
        )
        for outer_key, inner_dict in c_scores.items()
    }

    l_fixed_path = "data/listened_songs_fixed_split.csv"
    l_ratio_path = "data/listened_songs_ratio_split.csv"
    m_fixed_path = "data/masked_songs_fixed_split.csv"
    m_ratio_path = "data/masked_songs_ratio_split.csv"

    # Recommend songs based on listened song splits
    fixed_recs1 = recommend_songs_split(sorted_c_scores, l_fixed_path, m_fixed_path)
    ratio_recs1 = recommend_songs_split(sorted_c_scores, l_ratio_path, m_ratio_path)
    ratio_recs2 = recommend_songs_split(sorted_c_scores, l_ratio_path, m_ratio_path, True)

    confidence_level = c_score_path.split("_")[-1].split(".json")[0]

    # Save to JSON file
    with open(f"data/apriori_fixed10_recommendations_c{confidence_level}.json", "w") as json_file:
        json.dump(fixed_recs1, json_file, indent=4)
    with open(f"data/apriori_ratio10_recommendations_c{confidence_level}.json", "w") as json_file:
        json.dump(ratio_recs1, json_file, indent=4)
    with open(f"data/apriori_ratio50_recommendations_c{confidence_level}.json", "w") as json_file:
        json.dump(ratio_recs2, json_file, indent=4)

recommend_songs("data/confidence_dict_0.001.json")
recommend_songs("data/confidence_dict_0.0005.json")

# Example for running finding top songs
# sorted_c_scores = {
#     "song_id_1": {"song_id_2": 0.85, "song_id_3": 0.75, "song_id_4": 0.50},
#     "song_id_2": {"song_id_1": 0.90, "song_id_3": 0.65},
#     "song_id_3": {"song_id_1": 0.55, "song_id_5": 0.95},
#     "song_id_4": {"song_id_3": 0.55, "song_id_2": 0.95},
#     "song_id_5": {"song_id_4": 0.55, "song_id_2": 0.95},
# }

# k_songs = ["song_id_1", "song_id_3"]

# find_top_songs(sorted_c_scores, k_songs)