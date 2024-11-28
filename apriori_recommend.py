def recommend_songs(c_scores: dict, k_songs: list, limit=10) -> list:
    # Find itemsets that contain keys in k_songs
    found_sets = {key: value for key, value in c_scores.items() if key in k_songs}
    
    # Aggregate all sub-dict songs with their conf_values
    results = []
    for sub_dict in found_sets.values():
        for song, conf_value in sub_dict.items():
            results.append((song, float(conf_value)))  # Convert conf_value to float
    
    # Sort by conf_value in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Pick the top songs
    top_songs = []
    seen_songs = set()  # To avoid duplicates
    for song, conf_value in results:
        # Ignore k_songs and seen songs
        if song not in seen_songs and song not in k_songs:
            top_songs.append((song, conf_value))
            seen_songs.add(song)
        if len(top_songs) == limit:
            break

    print(top_songs)

    return top_songs


# Example usage
confidence_scores = {
    "song_id 1": {
        "song_id 2": "0.85",
        "song_id 3": "0.75",
        "song_id 6": "0.99",
        "song_id 5": "1"
    },
    "song_id 2": {
        "song_id 1": "0.90",
        "song_id 3": "0.65"
    },
    "song_id 3": {
        "song_id 1": "0.55",
        "song_id 2": "0.95"
    },
    "song_id 4": {
        "song_id 1": "0.2"
    },
    "song_id 5": {
        "song_id 2": "0.99"
    },
    "song_id 6": {
        "song_id 1": "0.2"
    },
    "song_id 7": {
        "song_id 1": "0.2"
    },
    "song_id 8": {
        "song_id 6": "0.2",
        "song_id 7": "0.60",
    }
}

k_songs = ["song_id 1", "song_id 3", "song_id 8"]

# confidence_scores contains a dict of songs with their scores
# k_songs is the songs to recommend based of
# blacklist is the already listened to songs by the user
top_songs = recommend_songs(confidence_scores, k_songs)
