import pandas as pd

def evaluate_recommendations(recommendations_df, masked_songs_df, k):
    """
    Evaluate the recommendations using Precision@k and Recall@k.
    
    Parameters:
    - recommendations_df (pd.DataFrame): DataFrame with 'user_id' and 'recommended_songs' (list of song_ids).
    - masked_songs_df (pd.DataFrame): DataFrame with 'user_id' and 'song_id' of masked songs.
    - k (int): The number of top recommendations to consider.
    
    Returns:
    - evaluation_df (pd.DataFrame): DataFrame with 'user_id', 'precision_at_k', 'recall_at_k'.
    """
    # Ensure the recommended_songs are lists
    recommendations_df['recommended_songs'] = recommendations_df['recommended_songs'].apply(list)

    # Group masked songs by user
    masked_songs_grouped = masked_songs_df.groupby('user_id')['song_id'].apply(set).reset_index()
    masked_songs_dict = dict(zip(masked_songs_grouped['user_id'], masked_songs_grouped['song_id']))
    
    evaluation_results = []
    
    for _, row in recommendations_df.iterrows():
        user_id = row['user_id']
        recommended_songs = row['recommended_songs'][:k]  # Consider only top-k recommendations
        
        # Get the masked songs for the user
        masked_songs = masked_songs_dict.get(user_id, set())
        
        if not masked_songs:
            # If there are no masked songs for the user, we cannot compute recall
            recall_at_k = None
        else:
            # Compute the number of relevant recommended songs
            relevant_recommendations = set(recommended_songs) & masked_songs
            num_relevant = len(relevant_recommendations)
            
            # Precision@k
            precision_at_k = num_relevant / k if k > 0 else 0
            # Recall@k
            recall_at_k = num_relevant / len(masked_songs) if len(masked_songs) > 0 else 0

            evaluation_results.append({
                'user_id': user_id,
                'precision_at_k': precision_at_k,
                'recall_at_k': recall_at_k
            })
    
    evaluation_df = pd.DataFrame(evaluation_results)
    return evaluation_df