# Music-Recommender-System

## Approach 1: Clustering of songs

- Use PCA + Kmeans & Autoencoder + Kmeans to make clustering of all songs 
- Make an analysis of the clusters to figure out what distinguishes and characterize the different clusters. 
- For each (user, song) pair recommend K songs based on the cluster that the song belongs to. 
    - Improve clustering by recommend: 
        - The most popular in term of recency (when a user likes the song how many times does they listen to it) song within the cluster 
        - Make user profiles: For instance, analyze the specific attributes (tempo, loudness, genre) that are more frequently found in the songs each user likes. Use this to rank songs within the cluster based on similarity to their preferences.
        - Do only recommend songs that they have not listened to. 

## Approach 2: Clustering of Users

Prepare Data: 
- Aggregate Metadata: For each user, aggregate the metadata of all songs they've listened to. Common aggregation techniques include calculating means, sums, variances. 

Clustering: 
- Clustering: Apply a clustering algorithm on these aggregated features to group users into distinct clusters based on their musical preferences.
- Interpret Clusters: Analyze cluster centroids or representative users to understand the defining characteristics of each cluster (e.g., genre preferences, tempo, popularity).
- Validate Quality: Use metrics like silhouette score, Davies-Bouldin index to ensure clusters are meaningful and distinct.

Recommender System: 
- For each cluster:
    * Aggregate Song Data: Compile all songs listened to by users in the cluster.
    * Determine Popularity: Identify the most frequently listened-to songs or those with high engagement within the cluster.
    * Analyze Metadata Trends: Look for common metadata attributes (e.g., genres, artists, moods) that are prevalent in the cluster's preferred songs.
- For each user: 
    * Select Top Songs: From the identified cluster that the user belongs to, select songs that are popular or have high engagement but the user hasn't listened to yet.
    * Filter and Rank: Apply additional filters (e.g., recency, diversity) and rank the recommendations based on relevance.

## Validate recommendation - run on another data where we have done split 
- For each user: Split the songs listened to into train and test - we call the test songs for relevant song. 
- Apply the clustering on the train set. 
- For each (user, song in train set), find the cluster that the (user, song) pair belongs to and recommend k new songs from the cluster. 
- Evaluate the recommendations  using precision, recall and F1 score: 
    - Recommended songs: []
    - Relevant songs: [] - songs from the test
    - Count how many  of the recommended songs that are in relevant songs:  Number of Relevant Songs in Top K
    - Precision: Number of Relevant Songs in Top K / K 
    - Recall: Number of Relevant Songs in Top K / Total number of relevant songs
    - F1: 2 * (precision * recall) / (precision+recall) 


 

## Final recommendation system: 
- Ensemble of all outputs! 
