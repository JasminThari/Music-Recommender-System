# Music-Recommender-System

### Data
Raw data can be downloaded from [here](https://drive.google.com/file/d/1kuEDAC_oX9hkf_a7zpBgMg_hXa6caiGO/view?usp=sharing) \
Unpack and put in a directory called `data/raw` in the root of the project. \
Then run create_data.ipynb to create the data for the project.
all_songs.csv are all the songs in the dataset. \
played_songs.csv only consist of songs that have been played by at least one user. \


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

## Approach 3: Collaborative Filtering: User-Based Collaborative Filtering with or without Singular Value Decomposition (SVD)
- Main idea: Collaborative Filtering relies on the principle that users with similar behavior (e.g., listening habits) will have similar preferences. In User-Based Collaborative Filtering, we focus on finding users who have similar tastes and recommend songs that these similar users have enjoyed but the target user hasn't listened to yet.
- Steps to implement: 
    * Create a User-Song Interaction Matrix (R): From user_df, construct a matrix where rows represent users, columns represent songs, and the entries represent the number of times a user has listened to a song.
    * Compute Similarities Between Users: Calculate the similarity between users using metrics like Cosine Similarity.
    * Find Similar Users and Recommend Songs:
        * For each target user:
        * Identify the top N similar users.
        * Aggregate the songs listened to by these users.
        * Recommend songs not yet listened to by the target user.
    * Consider to use Singular Value Decomposition (SVD) which is a matrix factorization technique used to reduce the dimensionality of data, capturing the latent factors influencing user preferences. Applying SVD can enhance the collaborative filtering process by handling sparsity and uncovering hidden relationships in the data.

## Approach 4: Content-Based Filtering
- Main idea: Content-Based Filtering recommends items similar to those a user has liked in the past, based on item features. In our case, it utilizes the metadata from Song_df to find songs similar to the ones a user has already enjoyed.
- Steps to implement: 
    * Prepare Song Features: Extract relevant features from Song_df such as genre, artist, tempo, mood, etc.
        * For categorical/textual data, use techniques like One-Hot Encoding or TF-IDF Vectorization.
    * Create a Feature Matrix: Combine all features into a single feature matrix.
    * Construct User Profiles: For each user, create a profile by aggregating the features of the songs they have listened to.
    * Calculate Similarity Between User Profile and Songs: Use cosine similarity to compute how similar each song is to the user's profile.
    * Filtering Out Already Listened Songs
        * Exclude Songs the User Has Already Heard: Recommend songs they have not listened to.
     

## Approach 5: Association Rule Mining - Apriori Algorithm
- Association Rule Mining seeks to discover relationships between variables in large datasets. In a music recommender system, it can find patterns of songs frequently listened to together. The Apriori Algorithm is used to identify frequent itemsets and derive association rules.
- Steps to implement:
    * Prepare Transaction Data
    * Create User Song Lists: Represent each user's listening history as a transaction.
    * Apply the Apriori Algorithm
    * Derive Rules from Frequent Itemsets
    * Recommend Based on Association Rules:
        * For a target user:
        * Identify songs they have listened to.
        * Find rules where the antecedent is a subset of these songs.
        * Recommend songs in the consequent that the user hasn't heard yet.
 
## Final recommendation system: 
- Ensemble of all outputs! 
