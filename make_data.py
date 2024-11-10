import os
import h5py
import pandas as pd
#%% Load the data
base_dir = "Data/MillionSongSubset"

data_list = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".h5"):
            file_path = os.path.join(root, file)
            
            # Extract relevant fields for a recommendation algorithm
            with h5py.File(file_path, 'r') as f:
                try:
                    # Metadata fields
                    artist_name = f['metadata']['songs'][:][0][f['metadata']['songs'].dtype.names.index('artist_name')].decode()
                    song_title = f['metadata']['songs'][:][0][f['metadata']['songs'].dtype.names.index('title')].decode()
                    artist_id = f['metadata']['songs'][:][0][f['metadata']['songs'].dtype.names.index('artist_id')].decode()
                    song_id = f['metadata']['songs'][:][0][f['metadata']['songs'].dtype.names.index('song_id')].decode()
                    release = f['metadata']['songs'][:][0][f['metadata']['songs'].dtype.names.index('release')].decode()
                    hotness = f['metadata']['songs'][:][0][f['metadata']['songs'].dtype.names.index('song_hotttnesss')]
                    artist_terms = [term.decode() for term in f['metadata']['artist_terms'][:]]
                    artist_terms_freq = [term for term in f['metadata']['artist_terms_freq'][:]]
                    
                    # Analysis fields
                    track_id = f['analysis']['songs'][:][0][f['analysis']['songs'].dtype.names.index('track_id')].decode()
                    tempo = f['analysis']['songs'][:][0][f['analysis']['songs'].dtype.names.index('tempo')]
                    loudness = f['analysis']['songs'][:][0][f['analysis']['songs'].dtype.names.index('loudness')]
                    duration = f['analysis']['songs'][:][0][f['analysis']['songs'].dtype.names.index('duration')]
                    danceability = f['analysis']['songs'][:][0][f['analysis']['songs'].dtype.names.index('danceability')]
                    energy = f['analysis']['songs'][:][0][f['analysis']['songs'].dtype.names.index('energy')]
                    key = f['analysis']['songs'][:][0][f['analysis']['songs'].dtype.names.index('key')]
                    mode = f['analysis']['songs'][:][0][f['analysis']['songs'].dtype.names.index('mode')]
                    year = f['musicbrainz']['songs'][:][0][f['musicbrainz']['songs'].dtype.names.index('year')]
                    
                    # Append extracted data to the list
                    data_list.append({
                        'song_id': song_id,
                        'track_id': track_id,
                        'artist_id': artist_id,
                        'song_title': song_title,
                        'artist_name': artist_name,
                        'release': release,
                        'artist_terms': artist_terms,
                        'artist_terms_freq': artist_terms_freq,
                        'song_hotness': hotness,
                        'tempo': tempo,
                        'loudness': loudness,
                        'duration': duration,
                        'danceability': danceability,
                        'energy': energy,
                        'key': key,
                        'mode': mode,
                        'year': year
                    })
                    
                except KeyError as e:
                    print(f"Missing key {e} in file {file_path}")

df_songs = pd.DataFrame(data_list)

#%% Load the genre data
df_genres_first = pd.read_csv('data/genre_first.cls', delimiter='\t', header=None, names=['track_id', 'genre'])
df_genres_second = pd.read_csv('data/genre_second.cls', delimiter='\t', comment='#', header=None, names=['track_id', 'genre'])
df_genres_third = pd.read_csv('data/genre_third.cls', delimiter='\t', comment='#', header=None, names=['track_id', 'genre'])

df_genres = pd.concat([df_genres_first, df_genres_second, df_genres_third]).drop_duplicates(subset=['track_id'], keep='first')

df_songs_genre = pd.merge(df_songs, df_genres, on='track_id', how='left')

#%% Load the user data
df_users = pd.read_csv('data/user_data.txt', sep='\t', header=None, names=['user_id', 'song_id', 'play_count'])
df_songs_users = pd.merge(df_users, df_songs_genre, on='song_id', how='inner')

#%% Save the data
df_songs_users.to_csv('data/song_user.csv', index=False)
df_songs_genre.to_csv('data/song_genre_data.csv', index=False)
df_users.to_csv('data/user_data.csv', index=False)

print("Data saved successfully")