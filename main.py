import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split

# Load dataset 
# Change path 
file_path = "u.data"  

# Load data in Surprise Dataset
reader = Reader(line_format='user item rating timestamp', sep='\t', rating_scale=(1, 5))
data = Dataset.load_from_file(file_path, reader=reader)

# Split the dataset 
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# Training algorithm
svd = SVD()
svd.fit(train_set)

# top-N recommendations for user 
def get_movie_recommendations_svd(user_id, top_n=5):
    # Predict ratings for unrated movies
    unrated_movies = [item for item in range(1, 1683) if not train_set.ur.get(user_id) or item not in train_set.ur[user_id]]

    predictions = [svd.predict(user_id, movie_id).est for movie_id in unrated_movies]

    # Get top-N movie recommendations
    top_recommendations = [movie_id for _, movie_id in sorted(zip(predictions, unrated_movies), reverse=True)][:top_n]
    return top_recommendations

# Testing recommendation system
test_user_id = 1
recommended_movies_svd = get_movie_recommendations_svd(test_user_id)
print(f"SVD-based recommended movies for user {test_user_id}: {recommended_movies_svd}")

# Item-Item Collaborative Filtering algorithm
item_item_collab = KNNBasic(sim_options={'user_based': False})
item_item_collab.fit(train_set)

# top-N recommendations for Item-Item Collaborative Filtering
def get_movie_recommendations_item_item_collab(user_id, top_n=5):
    # Predict ratings for unrated movies
    unrated_movies = [item for item in range(1, 1683) if not train_set.ur.get(user_id) or item not in train_set.ur[user_id]]

    predictions = [item_item_collab.predict(user_id, movie_id).est for movie_id in unrated_movies]

    # top-N movie recommendations
    top_recommendations = [movie_id for _, movie_id in sorted(zip(predictions, unrated_movies), reverse=True)][:top_n]
    return top_recommendations

# Testing Item-Item Collaborative Filtering recommendation system
recommended_movies_item_item_collab = get_movie_recommendations_item_item_collab(test_user_id)
print(f"Item-Item Collaborative Filtering recommended movies for user {test_user_id}: {recommended_movies_item_item_collab}")
