import streamlit as st
import pandas as pd
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares

# Load Data
@st.cache_data
def load_data():
    ratings = pd.read_csv('./data/rating_final.csv')
    user_cuisine = pd.read_csv('./data/usercuisine.csv')
    restaurant_cuisine = pd.read_csv('./data/chefmozcuisine.csv')
    restaurant_info = pd.read_csv('./data/geoplaces2.csv')
    restaurant_parking = pd.read_csv('./data/chefmozparking.csv')
    user_profile = pd.read_csv('./data/userprofile.csv')
    return ratings, user_cuisine, restaurant_cuisine, restaurant_info, restaurant_parking, user_profile

ratings, user_cuisine, restaurant_cuisine, restaurant_info, restaurant_parking, user_profile = load_data()

# Prepare data mappings for user-item indices
user_map = {user: idx for idx, user in enumerate(ratings['userID'].unique())}
item_map = {item: idx for idx, item in enumerate(ratings['placeID'].unique())}
item_map_inv = {v: k for k, v in item_map.items()}

ratings['user_idx'] = ratings['userID'].map(user_map)
ratings['item_idx'] = ratings['placeID'].map(item_map)

# Create a sparse matrix in CSR format
sparse_matrix = coo_matrix((ratings['rating'], (ratings['user_idx'], ratings['item_idx']))).tocsr()

# Train the ALS model for collaborative filtering
model = AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)
model.fit(sparse_matrix.T)

# Function for existing users: Collaborative Filtering with contextual filters
def recommend_for_existing_user(user_id, num_recommendations=5):
    user_idx = user_map.get(user_id)
    if user_idx is None:
        st.warning(f"User ID {user_id} not found.")
        return pd.DataFrame(columns=['placeID', 'score'])

    item_indices, scores = model.recommend(user_idx, sparse_matrix[user_idx], N=num_recommendations * 2)
    recommended_items = [item_map_inv[idx] for idx in item_indices]
    
    user_cuisines = user_cuisine[user_cuisine['userID'] == user_id]['Rcuisine'].tolist()
    refined_recommendations = []

    for placeID, score in zip(recommended_items, scores):
        restaurant_cuisines = restaurant_cuisine[restaurant_cuisine['placeID'] == placeID]['Rcuisine'].tolist()
        if set(restaurant_cuisines).intersection(user_cuisines):
            refined_recommendations.append((placeID, score))
        if len(refined_recommendations) >= num_recommendations:
            break

    if not refined_recommendations:
        return pd.DataFrame({'placeID': recommended_items[:num_recommendations], 'score': scores[:num_recommendations]})
    
    return pd.DataFrame(refined_recommendations, columns=['placeID', 'score'])

# Function for existing users: Recommend new items only
def recommend_new_item_for_existing_user(user_id, num_recommendations=5):
    user_idx = user_map.get(user_id)
    if user_idx is None:
        st.warning(f"User ID {user_id} not found.")
        return pd.DataFrame(columns=['placeID', 'score'])

    already_rated_items = set(ratings[ratings['userID'] == user_id]['placeID'])
    item_indices, scores = model.recommend(user_idx, sparse_matrix[user_idx], N=num_recommendations * 2)
    recommendations = [(item_map_inv[idx], score) for idx, score in zip(item_indices, scores) if item_map_inv[idx] not in already_rated_items]
    
    return pd.DataFrame(recommendations[:num_recommendations], columns=['placeID', 'score'])

# Function for new users: Content-Based Recommendations with additional features
def recommend_for_new_user(preferred_cuisines, preferred_ambience, price_range, accessibility, parking_required, activity=None, personality=None, num_recommendations=5):
    potential_restaurants = restaurant_info.copy()

    # Filter by cuisine
    if preferred_cuisines:
        matching_cuisines = restaurant_cuisine[restaurant_cuisine['Rcuisine'].isin(preferred_cuisines)]
        potential_restaurants = potential_restaurants[potential_restaurants['placeID'].isin(matching_cuisines['placeID'])]

    # Filter by ambience
    if preferred_ambience:
        potential_restaurants = potential_restaurants[potential_restaurants['Rambience'] == preferred_ambience]

    # Filter by price range
    if price_range:
        potential_restaurants = potential_restaurants[potential_restaurants['price'] == price_range]

    # Filter by accessibility
    if accessibility:
        potential_restaurants = potential_restaurants[potential_restaurants['accessibility'] == accessibility]

    # Filter by parking requirement
    if parking_required:
        parking_places = restaurant_parking['placeID'].unique()
        potential_restaurants = potential_restaurants[potential_restaurants['placeID'].isin(parking_places)]

    # Activity-based filtering
    if activity:
        if activity == "Student":
            potential_restaurants = potential_restaurants[(potential_restaurants['price'] <= 2) | (potential_restaurants['Rambience'] == "lively")]
        elif activity == "Professional":
            potential_restaurants = potential_restaurants[(potential_restaurants['Rambience'] == "quiet") | (potential_restaurants['price'] >= 3)]

    # Personality-based filtering
    if personality:
        if personality == "Outgoing":
            potential_restaurants = potential_restaurants[potential_restaurants['Rambience'] == "lively"]
        elif personality == "Reserved":
            potential_restaurants = potential_restaurants[potential_restaurants['Rambience'] == "quiet"]

    if potential_restaurants.empty:
        st.warning("No recommendations match the selected preferences. Showing top-rated restaurants instead.")
        fallback_recommendations = restaurant_info[['placeID', 'name', 'address', 'city']].head(num_recommendations)
        return fallback_recommendations

    return potential_restaurants[['placeID', 'name', 'address', 'city', 'price', 'Rambience', 'accessibility']].head(num_recommendations)

# Streamlit UI
st.title("Restaurant Recommendation System")
st.sidebar.header("User Type Selection")

# Ask if the user is new or existing
user_type = st.sidebar.radio("Are you a new or existing user?", ("Existing User", "New User", "Existing User - New Items Only"))

if user_type == "Existing User":
    user_id = st.sidebar.text_input("Enter your User ID")
    num_recommendations = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10, value=5)
    
    if st.sidebar.button("Get Recommendations"):
        recommendations = recommend_for_existing_user(user_id=user_id, num_recommendations=num_recommendations)
        
        if not recommendations.empty:
            recommendations = recommendations.merge(restaurant_info[['placeID', 'name']], on='placeID', how='left')
            max_score = recommendations['score'].max()
            recommendations['score'] = (recommendations['score'] / max_score * 100).round(2)
            recommendations = recommendations.rename(columns={'name': 'Restaurant Name', 'placeID': 'Restaurant ID', 'score': 'Relevance Score (%)'})
            st.subheader(f"Top {num_recommendations} Recommendations for User {user_id}")
            st.table(recommendations[['Restaurant Name', 'Restaurant ID', 'Relevance Score (%)']])

elif user_type == "Existing User - New Items Only":
    user_id = st.sidebar.text_input("Enter your User ID")
    num_recommendations = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10, value=5)
    
    if st.sidebar.button("Get New Item Recommendations"):
        recommendations = recommend_new_item_for_existing_user(user_id=user_id, num_recommendations=num_recommendations)
        
        if not recommendations.empty:
            recommendations = recommendations.merge(restaurant_info[['placeID', 'name']], on='placeID', how='left')
            max_score = recommendations['score'].max()
            recommendations['score'] = (recommendations['score'] / max_score * 100).round(2)
            recommendations = recommendations.rename(columns={'name': 'Restaurant Name', 'placeID': 'Restaurant ID', 'score': 'Relevance Score (%)'})
            st.subheader(f"Top {num_recommendations} New Item Recommendations for User {user_id}")
            st.table(recommendations[['Restaurant Name', 'Restaurant ID', 'Relevance Score (%)']])

elif user_type == "New User":
    st.sidebar.write("Tell us your preferences to get recommendations")
        # Get unique available cuisine types dynamically
    available_cuisines = sorted(restaurant_cuisine['Rcuisine'].unique())
    preferred_cuisines = st.sidebar.multiselect("Preferred Cuisines", options=available_cuisines)
    
    # Dynamic options for ambiance, price, and other filters
    ambience_options = sorted(restaurant_info['Rambience'].dropna().unique())
    preferred_ambience = st.sidebar.selectbox("Preferred Ambience", options=["Any"] + ambience_options)
    preferred_ambience = None if preferred_ambience == "Any" else preferred_ambience

    price_options = sorted(restaurant_info['price'].dropna().unique())
    price_range = st.sidebar.selectbox("Preferred Price Range", options=["Any"] + price_options)
    price_range = None if price_range == "Any" else price_range

    accessibility_options = sorted(restaurant_info['accessibility'].dropna().unique())
    accessibility = st.sidebar.selectbox("Accessibility", options=["Any"] + accessibility_options)
    accessibility = None if accessibility == "Any" else accessibility

    parking_required = st.sidebar.checkbox("Require Parking")

    # New inputs for activity and personality
    activity = st.sidebar.selectbox("Activity", options=["Any", "Student", "Professional"])
    activity = None if activity == "Any" else activity

    personality = st.sidebar.selectbox("Personality", options=["Any", "Outgoing", "Reserved"])
    personality = None if personality == "Any" else personality

    num_recommendations = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

    if st.sidebar.button("Get Recommendations"):
        # Call the content-based recommendation function for new users
        recommendations = recommend_for_new_user(
            preferred_cuisines=preferred_cuisines,
            preferred_ambience=preferred_ambience,
            price_range=price_range,
            accessibility=accessibility,
            parking_required=parking_required,
            activity=activity,
            personality=personality,
            num_recommendations=num_recommendations
        )
        
        # Display recommendations
        if not recommendations.empty:
            st.subheader(f"Top {num_recommendations} Recommendations Based on Your Preferences")
            st.table(recommendations[['name', 'placeID', 'address', 'city', 'price', 'Rambience', 'accessibility']])
        else:
            st.write("No recommendations available based on the selected preferences.")