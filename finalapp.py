import streamlit as st
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
from surprise.accuracy import rmse
from sklearn.metrics import mean_squared_error, precision_score, recall_score, average_precision_score

# Load Data
@st.cache_data
def load_data():
    ratings = pd.read_csv('/Users/beylouni/Documents/pucrs/sisrec/RestaurantRecommenderSystem/data/rating_final.csv')
    user_cuisine = pd.read_csv('/Users/beylouni/Documents/pucrs/sisrec/RestaurantRecommenderSystem/data/usercuisine.csv')
    restaurant_cuisine = pd.read_csv('/Users/beylouni/Documents/pucrs/sisrec/RestaurantRecommenderSystem/data/chefmozcuisine.csv')
    restaurant_info = pd.read_csv('/Users/beylouni/Documents/pucrs/sisrec/RestaurantRecommenderSystem/data/geoplaces2.csv', encoding='latin1')
    restaurant_parking = pd.read_csv('/Users/beylouni/Documents/pucrs/sisrec/RestaurantRecommenderSystem/data/chefmozparking.csv')
    user_profile = pd.read_csv('/Users/beylouni/Documents/pucrs/sisrec/RestaurantRecommenderSystem/data/userprofile.csv')
    return ratings, user_cuisine, restaurant_cuisine, restaurant_info, restaurant_parking, user_profile

ratings, user_cuisine, restaurant_cuisine, restaurant_info, restaurant_parking, user_profile = load_data()

# Data Preprocessing: Remove users and items with very few ratings to reduce sparsity
# Set thresholds
min_user_ratings = 1
min_item_ratings = 1

# Filter users
user_counts = ratings['userID'].value_counts()
users_to_keep = user_counts[user_counts >= min_user_ratings].index
ratings_filtered = ratings[ratings['userID'].isin(users_to_keep)]

# Filter items
item_counts = ratings_filtered['placeID'].value_counts()
items_to_keep = item_counts[item_counts >= min_item_ratings].index
ratings_filtered = ratings_filtered[ratings_filtered['placeID'].isin(items_to_keep)]

# Prepare data for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_filtered[['userID', 'placeID', 'rating']], reader)

# Perform GridSearchCV for hyperparameter tuning
param_grid = {
    'n_factors': [50, 100, 150],
    'n_epochs': [20, 30, 40],
    'lr_all': [0.002, 0.005, 0.007],
    'reg_all': [0.02, 0.05, 0.1]
}

gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5, n_jobs=-1)
gs.fit(data)

# Get best RMSE score
best_rmse = gs.best_score['rmse']
# Get best hyperparameters
best_params = gs.best_params['rmse']

# Use the best model
best_model = gs.best_estimator['rmse']

# Build full trainset and retrain the best model on it
trainset = data.build_full_trainset()
best_model.fit(trainset)

# Since we have already used cross-validation in GridSearchCV, we can also perform additional cross-validation
cv_results = cross_validate(best_model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Prepare evaluation metrics
# Let's build a testset for evaluation
trainset, testset = train_test_split(data, test_size=0.3, random_state=42)
best_model.fit(trainset)
predictions = best_model.test(testset)
rmse_value = rmse(predictions)

# Extended Model Evaluation Function with Additional Metrics
def evaluate_model_extended(predictions):
    true_ratings = [pred.r_ui for pred in predictions]
    predicted_scores = [pred.est for pred in predictions]
    threshold = 4  # Define a threshold for positive rating

    # Calculate Root Mean Squared Error (RMSE)
    rmse_value = np.sqrt(mean_squared_error(true_ratings, predicted_scores))

    # Calculate Precision and Recall based on a binary threshold
    true_ratings_binary = [1 if rating >= threshold else 0 for rating in true_ratings]
    predicted_scores_binary = [1 if score >= threshold else 0 for score in predicted_scores]
    precision = precision_score(true_ratings_binary, predicted_scores_binary, zero_division=1)
    recall = recall_score(true_ratings_binary, predicted_scores_binary, zero_division=1)

    # Calculate Mean Average Precision (MAP)
    map_score = average_precision_score(true_ratings_binary, predicted_scores)

    # Calculate Hit Rate (proportion of actual items that were recommended)
    hits = sum([1 for true, pred in zip(true_ratings_binary, predicted_scores_binary) if true == 1 and pred == 1])
    num_users = len(set(pred.uid for pred in predictions))
    hit_rate = hits / num_users if num_users > 0 else 0

    return rmse_value, precision, recall, map_score, hit_rate

# Get evaluation metrics
rmse_value, precision, recall, map_score, hit_rate = evaluate_model_extended(predictions)

# Display Model Evaluation Metrics in Streamlit
st.sidebar.subheader("Model Evaluation Metrics")
st.sidebar.write(f"Best RMSE from GridSearchCV: {best_rmse:.4f}")
st.sidebar.write(f"Root Mean Squared Error (RMSE): {rmse_value:.4f}")
st.sidebar.write(f"Precision: {precision:.4f}")
st.sidebar.write(f"Recall: {recall:.4f}")
st.sidebar.write(f"Mean Average Precision: {map_score:.4f}")
st.sidebar.write(f"Hit Rate: {hit_rate:.4f}")

# Functions to generate recommendations

def recommend_for_existing_user(user_id, num_recommendations=5, rating_threshold=4.0):
    # Get a list of all items
    all_items = ratings_filtered['placeID'].unique()
    # Get items rated by the user
    rated_items = ratings_filtered[ratings_filtered['userID'] == user_id]['placeID'].unique()
    # Predict ratings for all items
    predictions = [best_model.predict(user_id, item_id) for item_id in all_items if item_id not in rated_items]
    # Filter predictions with estimated rating above the threshold
    high_rating_predictions = [pred for pred in predictions if pred.est >= rating_threshold]
    # Sort predictions
    sorted_predictions = sorted(high_rating_predictions, key=lambda x: x.est, reverse=True)[:num_recommendations]
    recommended_items = [(pred.iid, pred.est) for pred in sorted_predictions]
    return pd.DataFrame(recommended_items, columns=['placeID', 'predicted_rating'])

def recommend_new_item_for_existing_user(user_id, num_recommendations=1, rating_threshold=4.0):
    # Similar to recommend_for_existing_user but for a new item
    # Get items not rated by the user
    rated_items = ratings_filtered[ratings_filtered['userID'] == user_id]['placeID'].unique()
    unrated_items = [item for item in ratings_filtered['placeID'].unique() if item not in rated_items]
    # Predict ratings for unrated items
    predictions = [best_model.predict(user_id, item_id) for item_id in unrated_items]
    # Filter predictions with estimated rating above the threshold
    high_rating_predictions = [pred for pred in predictions if pred.est >= rating_threshold]
    # Sort predictions
    sorted_predictions = sorted(high_rating_predictions, key=lambda x: x.est, reverse=True)[:num_recommendations]
    recommended_items = [(pred.iid, pred.est) for pred in sorted_predictions]
    return pd.DataFrame(recommended_items, columns=['placeID', 'predicted_rating'])

def recommend_for_new_user(
    preferred_cuisines,
    preferred_ambience,
    price_range,
    accessibility,
    parking_required,
    activity,
    personality,
    num_recommendations=5
):
    # Start with all restaurants
    filtered_restaurants = restaurant_info.copy()

    # Filter by preferred cuisines
    if preferred_cuisines:
        # Get restaurants that serve the preferred cuisines
        cuisine_restaurants = restaurant_cuisine[restaurant_cuisine['Rcuisine'].isin(preferred_cuisines)]
        filtered_restaurants = filtered_restaurants[filtered_restaurants['placeID'].isin(cuisine_restaurants['placeID'])]

    # Filter by preferred ambience
    if preferred_ambience:
        filtered_restaurants = filtered_restaurants[filtered_restaurants['Rambience'] == preferred_ambience]

    # Filter by price range
    if price_range:
        filtered_restaurants = filtered_restaurants[filtered_restaurants['price'] == price_range]

    # Filter by accessibility
    if accessibility:
        filtered_restaurants = filtered_restaurants[filtered_restaurants['accessibility'] == accessibility]

    # Filter by parking requirement
    if parking_required:
        parking_places = restaurant_parking[restaurant_parking['parking_lot'] != 'none']
        filtered_restaurants = filtered_restaurants[filtered_restaurants['placeID'].isin(parking_places['placeID'])]

    # You can add more filters based on activity and personality if your data supports it

    # Since we don't have ratings from this new user, we can recommend the most popular restaurants
    # For simplicity, let's assume popularity is based on the number of ratings
    restaurant_popularity = ratings_filtered.groupby('placeID').size().reset_index(name='num_ratings')
    filtered_restaurants = filtered_restaurants.merge(restaurant_popularity, on='placeID', how='left')
    filtered_restaurants['num_ratings'] = filtered_restaurants['num_ratings'].fillna(0)

    # Sort restaurants by number of ratings (popularity) in descending order
    filtered_restaurants = filtered_restaurants.sort_values(by='num_ratings', ascending=False)

    # Get the top N recommendations
    recommendations = filtered_restaurants.head(num_recommendations)

    # Select relevant columns to display
    recommendations = recommendations[['name', 'placeID', 'num_ratings']]
    recommendations = recommendations.rename(columns={
        'name': 'Nome do Restaurante',
        'placeID': 'ID do Restaurante',
        'num_ratings': 'Popularidade'
    })

    return recommendations

# Streamlit UI
st.title("Sistema de Recomendação de Restaurantes")
st.sidebar.header("Seleção de Tipo de Usuário")

# Select user scenario
user_type = st.sidebar.radio("Escolha o tipo de recomendação:", 
                             ("Usuário Existente - Lista de Itens", "Novo Usuário - Lista de Itens", "Usuário Existente - Novo Item"))

# Scenario 1: Existing User - List of Items
if user_type == "Usuário Existente - Lista de Itens":
    user_id = st.sidebar.text_input("Digite seu ID de Usuário")
    num_recommendations = st.sidebar.slider("Número de Recomendações", min_value=1, max_value=10, value=5)
    rating_threshold = st.sidebar.slider("Nota Mínima Prevista", min_value=1.0, max_value=5.0, value=4.0, step=0.1)

    if st.sidebar.button("Obter Recomendações"):
        recommendations = recommend_for_existing_user(user_id=user_id, num_recommendations=num_recommendations, rating_threshold=rating_threshold)

        if not recommendations.empty:
            recommendations = recommendations.merge(restaurant_info[['placeID', 'name']], on='placeID', how='left')
            recommendations = recommendations.rename(columns={
                'name': 'Nome do Restaurante', 
                'placeID': 'ID do Restaurante', 
                'predicted_rating': 'Nota Prevista'
            })
            st.subheader(f"Top {num_recommendations} Recomendações para o Usuário {user_id}")
            st.table(recommendations[['Nome do Restaurante', 'ID do Restaurante', 'Nota Prevista']])
        else:
            st.write("Desculpe, não encontramos recomendações adequadas para você no momento.")

# Scenario 2: New User - List of Items
elif user_type == "Novo Usuário - Lista de Itens":
    st.sidebar.write("Informe suas preferências para obter recomendações")
    
    available_cuisines = sorted(restaurant_cuisine['Rcuisine'].unique())
    preferred_cuisines = st.sidebar.multiselect("Cozinhas Preferidas", options=available_cuisines)
    
    ambience_options = sorted(restaurant_info['Rambience'].dropna().unique())
    preferred_ambience = st.sidebar.selectbox("Ambiente Preferido", options=["Qualquer"] + ambience_options)
    preferred_ambience = None if preferred_ambience == "Qualquer" else preferred_ambience

    price_options = sorted(restaurant_info['price'].dropna().unique())
    price_range = st.sidebar.selectbox("Faixa de Preço Preferida", options=["Qualquer"] + price_options)
    price_range = None if price_range == "Qualquer" else price_range

    accessibility_options = sorted(restaurant_info['accessibility'].dropna().unique())
    accessibility = st.sidebar.selectbox("Acessibilidade", options=["Qualquer"] + accessibility_options)
    accessibility = None if accessibility == "Qualquer" else accessibility

    parking_required = st.sidebar.checkbox("Requer Estacionamento")
    activity = st.sidebar.selectbox("Atividade", options=["Qualquer", "Estudante", "Profissional"])
    activity = None if activity == "Qualquer" else activity

    personality = st.sidebar.selectbox("Personalidade", options=["Qualquer", "Extrovertido", "Reservado"])
    personality = None if personality == "Qualquer" else personality

    num_recommendations = st.sidebar.slider("Número de Recomendações", min_value=1, max_value=10, value=5)

    if st.sidebar.button("Obter Recomendações"):
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

        if not recommendations.empty:
            st.subheader(f"Top {num_recommendations} Recomendações Baseadas nas Suas Preferências")
            st.table(recommendations)
        else:
            st.write("Desculpe, não encontramos restaurantes que correspondam às suas preferências.")

# Scenario 3: Existing User - New Item
elif user_type == "Usuário Existente - Novo Item":
    user_id = st.sidebar.text_input("Digite seu ID de Usuário")
    num_recommendations = st.sidebar.slider("Número de Recomendações", min_value=1, max_value=5, value=1)
    rating_threshold = st.sidebar.slider("Nota Mínima Prevista", min_value=1.0, max_value=5.0, value=4.0, step=0.1)

    if st.sidebar.button("Obter Recomendações de Novo Item"):
        recommendations = recommend_new_item_for_existing_user(user_id=user_id, num_recommendations=num_recommendations, rating_threshold=rating_threshold)

        if not recommendations.empty:
            recommendations = recommendations.merge(restaurant_info[['placeID', 'name']], on='placeID', how='left')
            recommendations = recommendations.rename(columns={
                'name': 'Nome do Restaurante', 
                'placeID': 'ID do Restaurante', 
                'predicted_rating': 'Nota Prevista'
            })
            st.subheader(f"Recomendação de Novo Item para o Usuário {user_id}")
            st.table(recommendations[['Nome do Restaurante', 'ID do Restaurante', 'Nota Prevista']])
        else:
            st.write("Desculpe, não encontramos recomendações adequadas para você no momento.")