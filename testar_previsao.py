import joblib
import pandas as pd

# 1. Carregar modelo e encoders
model = joblib.load("modelo_naive_bayes.pkl")
label_encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# 2. Verificar os valores únicos dos encoders
# print(label_encoders['rating_mpa'].classes_)
# print(label_encoders['genre_main'].classes_)
# print(label_encoders['country_origin'].classes_)
# print(label_encoders['language'].classes_)


# 2. Novo filme a ser testado (personalize aqui)
# OBS: Esse código só vai funcionar se os valores usados existirem nos encoders, ou seja:
# O idioma, país, gênero e rating MPA têm que ter aparecido na base de treino.)

# novo_filme = {
#     'year': 2025,
#     'duration_min': 110,
#     'rating_mpa': 'G',
#     'genre_main': 'Action',
#     'country_origin': 'West Germany, Spain',
#     'language': 'English'
# }

# novo_filme = {
#     'year': 1975,
#     'duration_min': 105,
#     'rating_mpa': 'Not Rated',
#     'genre_main': 'Comedy',
#     'country_origin': 'Denmark',
#     'language': 'Danish'
# }

novo_filme = {
    'year': 2024,
    'duration_min': 106,
    'rating_mpa': 'Not Rated',
    'genre_main': 'Drama',
    'country_origin': 'United States',
    'language': 'English'
}




# 3. Validar e codificar os campos categóricos
for col in ['rating_mpa', 'genre_main', 'country_origin', 'language']:
    encoder = label_encoders[col]
    if novo_filme[col] not in encoder.classes_:
        raise ValueError(f"Valor '{novo_filme[col]}' para '{col}' não foi visto durante o treinamento.\nValores possíveis: {list(encoder.classes_)}")
    novo_filme[col] = encoder.transform([novo_filme[col]])[0]

# 4. Montar DataFrame com colunas corretas
entrada_df = pd.DataFrame([{
    'year': novo_filme['year'],
    'duration_min': novo_filme['duration_min'],
    'rating_mpa': novo_filme['rating_mpa'],
    'genre_main': novo_filme['genre_main'],
    'country_origin': novo_filme['country_origin'],
    'language': novo_filme['language']
}])

# 5. Fazer a previsão
predicao = model.predict(entrada_df)[0]
classe_prevista = target_encoder.inverse_transform([predicao])[0]

print(f"Classe prevista: {classe_prevista}")
