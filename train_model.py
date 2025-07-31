import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re

# 1. Carregar a base de dados
df = pd.read_csv("data/world_imdb_movies_top_movies_per_year.csv")

# 2. Criar a coluna 'popularidade'
df = df[df['rating_imdb'].notna()]
df['popularidade'] = df['rating_imdb'].apply(lambda x: 'Popular' if x >= 7.0 else 'Nao_Popular')


# Verificar quantos filmes são populares e não populares
print("Distribuição de popularidade na base:")
print(df['popularidade'].value_counts())


# 3. Converter duração para minutos
def duration_to_minutes(dur):
    if isinstance(dur, str):
        h = re.search(r"(\d+)h", dur)
        m = re.search(r"(\d+)m", dur)
        total = 0
        if h:
            total += int(h.group(1)) * 60
        if m:
            total += int(m.group(1))
        return total
    return np.nan

df['duration_min'] = df['duration'].apply(duration_to_minutes)

# 4. Extrair o primeiro gênero da lista
df['genre_main'] = df['genre'].apply(lambda x: x.split(',')[0].strip() if pd.notna(x) else 'Unknown')

# 4.1 Extrair o primeiro idioma da lista
df['language'] = df['language'].apply(lambda x: x.split(',')[0].strip() if pd.notna(x) else 'Unknown')

# 4.2 Extrair o primeiro pais da lista
df['country_origin'] = df['country_origin'].apply(lambda x: x.split(',')[0].strip() if pd.notna(x) else 'Unknown')

# Remove linhas com valores 'Unknown' (dados ausentes em colunas categóricas)
for col in ['genre_main', 'language', 'country_origin', 'rating_mpa']:
    df = df[df[col] != 'Unknown']

# Especificamente para rating_mpa, remove 'Not Rated' e 'Unrated'
df = df[~df['rating_mpa'].isin(['Not Rated', 'Unrated'])]

# 5. Selecionar apenas as colunas úteis
df_model = df[['year', 'duration_min', 'rating_mpa', 'genre_main', 'country_origin', 'language', 'popularidade']]
df_model = df_model.dropna()

# 6. Codificar variáveis categóricas
le_cols = ['rating_mpa', 'genre_main', 'country_origin', 'language']
label_encoders = {}
for col in le_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

# 7. Codificar a variável alvo
le_target = LabelEncoder()
df_model['popularidade'] = le_target.fit_transform(df_model['popularidade'])

# 8. Separar features e alvo
X = df_model.drop('popularidade', axis=1)
y = df_model['popularidade']

# 9. Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 10. Treinar o modelo
model = GaussianNB()
model.fit(X_train, y_train)

# 11. Avaliar o modelo
y_pred = model.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# 11.1 Ver entradas que o modelo previu como Popular
y_pred_labels = le_target.inverse_transform(y_pred)
y_test_labels = le_target.inverse_transform(y_test)

# Montar DataFrame de resultados
resultados = X_test.copy()
resultados['previsto'] = y_pred_labels
resultados['real'] = y_test_labels

# Decodificar cada coluna com proteção contra erro
for col in ['rating_mpa', 'genre_main', 'country_origin', 'language']:
    encoder = label_encoders[col]
    col_array = resultados[col].values
    try:
        resultados[col] = encoder.inverse_transform(col_array)
    except:
        resultados[col] = [encoder.classes_[val] if val < len(encoder.classes_) else 'Unknown' for val in col_array]


# Filtrar previsões de Popular (já com texto)
populares_preditos = resultados[resultados['previsto'] == 'Popular']

# Mostrar os 5 primeiros
print("Exemplos classificados como 'Popular':")
print(populares_preditos.head())

# 12. Salvar modelo e encoders
joblib.dump(model, "modelo_naive_bayes.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(le_target, "target_encoder.pkl")
