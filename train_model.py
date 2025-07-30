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
df['genre_main'] = df['genre'].astype(str).apply(lambda x: x.split(',')[0].strip() if x else 'Unknown')

# 4.1 Extrair o primeiro idioma da lista
df['language'] = df['language'].astype(str).apply(lambda x: x.split(',')[0].strip() if x else 'Unknown')

# 5. Selecionar apenas as colunas úteis (sem 'vote')
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

# 12. Salvar modelo e encoders
joblib.dump(model, "modelo_naive_bayes.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(le_target, "target_encoder.pkl")