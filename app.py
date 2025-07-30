from flask import Flask, render_template, request
import joblib
import pandas as pd
import re

# Inicializa a aplicação Flask
app = Flask(__name__)

# Carrega o modelo treinado
model = joblib.load('modelo_naive_bayes.pkl')

# Carrega os encoders de label
label_encoders = joblib.load('label_encoders.pkl')
le_target = joblib.load('target_encoder.pkl')

# Carrega os dados reais para popular os campos do formulário
df = pd.read_csv('data/world_imdb_movies_top_movies_per_year.csv')

# Regex para capturar padrões válidos de duração como "1h 30min"
def is_valid_duration(d):
    return isinstance(d, str) and re.match(r'^\d{1,2}h(\s?\d{1,2}m(in)?)?$', d.strip().lower())

# Prepara listas únicas e ordenadas para os selects
durations = sorted(set(d for d in df['duration'].dropna() if is_valid_duration(d)))
ratings = sorted(df['rating_mpa'].dropna().unique())
ratings = [r for r in ratings if str(r).strip().lower() != 'nan' and str(r).strip() != '']
genres = sorted(df['genre'].dropna().unique())
countries = (
    df['country_origin']
    .dropna()
    .apply(lambda x: [c.strip() for c in x.split(',')])
    .explode()
    .dropna()
    .unique()
)
# Converte para lista e ordena
countries = sorted(countries)
languages = sorted(df['language'].dropna().unique())

# Página inicial com o formulário
@app.route('/')
def index():
    return render_template('index.html',
                           durations=durations,
                           ratings=ratings,
                           genres=genres,
                           countries=countries,
                           languages=languages)

# Rota que trata a previsão
@app.route('/predict', methods=['POST'])
def predict():
    try:
        year = int(request.form['year'])
    except ValueError:
        return render_template('result.html', erro=True, mensagem="Formato inválido para o ano.")

    # Valida o ano
    if year < 1960 or year > 2024:
        return render_template('result.html', erro=True, mensagem="Ano fora do intervalo permitido.")

    duration = request.form['duration']
    rating_mpa = request.form['rating_mpa']
    genre = request.form['genre'].split(',')[0].strip()
    country = request.form['country']
    language = request.form['language'].split(',')[0].strip()


    # Converte a duração para minutos
    def duration_to_minutes(dur):
        h = re.search(r"(\d+)h", dur)
        m = re.search(r"(\d+)m", dur)
        total = 0
        if h:
            total += int(h.group(1)) * 60
        if m:
            total += int(m.group(1))
        return total

    duration_min = duration_to_minutes(duration)

    # Codifica os campos com os LabelEncoders
    try:
        rating_mpa_encoded = label_encoders['rating_mpa'].transform([rating_mpa])[0]
        genre_encoded = label_encoders['genre_main'].transform([genre])[0]
        country_encoded = label_encoders['country_origin'].transform([country])[0]
        language_encoded = label_encoders['language'].transform([language])[0]
    except KeyError as e:
        return render_template('result.html', erro=True, mensagem=f"Erro na codificação: {e}")

    #  Nomes exatos como foram usados no treino!
    dados = pd.DataFrame([[year, duration_min, rating_mpa_encoded,
                           genre_encoded, country_encoded, language_encoded]],
                         columns=['year', 'duration_min', 'rating_mpa',
                                  'genre_main', 'country_origin', 'language'])

    # Previsão
    resultado = model.predict(dados)[0]

    return render_template('result.html', resultado=le_target.inverse_transform([resultado])[0])

# Inicia o servidor Flask
if __name__ == '__main__':
    app.run(debug=True)
