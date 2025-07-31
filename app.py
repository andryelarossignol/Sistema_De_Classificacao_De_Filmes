from flask import Flask, render_template, request
from flask import Flask, render_template, request, session
import joblib
import pandas as pd
import re

# Inicializa a aplicação Flask
app = Flask(__name__)

# Configura a chave secreta para sessões
app.secret_key = 'previsao'

# Carrega o modelo treinado
model = joblib.load('modelo_naive_bayes.pkl')

# Carrega os encoders de label
label_encoders = joblib.load('label_encoders.pkl')
le_target = joblib.load('target_encoder.pkl')

# Obtém os valores válidos diretamente dos encoders treinados
ratings = sorted(label_encoders['rating_mpa'].classes_)
genres = sorted(label_encoders['genre_main'].classes_)
countries = sorted(label_encoders['country_origin'].classes_)
languages = sorted(label_encoders['language'].classes_)


# Página inicial com o formulário
@app.route('/')
def index():

    retorno = request.args.get('retorno')
    if retorno:
        dados = session.get('dados_preenchidos', {})
    else:
        # Se foi acesso normal (como atualização), limpa a sessão
        session.pop('dados_preenchidos', None)
        dados = {}


    return render_template('index.html',
                           ratings=ratings,
                           genres=genres,
                           countries=countries,
                           languages=languages,
                           dados_preenchidos=dados)

# Rota que trata a previsão
@app.route('/predict', methods=['POST'])
def predict():
    # Verifica se o formulário foi preenchido corretament
    session['dados_preenchidos'] = request.form.to_dict()

    try:
        year = int(request.form['year'])
    except ValueError:
        return render_template('result.html', erro=True, mensagem="Formato inválido para o ano.")

    # O modelo foi treinado com dados de 1960 a 2024.
    # Permitimos anos até 2026 para prever filmes que ainda serão lançados.
    if year < 1960 or year > 2026:
        return render_template('result.html', erro=True, mensagem="Ano fora do intervalo permitido.")

    duration_min = int(request.form['duration'])
    rating_mpa = request.form['rating_mpa']
    genre = request.form['genre'].split(',')[0].strip()
    country = request.form['country']
    language = request.form['language'].split(',')[0].strip()

    # Mostra os valores brutos recebidos do formulário
    print("Valores recebidos do formulário:")
    print("Ano:", year)
    print("Duração (min):", duration_min)
    print("Classificação MPA:", rating_mpa)
    print("Gênero:", genre)
    print("País de origem:", country)
    print("Idioma:", language)

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
