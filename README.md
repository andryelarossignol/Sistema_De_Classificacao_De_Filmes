# 🎬 Sistema de Classificação de Filmes

Este é um sistema web simples construído com Flask que utiliza um modelo de machine learning (Naive Bayes) para prever a classificação de filmes com base em seu **ano**, **duração** e **classificação MPA (Motion Picture Association)**.

---

## 🚀 Como rodar o projeto localmente

### ✅ Pré-requisitos

- Python 3.8 ou superior
- `pip` instalado (gerenciador de pacotes do Python)

---

### 🛠️ Passos para execução

1. **Clone o repositório:**

```bash
git clone https://github.com/andryelarossignol/Sistema_De_Classificacao_De_Filmes.git
cd Sistema_De_Classificacao_De_Filmes
```
 2. **Crie um ambiente virtual:**
```bash
python -m venv venv
```
 3. **Ative o ambiente virtual:**
```bash
venv\Scripts\activate
```
 4. **Instale as dependências:**
```bash
pip install flask pandas scikit-learn
```
 5. **Verifique se os seguintes arquivos estão presentes:**
    
    modelo_naive_bayes.pkl

    label_encoders.pkl

    data/top_movies_real.csv
    
 6. **Execute o servidor Flask:**
```bash
python app.py
```

 6. **Acesse a aplicação no navegador:**
```bash
http://127.0.0.1:5000
```
