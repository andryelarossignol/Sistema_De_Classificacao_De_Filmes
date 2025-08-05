# 🎬 Sistema de Classificação de Filmes

Este é um sistema web simples construído com Flask que utiliza um modelo de machine learning (Naive Bayes) para prever a classificação de filmes com base em seu **ano**, **duração** e **classificação MPA (Motion Picture Association)**.

---

<img width="1920" height="1080" alt="1" src="https://github.com/user-attachments/assets/5247be29-9bfe-4d62-a03d-ccd57138ca8e" />

<img width="1920" height="1080" alt="2" src="https://github.com/user-attachments/assets/4847825d-2d14-409c-9a40-4b78911ea1b6" />

<img width="1920" height="1080" alt="3" src="https://github.com/user-attachments/assets/712e452c-0281-4317-9ac1-56e5033fb8e0" />

<img width="1920" height="1080" alt="4" src="https://github.com/user-attachments/assets/b615415b-c4a2-445f-aba6-d6362c025b63" />

<img width="1920" height="1080" alt="5" src="https://github.com/user-attachments/assets/b58e79dc-332f-4663-aa1e-bd5588bbb8b7" />

<img width="1920" height="1080" alt="6" src="https://github.com/user-attachments/assets/b6f31772-bf91-4d8f-8b54-fb532ebecb09" />

---

## 🚀 Rodar o projeto localmente

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
