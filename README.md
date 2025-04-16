# Machine-Learning-API
Machine Learning API for Salary Prediction  \ API de Machine Learning para Previsão de Salário
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Carrega o dataset
df = pd.read_csv("Salary-2.csv")
print("Dimensão do dataset:", df.shape)

# Define as variáveis preditoras (X) e a variável alvo (y)
X = df[['YearsExperience']]
y = df['Salary']

# Cria e treina o modelo de regressão linear
model = LinearRegression()
model.fit(X, y)

# Mostra o desempenho do modelo (R²)
print("R² do modelo:", model.score(X, y))

# Salva o modelo treinado em um arquivo .pkl
with open("modelo_salario.pkl", "wb") as f:
    pickle.dump(model, f)

from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Carrega o modelo treinado
with open("modelo_salario.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    # Lê os dados enviados em JSON
    data = request.get_json(force=True)
    
    # Utiliza a primeira chave do JSON, independente do nome, para extrair o valor
    key = list(data.keys())[0]
    try:
        value = float(data[key])
    except ValueError:
        return jsonify({"error": f"Years of experience of user {key} must be a numeric!"}), 400

    if value < 0:
        return jsonify({"error": f"Years of experience of user {key} must be nonnegative!"}), 400

    # Realiza a previsão utilizando o modelo
    prediction = model.predict(np.array([[value]]))
    # Retorna o resultado arredondado para duas casas decimais
    return jsonify({"Salary": round(prediction[0], 2)})

if __name__ == "__main__":
    app.run(debug=True)

 Flask==2.2.5
pandas==1.5.3
numpy==1.23.5
scikit-learn==1.2.2

# Machine Learning API for Salary Prediction / API de Machine Learning para Previsão de Salário

This project implements a machine learning solution to predict annual salary based on years of experience. It includes:
- A **training script** (`train.py`) that loads a CSV dataset, trains a linear regression model using scikit-learn, and saves the trained model.
- A **Flask-based API** (`app.py`) that receives HTTP POST requests with experience data (JSON format) and returns salary predictions.

Este projeto implementa uma solução de machine learning para prever o salário anual com base nos anos de experiência. Ele inclui:
- Um **script de treinamento** (`train.py`) que lê um dataset CSV, treina um modelo de regressão linear usando scikit-learn e salva o modelo treinado.
- Uma **API em Flask** (`app.py`) que recebe requisições POST com dados de experiência (formato JSON) e retorna a previsão do salário.

## Repository Structure / Estrutura do Repositório



