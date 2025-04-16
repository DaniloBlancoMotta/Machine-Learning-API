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


