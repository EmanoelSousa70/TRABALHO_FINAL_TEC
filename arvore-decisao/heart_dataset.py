# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Importação e tratamento dos dados (Repositório UCI - Cleveland)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
colunas = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Carrega os dados tratando as interrogações '?' como valores nulos (NaN)
df = pd.read_csv(url, names=colunas, na_values='?')

# Remove as linhas com valores nulos (uma boa prática para manter a consistência)
df = df.dropna()

# Converte o alvo para classificação binária: 0 (Saudável) e 1 (Doente)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Separação das características (X) e do alvo (y)
X = df.drop('target', axis=1)
y = df['target']

# Nome das classes para exibição posterior
class_names = ['Saudável', 'Doente']

# Separação em conjunto de Treinamento (70%) e Teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicialização e Treinamento do Modelo
# Usamos max_depth=3 para manter a árvore altamente interpretável por médicos
modelo_arvore = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
modelo_arvore.fit(X_train, y_train)

# Previsão e Avaliação com Métricas
y_pred = modelo_arvore.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
precisao = precision_score(y_test, y_pred, average='binary')
revocacao = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print("--- MÉTRICAS DE AVALIAÇÃO ---")
print(f"Acurácia:  {acuracia:.4f}")
print(f"Precisão:  {precisao:.4f}")
print(f"Revocação: {revocacao:.4f}")
print(f"F1-Score:  {f1:.4f}\n")

# Extração e Exibição das Regras Geradas
print("--- REGRAS DE DECISÃO GERADAS ---")
regras = export_text(modelo_arvore, feature_names=list(X.columns))
print(regras)

# Representação Gráfica da Árvore
plt.figure(figsize=(15, 9))
plot_tree(modelo_arvore, 
          feature_names=X.columns,  
          class_names=class_names,
          filled=True, 
          rounded=True,
          fontsize=10)
plt.title("Árvore de Decisão - Heart Disease Dataset")
plt.tight_layout()
plt.show()