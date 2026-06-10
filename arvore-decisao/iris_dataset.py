# Importação das bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Importação e exploração dos dados
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names) # Características (Features)
y = iris.target # Rótulos (Classes: 0=setosa, 1=versicolor, 2=virginica)

# Separação em conjunto de Treinamento (70%) e Teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicialização e Treinamento do Modelo
# Usamos max_depth=3 para evitar overfitting e manter as regras legíveis
modelo_arvore = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
modelo_arvore.fit(X_train, y_train)

# Previsão e Avaliação com Métricas
y_pred = modelo_arvore.predict(X_test)

# Como a base Iris tem 3 classes, usamos average='macro' para calcular a média das métricas
acuracia = accuracy_score(y_test, y_pred)
precisao = precision_score(y_test, y_pred, average='macro')
revocacao = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("--- MÉTRICAS DE AVALIAÇÃO ---")
print(f"Acurácia:  {acuracia:.4f}")
print(f"Precisão:  {precisao:.4f}")
print(f"Revocação: {revocacao:.4f}")
print(f"F1-Score:  {f1:.4f}\n")

# Extração e Exibição das Regras Geradas
print("--- REGRAS DE DECISÃO GERADAS ---")
regras = export_text(modelo_arvore, feature_names=list(iris.feature_names))
print(regras)

# Representação Gráfica da Árvore
plt.figure(figsize=(12, 8))
plot_tree(modelo_arvore, 
          feature_names=iris.feature_names,  
          class_names=iris.target_names,
          filled=True, 
          rounded=True,
          fontsize=10)
plt.title("Árvore de Decisão - Iris Dataset")
plt.show()