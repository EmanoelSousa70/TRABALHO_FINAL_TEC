# Importação das bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#  Importação e exploração dos dados
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names) # 13 características químicas
y = wine.target # Rótulos (Classes: 0, 1 ou 2, representando os tipos de cultivo)

# Separação em conjunto de Treinamento (70%) e Teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicialização e Treinamento do Modelo
# Fixamos max_depth=3 para manter a árvore interpretável
modelo_arvore = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
modelo_arvore.fit(X_train, y_train)

# Previsão e Avaliação com Métricas
y_pred = modelo_arvore.predict(X_test)

# Cálculo das métricas exigidas pelo trabalho
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
regras = export_text(modelo_arvore, feature_names=list(wine.feature_names))
print(regras)

# Representação Gráfica da Árvore
plt.figure(figsize=(14, 8))
plot_tree(modelo_arvore, 
          feature_names=wine.feature_names,  
          class_names=wine.target_names, # ['class_0', 'class_1', 'class_2']
          filled=True, 
          rounded=True,
          fontsize=10)
plt.title("Árvore de Decisão - Wine Dataset")
plt.tight_layout()
plt.show()