# Importação das bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Importação e exploração dos dados
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names) # 30 características do tumor
y = cancer.target # Rótulos (Classes: 0 = Maligno, 1 = Benigno)

# Separação em conjunto de Treinamento (70%) e Teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicialização e Treinamento do Modelo
# Usamos max_depth=4 pois temos 30 variáveis, permitindo um pouco mais de complexidade
modelo_arvore = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
modelo_arvore.fit(X_train, y_train)

# Previsão e Avaliação com Métricas
y_pred = modelo_arvore.predict(X_test)

# Mantemos o average='macro' para extrair a média justa entre as classes Maligno e Benigno
acuracia = accuracy_score(y_test, y_pred)
precisao = precision_score(y_test, y_pred, average='macro')
revocacao = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("--- MÉTRICAS DE AVALIAÇÃO ---")
print(f"Acurácia:  {acuracia:.4f}")
print(f"Precisão:  {precisao:.4f}")
print(f"Revocação: {revocacao:.4f}")
print(f"F1-Score:  {f1:.4f}\n")

# Extração e Exibição das Regras Geradas (Requisito do Projeto)
print("--- REGRAS DE DECISÃO GERADAS ---")
regras = export_text(modelo_arvore, feature_names=list(cancer.feature_names))
print(regras)

# Representação Gráfica da Árvore
plt.figure(figsize=(16, 10)) # Tamanho ampliado devido à maior quantidade de nós
plot_tree(modelo_arvore, 
          feature_names=cancer.feature_names,  
          class_names=cancer.target_names, # ['malignant', 'benign']
          filled=True, 
          rounded=True,
          fontsize=9)
plt.title("Árvore de Decisão - Breast Cancer Wisconsin")
plt.tight_layout()
plt.show()
