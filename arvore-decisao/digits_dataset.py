# Importação das bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Importação e exploração dos dados
digits = load_digits()
# Cria colunas nomeadas de pixel_0_0 até pixel_7_7 para facilitar a leitura das regras
feature_names = [f"pixel_{i}_{j}" for i in range(8) for j in range(8)]
X = pd.DataFrame(digits.data, columns=feature_names) # 64 pixels (características)
y = digits.target # Rótulos (Classes de 0 a 9)

# Separação em conjunto de Treinamento (70%) e Teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicialização e Treinamento do Modelo
# Aumentamos o max_depth para 6 para dar conta das 10 classes do problema
modelo_arvore = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=42)
modelo_arvore.fit(X_train, y_train)

# Previsão e Avaliação com Métricas
y_pred = modelo_arvore.predict(X_test)

# 'macro' calcula o desempenho individual de cada dígito e tira a média
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
print("--- REGRAS DE DECISÃO GERADAS (Primeiras linhas para análise) ---")
regras = export_text(modelo_arvore, feature_names=feature_names)
# Exibindo apenas os primeiros 1000 caracteres para não inundar o terminal
print(regras[:1000] + "\n... [Texto cortado para fins de leitura] ...")

# Representação Gráfica da Árvore
plt.figure(figsize=(20, 10)) # Dimensões largas devido à expansão horizontal dos ramos
plot_tree(modelo_arvore, 
          feature_names=feature_names,  
          class_names=[str(i) for i in digits.target_names], 
          filled=True, 
          rounded=True,
          fontsize=8)
plt.title("Árvore de Decisão - Digits Dataset (Profundidade 6)")
plt.tight_layout()
plt.show()