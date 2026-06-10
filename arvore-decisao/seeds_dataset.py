# Importação das bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Importação e tratamento dos dados (Repositório UCI)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
colunas = [
    'area', 'perimeter', 'compactness', 'length_kernel', 
    'width_kernel', 'asymmetry_coefficient', 'length_groove', 'target'
]

# Carrega os dados tratando múltiplos espaços/tabs como separadores
df = pd.read_csv(url, sep=r'\s+', names=colunas)

# Remove possíveis linhas com valores nulos
df = df.dropna()

# Separação de características (X) e rótulos (y)
X = df.drop('target', axis=1)
y = df['target']

# Ajusta os rótulos originais (1, 2, 3) para a nomenclatura real das variedades
class_names = ['Kama', 'Rosa', 'Canadian']

# Separação em conjunto de Treinamento (70%) e Teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicialização e Treinamento do Modelo
# Mantemos max_depth=3 devido ao tamanho controlado da base (210 instâncias)
modelo_arvore = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
modelo_arvore.fit(X_train, y_train)

# Previsão e Avaliação com Métricas
y_pred = modelo_arvore.predict(X_test)

# Como são 3 classes, usamos average='macro'
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
regras = export_text(modelo_arvore, feature_names=list(X.columns))
print(regras)

# Representação Gráfica da Árvore
plt.figure(figsize=(14, 8))
plot_tree(modelo_arvore, 
          feature_names=X.columns,  
          class_names=class_names,
          filled=True, 
          rounded=True,
          fontsize=10)
plt.title("Árvore de Decisão - Seeds Dataset")
plt.tight_layout()
plt.show()