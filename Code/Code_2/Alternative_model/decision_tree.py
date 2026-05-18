import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import os

OUTPUT_DIR = '../Report_2/images'

def prepare_data(pokemon_path, combats_path):
    print("Wczytywanie i przygotowywanie danych...")
    # Load data
    pokemon = pd.read_csv(pokemon_path)
    combats = pd.read_csv(combats_path)

    # Set index to Pokemon ID for easy lookup
    pokemon.set_index('#', inplace=True)

    # Pre-allocate arrays for performance
    n_samples = len(combats)
    diff_speed = np.zeros(n_samples)
    diff_attack = np.zeros(n_samples)
    diff_defense = np.zeros(n_samples)
    diff_hp = np.zeros(n_samples)
    diff_sp_atk = np.zeros(n_samples)
    diff_sp_def = np.zeros(n_samples)
    diff_legendary = np.zeros(n_samples)
    targets = np.zeros(n_samples)

    # Vectorized operations are much faster than iterrows
    p1_ids = combats['First_pokemon'].values
    p2_ids = combats['Second_pokemon'].values
    
    p1_stats = pokemon.loc[p1_ids]
    p2_stats = pokemon.loc[p2_ids]
    
    diff_speed = p1_stats['Speed'].values - p2_stats['Speed'].values
    diff_attack = p1_stats['Attack'].values - p2_stats['Attack'].values
    diff_defense = p1_stats['Defense'].values - p2_stats['Defense'].values
    diff_hp = p1_stats['HP'].values - p2_stats['HP'].values
    diff_sp_atk = p1_stats['Sp. Atk'].values - p2_stats['Sp. Atk'].values
    diff_sp_def = p1_stats['Sp. Def'].values - p2_stats['Sp. Def'].values
    diff_legendary = p1_stats['Legendary'].astype(int).values - p2_stats['Legendary'].astype(int).values

    # Target: 1 if First_pokemon won, 0 if Second_pokemon won
    targets = (combats['Winner'].values == combats['First_pokemon'].values).astype(int)

    features = pd.DataFrame({
        'diff_Speed': diff_speed,
        'diff_Attack': diff_attack,
        'diff_Defense': diff_defense,
        'diff_HP': diff_hp,
        'diff_Sp_Atk': diff_sp_atk,
        'diff_Sp_Def': diff_sp_def,
        'diff_Legendary': diff_legendary
    })
    
    return features, targets

def evaluate_model(model, X_test, y_test, name):
    print(f"\n{'='*40}")
    print(f"Ewaluacja modelu: {name}")
    print(f"{'='*40}")
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy:  {acc:.4f} (wymagane: >= 0.80)")
    print(f"F1-Score:  {f1:.4f} (wymagane: >= 0.75)")
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Wygra P2 (0)', 'Wygra P1 (1)'],
                yticklabels=['Wygra P2 (0)', 'Wygra P1 (1)'])
    plt.title(f'Macierz pomyłek - {name}')
    plt.ylabel('Rzeczywistość')
    plt.xlabel('Predykcja')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/cm_{name.replace(" ", "_")}.png')
    plt.close()

def plot_decision_tree(tree_model, feature_names):
    plt.figure(figsize=(20, 10))
    # We limit depth for visualization, otherwise it's unreadable
    plot_tree(tree_model, max_depth=3, feature_names=feature_names, 
              class_names=['P2 Wins', 'P1 Wins'], filled=True, rounded=True, fontsize=10)
    plt.title("Fragment drzewa decyzyjnego (max_depth=3)")
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/decision_tree_vis.png', dpi=300)
    plt.close()
    print(f"Wizualizacja drzewa zapisana do {OUTPUT_DIR}/decision_tree_vis.png")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pokemon_file = '../../../Dataset/pokemon.csv'
    combats_file = '../../../Dataset/combats.csv'
    
    if not os.path.exists(pokemon_file) or not os.path.exists(combats_file):
        print("Nie znaleziono plików w folderze Dataset!")
        return

    # 1. Prepare data
    X, y = prepare_data(pokemon_file, combats_file)
    
    print(f"Dane przygotowane. Liczba obserwacji: {len(X)}")
    print(f"Dostępne cechy: {list(X.columns)}")
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Experiment with tree depth (max_depth)
    depths = [3, 5, 10, 15, None]
    results = []
    
    print("\nBadanie wpływu parametru max_depth na wyniki Decision Tree...")
    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42, criterion='gini')
        dt.fit(X_train, y_train)
        preds = dt.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        results.append({
            'max_depth': 'Bez ograniczenia' if d is None else str(d),
            'Accuracy': acc,
            'F1-score': f1
        })
    
    results_df = pd.DataFrame(results)
    print("\nWyniki dla różnych wartości max_depth:")
    print(results_df.to_string(index=False))
    
    # Train optimal model (max_depth=5) for evaluation and visualization
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42, criterion='gini')
    dt_model.fit(X_train, y_train)
    
    evaluate_model(dt_model, X_test, y_test, "Decision Tree (max_depth=5)")
    plot_decision_tree(dt_model, X.columns.tolist())
    
    # Feature importance for decision tree
    dt_importances = pd.Series(dt_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nWażność cech (Decision Tree):")
    print(dt_importances)

if __name__ == "__main__":
    main()
