import pandas as pd
import numpy as np
# Assicurati che il percorso al modulo sia corretto.
# Se il modulo si chiama lda_parallel.py e contiene la funzione analyze_lda_parallel:
from .parallel_lda import analyze_lda_parallel

# 1. Carica i dati
df = pd.read_csv(r'C:\Users\andre\Desktop\LMT dim reduction toolkit\data\behavior_stats_intervals_to_analize\merged_analysis_behavior_stats_intervals.csv')

# 2. Prepara i dati
# Supponiamo che la colonna 'mouse_id' sia la label e le altre colonne siano feature.
y = df['mouse_id'].values
# Se vuoi escludere eventuali colonne non informative, ad esempio date o timestamp, rimuovile:
feature_columns = [col for col in df.columns if col not in ['mouse_id', 'interval_start']]
X = df[feature_columns].values

# 3. Esegui l'analisi LDA parallela
results = analyze_lda_parallel(X, y)

# 4. Visualizza i risultati
print("\nBasic Analysis Results:")
print("Numero di componenti stabili trovate:", results['n_components'])
print("Stability score medio:", results['stability_score'])
print("Eigenvalues delle componenti:", results['eigenvalues'])
print("Discriminative Power:", results['discriminative_power'])

# 5. Visualizza informazioni dettagliate sulle feature
print("\nDetailed Feature Analysis:")
feature_mask = results['feature_mask']
components = results['components']

# Ottieni feature selezionate
selected_features = [feat for feat, selected in zip(feature_columns, feature_mask) if selected]

# Per ogni componente
for comp_idx in range(components.shape[0]):
    print(f"\nComponent {comp_idx + 1} Features:")
    # Ottieni pesi delle feature per questo componente
    weights = components[comp_idx]
    # Crea coppie feature-peso e ordina per peso assoluto
    feature_weights = list(zip(selected_features, weights))
    feature_weights.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Stampa feature di maggior contributo (peso assoluto > 0.1)
    print("Top contributing features (|weight| > 0.1):")
    for feature, weight in feature_weights:
        if abs(weight) > 0.1:
            print(f"{feature}: {weight:.3f}")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap

# Generate 28 unique colors using a colormap (e.g., 'hsv' or 'tab20b')
colors = cm.get_cmap('tab20b', 28).colors  # Generates 28 unique colors from the 'tab20b' colormap

# Map each mouse_id to a unique color
unique_ids = np.unique(y)
color_dict = {mouse_id: colors[i] for i, mouse_id in enumerate(unique_ids)}

# Assign colors to each point based on mouse_id
point_colors = [color_dict[val] for val in y]

