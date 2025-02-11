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
print("Numero di componenti stabili trovate:", results['n_components'])
print("Stability score medio:", results['stability_score'])
print("Eigenvalues delle componenti:", results['eigenvalues'])
print("Componenti (Identity Domains):", results['components'])
print("Discriminative Power:", results['discriminative_power'])
print("Feature mask:", results['feature_mask'])
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

