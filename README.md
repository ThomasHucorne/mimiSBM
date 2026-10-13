# Modèle à blocs stochastiques multicouches pour l'étude du devenir fonctionnel de gènes dupliqués chez Arabidopsis thaliana

L’objectif du projet est d’exploiter les différentes sources d’information à l’aide d’un modèle de clustering multivues basé sur un mélange de modèles de blocs stochastiques multicouches.

**Etapes** :
- **Réduction dimensionnelle** : Réduire l'espace de 30+ variables à une représentation 2D compréhensible via t-SNE
- **Visualisation** : Générer des graphiques exploratoires montrant la séparation des classes dans l'espace réduit
- **K-means** : Implémentation d'une méthode de clustering de manière à séparer en 2 classes chaque descripteur, permettant de trouver un seuil pour derrière récupérer une matrice d'ajdacence par descripteur défini par ce seuil.
- **Matrice d'adjacence** : 
- **Classification** : Analyser les paires de gènes selon leur statut de duplication (classe 0 : redondant fonctionnellement, classe 1 : divergent fonctionnellement)


Le notebook t_SNE inclut :
1. **Chargement & exploration** des données LAMME
2. **Nettoyage & prétraitement** : imputation des valeurs manquantes, one-hot encoding des variables catégorielles
3. **PCA pré-réduction** : réduire le bruit et les dimensions avant t-SNE (95% variance conservée)
4. **t-SNE** : projection 2D avec analyse de sensibilité à la perplexité
5. **Visualisation** : coloration par classe, environnement TE, et variables continues
6. **Export** : sauvegarde des coordonnées t-SNE pour analyses ultérieures

Le notebook kmeans inclut :
1. **Chargement** des données nettoyées
2. **Identification** des descripteurs numériques et booléens
3. **K-Means** par descripteur et détection du seuil
4. **Tableau sommaire** de l'ensemble des seuils et des informations du cluster
5. **Visualisation** de la distribution et du cluster par descripteur
7. **Exportation** des informations statistiques des clusters par descripteur sous forme de dataset
---

## Structure du Projet

```
.
├── README.md
├── t_SNE.ipynb
├── t_SNE_train_clean.png
├── kmeans.ipynb
│
├── Données/
│   ├── lamme_training.csv             # Dataset d'entraînement (paires de gènes)
│   ├── lamme_prediction.csv           # Dataset de prédiction
│   ├── X_train_clean.csv              # Features nettoyées (training)
│   ├── X_pred_clean.csv               # Features nettoyées (prediction)
│   ├── lamme_tsne_coords.csv          # Sortie : coordonnées t-SNE calculées
│   └── dataset_samuel.csv             # Données complémentaires
|
├── Plots/
|   ├── kmeans                         # Visualisation des figures récupérés par rapport au K-Means
|       ├── cluster_correlation_heatmap.png
|       ├── kmeans_gallery.png
|       ├── silhouette_barplot.png
|   ├── lamme_training/                 # Visualisation t-SNE sur le dataset lamme_training.csv
|       ├── correlation_matrix.png
|       ├── family_size_distribution.png
|       ├── pca_variance_explained.png
|       ├── tsne.png
|   ├── train_clean                    # Visualisation t-SNE sur le dataset X_train_clean.csv
|       ├── correlation_matrix.png
|       ├── family_size_distribution.png
|       ├── pca_variance_explaine.png
|       ├── tsne_overlays.png
|       ├── tsne.png
└── .venv/                             # Environnement virtuel Python (ignoré dans un possible git)
```

---

## Installation & Configuration

### Prérequis
- **Python** 3.9+ (3.11 pour mon cas mais à voir plus tard)
- **pip** ou **conda**

### Étape 1 : Créer l'environnement virtuel

```bash
# Créer un environnement virtuel
python -m venv .venv

# Activer l'environnement
source .venv/bin/activate  # Linux/macOS
# ou
.venv\Scripts\activate     # Windows
```

### Étape 2 : Installer les dépendances

```bash
pip install -r requirements.txt
```

Ou installer manuellement :

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter notebook
```

---

## Utilisation

### Lancer l'analyse t-SNE

1. **Activer l'environnement** :
   ```bash
   source .venv/bin/activate
   ```

2. **Lancer Jupyter** :
   ```bash
   jupyter notebook
   ```

3. **Ouvrir le notebook** `t_SNE.ipynb` et exécuter les cellules dans l'ordre :
   - **Imports** : Charger toutes les librairies
   - **Chargement dataset** : Lire `lamme_training.csv`
   - **Preprocessing** : One-hot encoding, imputation, standardisation
   - **PCA** : Réduction de 30+ variables à ~20 (95% variance)
   - **t-SNE** : Projection 2D avec perplexité=30
   - **Visualisation** : Graphiques de classification et environnement TE
   - **Export** : Sauvegarder `lamme_tsne_coords.csv`

### Résultats principaux générés

- `tsne_lamme.png` : Projection t-SNE colorée par classe et environnement TE
- `tsne_lamme_perplexity.png` : Analyse de sensibilité à la perplexité
- `lamme_tsne_coords.csv` : Données exportées avec coordonnées t-SNE

---

## Description des Variables

### Features numériques utilisées
- **Interactions PPI** : `PPI_Interactant_G1`, `PPI_Interactant_G2`, `PPI_Intersection`, `PPI_Union`, `PPI_Jaccard`
- **Domaines Pfam** : `Pfam_G1`, `Pfam_G2`, `Pfam_Inter`, `Pfam_Union`, `Pfam_Jaccard`
- **Divergence** : `Ks`, `Ka`, `Ka_Ks`, `Ks_SD`, `Ka_SD`, `Distance`
- **Élément Transposable** : `TE_Coverage_G1/G2`, `TE_Density_G1/G2`, `TE_Number_G1/G2`
- **Taille familiale** : `Family_Size`

### Features catégorielles utilisées
- **Elément transposable** : `TE_Env_G1`, `TE_Env_G2`, `TE_Env_Pair`

### Variables de classification
- **Classification** : 0 (redondant fonctionnellement), 1 (divergent fonctionnellement)
- **TE_Env_Class** : `free`, `half`, `rich` (densité d'éléments transposables)

---

## Auteur

Thomas Hucorne - Etudiant M2DS & ensIIE (Promo 2026)

---
