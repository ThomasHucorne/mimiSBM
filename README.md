# Modèle à blocs stochastiques multicouches pour l'étude du devenir fonctionnel de gènes dupliqués chez ***Arabidopsis thaliana***

Ce dossier rassemble des notebooks d’exploration et de clustering de paires de gènes, ainsi que des essais d’application du modèle mimiSBM à des matrices d’adjacence multicouches. Les analyses couvrent notamment les descripteurs Gene Expression, Pfam, PPI et éléments transposables (TE).

## Structure du dossier

```text
mimiSBM/
├── README.md
├── requirements.txt
├── cleaning_variables.py
├── LDA/                         # Sélection de descripteurs par LDA et K-means
├── t_SNE/                       # Projections t-SNE globales et par famille
├── implication_genes/           # Analyses des gènes et des paires
├── mimisbm/                     # Pipelines et simulations mimiSBM
└── Plots/                       # Figures, matrices d'adjacence et modèles exportés
    ├── kmeans/
    ├── lda/
    ├── mimiSBM/
    ├── t_SNE_family/
    ├── t_SNE_global/
    └── ...                      
```

Le dossier `Plots/` contient également des figures à sa racine et des résultats sérialisés (`.npy`, `.pkl`, `.rds`). Les données d’entrée ne sont pas présentes dans l’arborescence actuelle de `mimiSBM` : plusieurs notebooks attendent des fichiers CSV et des matrices d’adjacence dans un dossier `Données/`. Les Données ne sont pas renseignées dans ce github car elles ne sont pas encore publiés et appartiennent à une publication scientifique (Ezoe et al. 2021) Pour compiler, il faut fournir ces fichiers et, selon le notebook, adapter les chemins relatifs ou absolus configurés dans ses cellules.

## Description des Variables
### Features numériques utilisées
*Interactions PPI* : PPI_Interactant_G1, PPI_Interactant_G2, PPI_Intersection, PPI_Union, PPI_Jaccard
*Domaines Pfam* : Pfam_G1, Pfam_G2, Pfam_Inter, Pfam_Union, Pfam_Jaccard
*Expression des gènes* : Ks, Ka, Ka/Ks, Distance
*Élément Transposable* : TE_Coverage_Asymmetry, TE_Coverage_Mean, TE_Density_Asymmetry, TE_Density_Mean, TE_Number_Asymmetry, TE_Number_Mean
### Variables de classification
*Classification* : 0 (redondant fonctionnellement), 1 (divergent fonctionnellement)

## Préparation et dépendances

`cleaning_variables.py` transforme des métriques par gène en variables de paire, traite les valeurs manquantes, encode les variables catégorielles et exporte les jeux nettoyés d’entraînement et de prédiction. Le script attend notamment les fichiers `Données/lamme_training.csv` et `Données/lamme_prediction.csv`.

Les dépendances Python listées dans `requirements.txt` s’installent avec :

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Pour lancer une analyse, ouvrir le notebook voulu dans Jupyter et exécuter ses cellules dans l’ordre. Vérifier d’abord que ses fichiers d’entrée existent et que les chemins définis dans les cellules correspondent à l’emplacement des données.

## Notebooks

### LDA et K-means — `LDA/`

- **`lda_kaks.ipynb`** — applique la LDA puis le K-means aux descripteurs de la famille expression des gènes (GE) et produit des résultats de clustering et des matrices d’adjacence.
- **`lda_pfam.ipynb`** — même démarche pour les descripteurs Pfam.
- **`lda_ppi.ipynb`** — même démarche pour les descripteurs d’interactions protéine-protéine (PPI).
- **`lda_te.ipynb`** — même démarche pour les descripteurs liés aux éléments transposables (TE).
- **`k_means_lda.ipynb`** — classe les descripteurs à l’aide de la LDA, applique K-means aux descripteurs retenus, compare plusieurs critères de qualité et étudie aussi les clusters dans un espace t-SNE.

- **`k_means_lda_silh.ipynb`** — sélectionne les descripteurs par LDA et utilise notamment le score de silhouette pour évaluer les clusters et construire les matrices d’adjacence.
- **`k_means_lda_bic.ipynb`** — variante qui utilise le BIC pour évaluer le K-means par descripteur. Elle traite à part les valeurs nulles des descripteurs `_Asymmetry`.
- **`k_means_lda_v2.ipynb`** — variante du pipeline LDA/K-means qui isole notamment un cluster de valeurs nulles pour les descripteurs `_Asymmetry`.

### Réduction de dimension — `t_SNE/`

- **`t_SNE.ipynb`** — explore le jeu d’entraînement, effectue une réduction par PCA puis une projection t-SNE et produit des visualisations.
- **`t_SNE_family.ipynb`** — applique t-SNE aux variables organisées par familles de descripteurs et compare les projections avec et sans PCA.
- **`t_SNE_global.ipynb`** — réalise une projection t-SNE globale des variables nettoyées, avec visualisations et analyse de sensibilité à la perplexité.

### Analyses des gènes — `implication_genes/`

- **`analyse_implication_genes.ipynb`** — étudie la fréquence d’apparition des gènes dans les paires, les éventuelles paires d’un même gène, les relations représentées en réseau et le lien entre cette fréquence et les clusters.
- **`compter_paires_te_zero.ipynb`** — compte les paires pour lesquelles les deux gènes ont une valeur TE égale à zéro pour les métriques analysées.

### mimiSBM — `mimisbm/`

- **`mimisbm_pipeline.ipynb`** — charge des matrices d’adjacence regroupées par familles de descripteurs, les assemble en tenseur multicouche et ajuste le modèle mimiSBM.
- **`mimisbm_pipeline_descriptors.ipynb`** — variante du pipeline où chaque couche du tenseur correspond à un descripteur individuel plutôt qu’à une famille.
- **`mimisbm_pipeline_bic_descriptors.ipynb`** — applique le pipeline par descripteur aux matrices d’adjacence associées au critère BIC.
- **`mimisbm_pipeline_criteria.ipynb`** — compare, sur une grille de paramètres, l’ELBO/ILvb, l’ICL variationnel et un ICL approché avec pénalité de type BIC.
- **`mimisbm_pipeline_generic.ipynb`** — version configurable du pipeline : chargement des matrices par un dictionnaire, recherche sur les paramètres, test de stabilité des partitions et vérification sur des données simulées.
- **`mimisbm_kaks.ipynb`** — ajuste mimiSBM à des matrices d’adjacence de la famille Ka/Ks et examine les résultats du modèle.
- **`mimisbm_gene_expression.ipynb`** — notebook d’application de mimiSBM à un jeu de matrices d’adjacence de descripteurs, avec construction du tenseur et estimation du modèle.
- **`mimisbm_simulation_package_python.ipynb`** — génère des réseaux multicouches selon une structure connue, ajuste la classe Python mimiSBM et compare les partitions estimées aux partitions simulées.
- **`mimisbm_simulation_r_vs_python.ipynb`** — porte en Python des simulations décrites côté R, puis compare le comportement de mimiSBM sur les réseaux simulés, avec et sans perturbation des labels.
- **`recouvrement_kaks.ipynb`** — calcule le recouvrement des clusters Ka et Ks, puis teste la robustesse du modèle **mimiSBM** à la permutation d’une partie des labels.


