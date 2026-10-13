# Protocole de sélection du nombre de clusters pour un k-means

## Objectif

On essaie de se donner une procédure reproductible pour appliquer un **k-means** sur chaque famille de descripteurs (au moins celles pour qui les scores n'étaient pas tous concordants) et sélectionner un nombre de clusters raisonnable parmi :

\[
K \in \{2,3,4,5\}.
\]

La sélection ne repose pas sur un seul critère, mais sur la combinaison de plusieurs éléments :

1. qualité interne de la partition via de scores (inertie, silhouette, etc.) ;
2. stabilité par sous-échantillonnage et comparaison des partitions par ARI ;
4. contrôle du caractère non trivial des tailles de clusters ;
5. interprétabilité des groupes obtenus.

L’objectif n’est pas nécessairement de trouver un unique `meilleur K` automatique, mais d’identifier une valeur de \(K\) donnant une partition stable, non dégénérée et interprétable.

---

## 1. Préparation des données

`Fait` 

Le k-means repose sur des distances euclidiennes. Il est donc important que les variables soient sur des échelles comparables. L'analyse principale doit donc être réalisée sur les **descripteurs standardisés** :

\[
X \longrightarrow X_{\text{standardisé}} \longrightarrow \text{k-means}.
\]

La représentation T-SNE ne doit pas être utilisée comme support principal du clustering. Elle peut être utilisée ensuite comme outil de visualisation, en colorant les points par les labels obtenus par k-means sur les descripteurs standardisés.

Autrement dit, la procédure recommandée est :

\[
X_{\text{standardisé}} \longrightarrow \text{k-means}
\]

avec, en parallèle :

\[
X_{\text{standardisé}} \longrightarrow \text{T-SNE pour visualisation}.
\]

On peut conserver une visualisation T-SNE si elle a déjà été produite (T-SNE effectué sur des données standardisées ?), mais il vaut mieux éviter de choisir les clusters à partir des coordonnées T-SNE seules. Un k-means appliqué directement au T-SNE peut éventuellement être mentionné comme analyse exploratoire, mais pas comme analyse principale.

Si certaines variables sont très asymétriques, il peut être utile de tester au préalable une transformation logarithmique ou robuste, mais cela doit être justifié (à voir avec Erine si elle a déjà fait ça ?).

---

## 2. Grille de nombres de clusters à tester

`Fait` 

On teste :

\[
K \in \{2,3,4,5\}.
\]

Pour chaque famille et chaque valeur de \(K\), on applique un k-means sur les **descripteurs standardisés**. La projection T-SNE peut ensuite servir à visualiser les clusters obtenus, mais elle ne constitue pas le support principal du clustering.

### Code Python avec valeurs d'options

```python
from sklearn.cluster import KMeans

K_values = [2, 3, 4, 5]

results = {}

for K in K_values:
    model = KMeans(
        n_clusters=K,
        init='k-means++',
        n_init=15, # je ne sais pas si tu l'as fait ?
        random_state=42, # whatever
    )
    labels = model.fit_predict(X_scaled)

    results[K] = {
        "model": model,
        "labels": labels,
        "inertia": model.inertia_
    }
```

### Remarque sur `n_init`

Le k-means dépend de l’initialisation des centres. L’option `init='k-means++'` initialise les centres de façon plus stable qu’une initialisation aléatoire simple. L’option `n_init=15` demande à scikit-learn de lancer plusieurs initialisations et de retenir celle ayant la plus faible inertie.

---

## 3. Critères internes de qualité

### 3.1 Inertie intra-classe

`Fait` 

Le k-means cherche à minimiser l’inertie intra-classe :

\[
\sum_{k=1}^{K} \sum_{i \in C_k} \|x_i - \mu_k\|^2,
\]

où :

- \(C_k\) est le cluster \(k\) ;
- \(\mu_k\) est le centre du cluster \(k\) ;
- \(x_i\) est l’observation \(i\).

L’inertie diminue mécaniquement quand \(K\) augmente. On ne choisit donc pas simplement le \(K\) qui minimise l’inertie.

On observe plutôt la courbe d’inertie en fonction de \(K\) et on cherche un éventuel **coude** : une valeur de \(K\) après laquelle le gain devient faible.

### À consigner

Pour chaque famille et chaque \(K\), noter :

- l’inertie ;
- la diminution d’inertie par rapport au \(K\) précédent ; 
- l’existence ou non d’un coude visible.

---

### 3.2 Score de silhouette (c'était le plus clair il me semble de tous les scores ?)

`Fait` 

Le score de silhouette mesure si les points sont plus proches de leur propre cluster que des autres clusters.

Pour une observation \(i\), on définit :

- \(a(i)\) : distance moyenne entre \(i\) et les autres points de son cluster ;
- \(b(i)\) : distance moyenne entre \(i\) et les points du cluster voisin le plus proche.

La silhouette de \(i\) est :

\[
s(i) = \frac{b(i)-a(i)}{\max(a(i),b(i))}.
\]

Elle varie entre -1 et 1.

- proche de 1 : point bien affecté ;
- proche de 0 : point entre deux clusters ;
- négatif : point probablement mal affecté.

Le score de silhouette global est la moyenne des \(s(i)\).

### À consigner

Pour chaque famille et chaque \(K\), noter :

- le score de silhouette moyen (tu peux mettre les autres aussi) ;
- le ou les \(K\) maximisant ce score ;
- les cas où le score reste faible pour toutes les valeurs de \(K\).

Attention : la silhouette favorise souvent des clusters convexes, bien séparés et de taille relativement équilibrée. Elle ne doit donc pas être utilisée seule.

Éventuellement reporter les autres scores, mais globalement si report d'un seul, prendr ele plus stable.

---

## 4. Contrôle des tailles de clusters

`Fait mais à visualiser par famille` 

Une partition peut être techniquement stable mais peu informative si elle produit un cluster minuscule ou un découpage trop déséquilibré.

### Principe

Pour chaque \(K\), on calcule les tailles des clusters :

\[
n_k = |C_k|.
\]

On vérifie ensuite que la partition n’est pas triviale.

Par exemple, on peut signaler comme problématique une partition contenant :

- un cluster de taille 1 ;
- un cluster représentant moins de 5 % des observations ;
- un cluster contenant presque toutes les observations.

Ces seuils ne sont pas universels. Ils doivent être adaptés au contexte biologique/statistique de l’étude.

### Code Python

```python
from collections import Counter

for K in K_values:
    labels = results[K]["labels"]
    counts = Counter(labels)
    sizes = np.array([counts[c] for c in sorted(counts)])
    proportions = sizes / len(labels)

    results[K]["cluster_sizes"] = sizes
    results[K]["cluster_proportions"] = proportions

    print("K =", K)
    print("tailles :", sizes)
    print("proportions :", np.round(proportions, 3))
```

### À consigner

Pour chaque famille et chaque \(K\), noter :

- la taille minimale de cluster ;
- la taille maximale de cluster ;
- les proportions des clusters ;
- si la partition est jugée acceptable ou triviale.

Exemple de tableau :

| K | Taille min | Taille max | Proportion min | Proportion max | Partition acceptable ? |
|---|------------|------------|----------------|----------------|-------------------------|
| 2 |            |            |                |                |                         |
| 3 |            |            |                |                |                         |
| 4 |            |            |                |                |                         |
| 5 |            |            |                |                |                         |

---

## 5. Stabilité par sous-échantillonnage

--> Ce serait pas mal, en particulier pour les familles pour lesquelles on a pas un résultat clair par les critères (PFAM et PPI).


### Principe général

Un bon choix de \(K\) doit donner une partition relativement stable lorsque l’on retire une partie des observations. Ou formulé différmement, si je retire une partie des observations, est-ce que la structure de clusters retrouvée reste globalement la même ?

On répète donc la procédure suivante :

1. tirer aléatoirement un sous-échantillon des observations, par exemple 80 % des données ;
2. **ré-effectuer entièrement le k-means depuis zéro sur ce sous-échantillon**, avec le même nombre de clusters \(K\), mais sans réutiliser les labels obtenus sur l'échantillon complet ;
3. stocker les indices des observations retenues et les labels obtenus sur ce sous-échantillon ;
4. comparer les partitions obtenues entre répétitions ;
5. calculer un score de stabilité moyen à l’aide de l’ARI.

Le point important est que la stabilité ne consiste pas à reprendre une partition déjà calculée et à la restreindre au sous-échantillon. À chaque répétition, on relance un nouveau k-means sur les données du sous-échantillon.

Comme les sous-échantillons ne contiennent pas toujours exactement les mêmes observations, les partitions doivent ensuite être comparées uniquement sur les observations communes aux deux sous-échantillons.

---

### 5.1. L’Adjusted Rand Index, ou ARI 

### Principe

L’ARI permet de comparer deux partitions d’un même ensemble d’observations et vaut :
- 1 si les deux partitions sont identiques ;
- environ 0 si l’accord est du niveau attendu au hasard ;
- éventuellement négatif si l’accord est inférieur au hasard.

L’ARI ne dépend pas du nom des clusters. Deux partitions identiques mais avec des labels permutés auront un ARI égal à 1.

#### Formule

Soient deux partitions \(U = \{U_1, \dots, U_R\}\) et \(V = \{V_1, \dots, V_C\}\) d’un même ensemble de \(n\) observations.

On définit :

\[
n_{rc} = |U_r \cap V_c|,
\]

\[
a_r = \sum_c n_{rc},
\qquad
b_c = \sum_r n_{rc}.
\]

L’indice de Rand ajusté s’écrit :

\[
ARI =
\frac{
\sum_{r,c} {n_{rc} \choose 2}
- \frac{\sum_r {a_r \choose 2} \sum_c {b_c \choose 2}}{{n \choose 2}}
}{
\frac{1}{2}\left[\sum_r {a_r \choose 2} + \sum_c {b_c \choose 2}\right]
- \frac{\sum_r {a_r \choose 2} \sum_c {b_c \choose 2}}{{n \choose 2}}
}.
\]

Dans la pratique, on utilise directement la fonction scikit-learn :

```python
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(labels_1, labels_2)
```

---

### 5.2. Sous-échantillonnage et ARI avec Python

#### Fonction de sous-échantillonnage

```python
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np

def run_kmeans_subsample(X_scaled, K, sample_fraction=0.8, random_state=None):
    rng = np.random.default_rng(random_state)
    n = X_scaled.shape[0]
    sample_size = int(sample_fraction * n)

    # 1. Tirage du sous-échantillon
    indices = rng.choice(n, size=sample_size, replace=False)
    X_sub = X_scaled[indices]

    # 2. Ré-estimation complète du k-means sur ce sous-échantillon
    # Important : on ne restreint pas les labels d'un k-means déjà appris.
    # On relance bien un nouveau k-means à chaque sous-échantillonnage.
    model = KMeans(
        n_clusters=K,
        init='k-means++',
        n_init=15,
        random_state=random_state
    )
    labels_sub = model.fit_predict(X_sub)

    return indices, labels_sub
```

#### Comparaison de deux sous-échantillons

```python
def ari_on_common_indices(indices_1, labels_1, indices_2, labels_2, min_common=10):
    # observations communes aux deux sous-échantillons
    common = np.intersect1d(indices_1, indices_2)

    # Sécurité : si l'intersection est trop petite, l'ARI est peu informatif
    if len(common) < min_common:
        return np.nan

    # dictionnaires observation -> label
    dict_1 = {idx: lab for idx, lab in zip(indices_1, labels_1)}
    dict_2 = {idx: lab for idx, lab in zip(indices_2, labels_2)}

    # labels des mêmes observations dans les deux partitions
    common_labels_1 = [dict_1[idx] for idx in common]
    common_labels_2 = [dict_2[idx] for idx in common]

    return adjusted_rand_score(common_labels_1, common_labels_2)
```

#### Boucle complète de stabilité

```python
def stability_by_ari(X_scaled, K, n_repeats=50, sample_fraction=0.8):
    subsamples = []

    for b in range(n_repeats):
        indices, labels = run_kmeans_subsample(
            X_scaled,
            K,
            sample_fraction=sample_fraction,
            random_state=b
        )
        subsamples.append((indices, labels))

    ari_values = []

    for i in range(n_repeats):
        for j in range(i + 1, n_repeats):
            ari = ari_on_common_indices(
                subsamples[i][0], subsamples[i][1],
                subsamples[j][0], subsamples[j][1]
            )
            if not np.isnan(ari):
                ari_values.append(ari)

    return np.array(ari_values)
```

#### Application à la grille de K

```python
stability_results = {}

for K in K_values:
    ari_values = stability_by_ari(
        X_scaled,
        K,
        n_repeats=50,
        sample_fraction=0.8
    )

    stability_results[K] = {
        "ari_mean": np.mean(ari_values),
        "ari_std": np.std(ari_values),
        "ari_median": np.median(ari_values),
        "ari_q25": np.quantile(ari_values, 0.25),
        "ari_q75": np.quantile(ari_values, 0.75)
    }

    print("K =", K)
    print(stability_results[K])
```

---

### 5.4. Visualisation des scores de stabilité

```python
ari_means = [stability_results[K]["ari_mean"] for K in K_values]
ari_q25 = [stability_results[K]["ari_q25"] for K in K_values]
ari_q75 = [stability_results[K]["ari_q75"] for K in K_values]

plt.figure()
plt.plot(K_values, ari_means, marker="o", label="ARI moyen")
plt.fill_between(K_values, ari_q25, ari_q75, alpha=0.2, label="Q25-Q75")
plt.xlabel("Nombre de clusters K")
plt.ylabel("ARI")
plt.title("Stabilité par sous-échantillonnage")
plt.legend()
plt.show()
```

### À consigner

Pour chaque famille et chaque \(K\), noter :

- ARI moyen ;
- ARI médian ;
- intervalle interquartile ;
- variabilité des ARI ;
- présence éventuelle d’un \(K\) nettement plus stable.

---

## 6. Comparaison synthétique des critères

Construire un tableau final résumant tous les critères.

| K | Inertie | Silhouette | ARI moyen | ARI médian | Taille min | Taille max | Partition non triviale ? | Interprétation |
|---|---------|------------|-----------|------------|------------|------------|----------------------------|----------------|
| 2 |         |            |           |            |            |            |                            |                |
| 3 |         |            |           |            |            |            |                            |                |
| 4 |         |            |           |            |            |            |                            |                |
| 5 |         |            |           |            |            |            |                            |                |

La sélection finale doit être argumentée en combinant les critères.

---

## 7. Règles de décision à envisager

On privilégie une valeur de \(K\) si :

1. elle présente une bonne stabilité par sous-échantillonnage ;
2. son ARI moyen ou médian est élevé par rapport aux autres valeurs de \(K\) ;
3. les tailles de clusters ne sont pas triviales ;
4. la silhouette est raisonnable ;
5. l’inertie ne diminue plus fortement au-delà de cette valeur ;
6. les clusters ont une interprétation cohérente avec le problème étudié.

Il faut éviter de choisir un \(K\) uniquement parce que :

- il maximise légèrement la silhouette ;
- il minimise l’inertie ;
- il donne un cluster très petit mais apparemment spécifique.

---

## 8. Exemple de conclusion attendue

Exemple de formulation possible :

> Les valeurs \(K=2,3,4,5\) ont été comparées par inertie, silhouette, stabilité par sous-échantillonnage et contrôle des tailles de clusters. La valeur \(K=3\) présente une stabilité supérieure aux autres valeurs, avec un ARI médian élevé et des clusters de tailles non triviales. La valeur \(K=4\) améliore légèrement l’inertie mais introduit un petit cluster difficile à interpréter. La valeur \(K=3\) est donc retenue comme compromis entre stabilité, simplicité et interprétabilité.

---

## 9. Points d’attention

- Le k-means suppose implicitement des clusters plutôt compacts, convexes et comparables en variance.
- Il peut mal fonctionner si les clusters ont des formes non sphériques ou des densités très différentes.
- Le résultat dépend de la distance euclidienne : le prétraitement des variables est donc essentiel. Le clustering principal doit être réalisé sur les descripteurs standardisés, pas sur les coordonnées T-SNE.
- Une partition interprétable mais instable doit être considérée avec prudence.
- Une partition stable ne sera pas forcément biologiquement pertinente (en tout cas pas au sens redondant / divergent).

---

## 10. Livrables attendus

Pour chaque famille de descripteurs

1. un tableau récapitulatif des critères pour \(K=2,3,4,5\) ;
2. la courbe d’inertie et les scores de silhouette ;
3. les distributions d’ARI par sous-échantillonnage ;
4. les tailles et proportions des clusters ;
5. une recommandation argumentée du nombre de clusters retenu ;

Avec les clusters retenus, une comparaison des paires qui sont classées ensemble via
1. un résumé graphique / numérique famille x clusters avec le nombre par cluster et la répartition (redondant / divergent)
2. une heat map (0 famille, 1 famille, ..., 4 familles)

---

## 11. Script minimal récapitulatif

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Standardisation des descripteurs d'origine
# Le k-means principal est appliqué sur X_scaled, pas sur les coordonnées T-SNE.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

K_values = [2, 3, 4, 5]
results = {}

# K-means sur l'ensemble des données
for K in K_values:
    model = KMeans(
        n_clusters=K,
        init='k-means++',
        n_init=15,
        random_state=0
    )
    labels = model.fit_predict(X_scaled)

    sizes = np.array([Counter(labels)[c] for c in sorted(set(labels))])
    proportions = sizes / len(labels)

    results[K] = {
        "labels": labels,
        "inertia": model.inertia_,
        "silhouette": silhouette_score(X_scaled, labels),
        "cluster_sizes": sizes,
        "cluster_proportions": proportions
    }

# Fonctions de stabilité

def run_kmeans_subsample(X_scaled, K, sample_fraction=0.8, random_state=None):
    rng = np.random.default_rng(random_state)
    n = X_scaled.shape[0]
    sample_size = int(sample_fraction * n)

    # 1. Tirage du sous-échantillon
    indices = rng.choice(n, size=sample_size, replace=False)
    X_sub = X_scaled[indices]

    # 2. Ré-estimation complète du k-means sur ce sous-échantillon
    # Important : on ne restreint pas les labels d'un k-means déjà appris.
    # On relance bien un nouveau k-means à chaque sous-échantillonnage.
    model = KMeans(
        n_clusters=K,
        init='k-means++',
        n_init=15,
        random_state=random_state
    )
    labels_sub = model.fit_predict(X_sub)

    return indices, labels_sub


def ari_on_common_indices(indices_1, labels_1, indices_2, labels_2, min_common=10):
    common = np.intersect1d(indices_1, indices_2)
    if len(common) < min_common:
        return np.nan

    dict_1 = {idx: lab for idx, lab in zip(indices_1, labels_1)}
    dict_2 = {idx: lab for idx, lab in zip(indices_2, labels_2)}

    common_labels_1 = [dict_1[idx] for idx in common]
    common_labels_2 = [dict_2[idx] for idx in common]

    return adjusted_rand_score(common_labels_1, common_labels_2)


def stability_by_ari(X_scaled, K, n_repeats=50, sample_fraction=0.8):
    subsamples = []
    for b in range(n_repeats):
        indices, labels = run_kmeans_subsample(
            X_scaled,
            K,
            sample_fraction=sample_fraction,
            random_state=b
        )
        subsamples.append((indices, labels))

    ari_values = []
    for i in range(n_repeats):
        for j in range(i + 1, n_repeats):
            ari = ari_on_common_indices(
                subsamples[i][0], subsamples[i][1],
                subsamples[j][0], subsamples[j][1]
            )
            if not np.isnan(ari):
                ari_values.append(ari)

    return np.array(ari_values)

# Calcul de stabilité
stability_results = {}

for K in K_values:
    ari_values = stability_by_ari(X_scaled, K, n_repeats=50, sample_fraction=0.8)
    stability_results[K] = {
        "ari_mean": np.mean(ari_values),
        "ari_median": np.median(ari_values),
        "ari_std": np.std(ari_values),
        "ari_q25": np.quantile(ari_values, 0.25),
        "ari_q75": np.quantile(ari_values, 0.75)
    }

# Affichage synthétique
for K in K_values:
    print("\nK =", K)
    print("Inertie :", results[K]["inertia"])
    print("Silhouette :", results[K]["silhouette"])
    print("Tailles :", results[K]["cluster_sizes"])
    print("Proportions :", np.round(results[K]["cluster_proportions"], 3))
    print("ARI stabilité :", stability_results[K])
```