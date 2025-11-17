## Contexte du projet de Machine Learning (Albert School)
https://www.kaggle.com/datasets/zahranusrat/lifestyle-and-health-risk-prediction-dataset?resource=download

https://www.kaggle.com/code/zahranusrat/lifestyle-and-health-risk-prediction

https://www.kaggle.com/code/devraai/lifestyle-and-health-risk-prediction-analysis

Ce document sert de **guideline** pour tout le projet Kaggle de Machine Learning, afin de rester aligné avec :
- les **exigences du professeur**,
- sa **manière de structurer le code et les notebooks**,
- les **bonnes pratiques** attendues en EDA et en modélisation.

---

## 1. Objectif général du projet

- **Utiliser un dataset Kaggle** (tabulaire) pour construire un projet de machine learning complet.
- **Réaliser une EDA (Explanatory Data Analysis)** claire et structurée.
- **Tester plusieurs algorithmes de ML** pour comparer leurs performances :
  - Dummy (baseline)  
  - Logistic Regression  
  - Polynomial Regression (ou Polynomial Features + modèle linéaire / logistique)  
  - Decision Tree  
  - k-Nearest Neighbors (kNN)  
  - Random Forest  
- **Choisir le meilleur modèle** selon des métriques pertinentes et l’interpréter.
- **Travailler proprement en notebook**, en respectant le style et les consignes du professeur.

---

## 2. Exigences implicites du professeur

À partir des notebooks fournis (`Regression_Hyperparameter_ModelSelection_FULL.ipynb`, `Exercise_1_EDA_to_ML_1_Statement.ipynb`, `Warmup.ipynb`) on déduit :

- **Tout doit être obtenu par code**
  - Aucun chiffre “tapé à la main” dans les réponses (pas de hard-coding).
  - Toujours calculer les dimensions, proportions, moyennes, etc. avec `df.shape`, `df.isna().sum()`, etc.

- **Structure claire et commentée**
  - Sections bien séparées avec des titres : `#`, `##`, `###`.
  - Textes explicatifs en markdown avant/après les blocs de code importants.
  - Affichage lisible des résultats (`print` avec phrases) et non pas seulement des nombres bruts.

- **Séparation nette entre étapes**
  - Import des librairies dans une seule cellule.
  - Chargement des données + inspection basique.
  - EDA (questions / réponses, visualisations).
  - Prétraitement / nettoyage.
  - Séparation `X` / `y` puis train/test.
  - Modélisation (plusieurs modèles, même pipeline).
  - Comparaison et conclusion.

- **Utilisation de fonctions réutilisables**
  - Fonctions utilitaires pour les graphiques (ex. histogrammes de CV scores, predicted vs actual).
  - Fonctions pour afficher la répartition d’une variable (ex. pie chart de n’importe quelle colonne).
  - Éviter le copier-coller de code identique.

- **Randomness contrôlée**
  - Toujours fixer `random_state` pour `train_test_split`, arbres, forêts, etc.

- **Utilisation de validation croisée (cross-validation)**
  - `KFold` ou `StratifiedKFold`.
  - `cross_val_score` pour estimer les performances moyennes ± écart-type.

- **Hyperparamètres explorés de façon simple et lisible**
  - Boucles `for` sur une petite grille de valeurs (ex. degrés du polynôme, `n_neighbors`, `max_depth`, etc.).
  - Résultats stockés dans une liste/`DataFrame` + visualisation (courbe ou barres).

---

## 3. Style de code et de notebook à respecter

- **Imports groupés au début**
  - `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`.
  - `train_test_split`, `cross_val_score`, `KFold` / `StratifiedKFold`, `GridSearchCV` (si besoin).
  - Modèles de `sklearn` nécessaires.
  - Métriques (`accuracy_score`, `f1_score`, `classification_report`, `roc_auc_score` si pertinent).

- **Utilisation de `display` pour les DataFrames clés**
  - `display(df.head())`, `display(df.describe())`, etc. pour une lecture confortable.

- **Prints explicatifs**
  - Exemple :  
    - `print(f"Le nombre d'individus est {n_rows}")`  
    - `print(f"Dummy — Train: accuracy={acc_train:.3f}, f1={f1_train:.3f}")`

- **Code clair, peu de “one-liners” obscurs**
  - Privilégier la lisibilité à la concision extrême.

- **Visualisations**
  - Utilisation de `matplotlib` / `seaborn`.
  - Titres, axes, légendes, tailles de figures raisonnables.
  - Graphiques réutilisables via des fonctions (ex: `plot_cv_hist`, `plot_confusion_matrix`, etc.).

---

## 4. Structure recommandée du notebook de projet

Un notebook type pourrait s’appeler `Projet_ML_Kaggle.ipynb` et être structuré comme suit :

### 4.1. Titre et description

- Titre du projet.
- Lien vers le dataset Kaggle.
- Description courte :
  - Problème (classification binaire / multiclass / régression).
  - Variable cible (`target`).
  - Grandes étapes du notebook.

### 4.2. Imports

- Importer toutes les librairies nécessaires.
- Paramètres globaux (options pandas, style de plots, `random_state`, etc.).
- Définir un objet de validation croisée (ex: `kfold = KFold(n_splits=5, shuffle=True, random_state=42)` ou `StratifiedKFold`).

### 4.3. Chargement des données

- `df = pd.read_csv('...')` ou chargement depuis Kaggle.
- `display(df.head())`.
- `df.info()`.
- `df.describe(include='all')`.

### 4.4. EDA (Exploratory Data Analysis)

Exemples d’analyses inspirées de l’exo marketing :

- Nombre d’individus et de variables.
- Nombre total et proportion de valeurs manquantes.
- Types de variables (numériques, catégorielles).
- Statistiques descriptives des variables numériques.
- Répartition des variables catégorielles (countplot, pie chart).
- Corrélations entre variables numériques (`df.corr(numeric_only=True)` + heatmap).
- Visualisation des distributions (histogrammes, boxplots) pour détecter des outliers.
- Commentaires écrits (en markdown) sur ce que tu observes :
  - Variables importantes / fortement corrélées.
  - Présence d’outliers potentiels.
  - Problèmes de déséquilibre de classe, etc.

### 4.5. Nettoyage et prétraitement

- Gestion des valeurs manquantes (imputation simple ou suppression raisonnée).
- Suppression des variables constantes ou inutiles.
- Encodage des variables catégorielles (one-hot encoding ou autres).
- Standardisation / normalisation si nécessaire (important pour Logistic Regression, kNN).
- Création d’éventuelles nouvelles variables (feature engineering simple).

### 4.6. Définition de `X` et `y` + train/test split

- `target_col = '...'`
- `X = df.drop(columns=[target_col])`
- `y = df[target_col]`
- `X_train, X_test, y_train, y_test = train_test_split(..., test_size=0.2, random_state=42, stratify=y si classification)`
- Afficher les dimensions train/test.

### 4.7. Fonctions utilitaires pour l’évaluation

- Exemple de fonctions :
  - `plot_cv_hist(scores, title)` → histogramme des scores de CV.
  - `evaluate_classifier(name, model, X_train, y_train, X_test, y_test)` → cross-val + fit + métriques + affichage.
  - `plot_confusion_matrix` ou utilisation de `sklearn.metrics.ConfusionMatrixDisplay`.

Ces fonctions doivent **réduire la duplication** entre modèles.

### 4.8. Modèles à tester

Pour chaque modèle, suivre un pattern similaire à l’exemple de régression :

1. **Baseline : Dummy**
   - `DummyClassifier(strategy='most_frequent')`
   - Cross-validation (accuracy ou f1).
   - Fit sur train, évaluation train/test.
   - Afficher les métriques.

2. **Logistic Regression**
   - Standardisation recommandée si features de différentes échelles.
   - `LogisticRegression(max_iter=1000, penalty='l2', solver=...)`.
   - Cross-validation.
   - Fit + prédictions + métriques + éventuellement ROC/AUC (si binaire).

3. **Polynomial Regression / Polynomial Features**
   - Soit en régression pure (comme l’exemple Boston).
   - Soit en classification : `PolynomialFeatures` sur X + `LogisticRegression`.
   - Boucle sur les degrés du polynôme, stockage des scores, visualisation.

4. **Decision Tree**
   - `DecisionTreeClassifier(max_depth=..., random_state=42)`.
   - Tester quelques `max_depth` et/ou `min_samples_leaf`.
   - Cross-validation, fit, métriques.

5. **kNN**
   - `KNeighborsClassifier(n_neighbors=..., ...)`.
   - Bien penser à la standardisation des features.
   - Tester plusieurs valeurs de `k` (ex: 3, 5, 7, 9).

6. **Random Forest**
   - `RandomForestClassifier(n_estimators=100, max_depth=..., random_state=42)`.
   - Tester quelques profondeurs / nombres d’arbres.
   - Importance des features (optionnel mais intéressant).

Pour chaque modèle :
- Utiliser la **même stratégie de validation** (même `kfold`).
- Comparer sur les **mêmes métriques** (accuracy, f1, etc.).

### 4.9. Comparaison des modèles

- Créer un tableau récapitulatif (DataFrame) avec :
  - Nom du modèle.
  - Score moyen de cross-validation (+/- std).
  - Score train / test.
  - Éventuellement d’autres métriques (F1, AUC, etc.).
- Visualiser les scores (barplot).
- Discuter :
  - Modèles qui overfit / underfit.
  - Meilleur compromis performance / simplicité.

### 4.10. Conclusion

- Rappeler le dataset et le problème.
- Résumer la démarche :
  - EDA → nettoyage → prétraitement → plusieurs modèles → sélection.
- Dire quel modèle est retenu et pourquoi (métriques + interprétation).
- Mentionner les limites et pistes d’amélioration :
  - Plus de features, tuning plus avancé, modèles plus complexes, etc.

---

## 5. Checklist rapide à garder en tête

- **Dataset Kaggle choisi et bien documenté** (lien + description).
- **EDA complète** :
  - Dimensions, NA, types, distributions, corrélations.
  - Visualisations + commentaires.
- **Prétraitement propre** :
  - Gestion des NA.
  - Encodage des catégorielles.
  - Scaling si nécessaire.
- **Séparation X/y + train/test** avec `random_state` et `stratify` si classification.
- **Plusieurs modèles testés** :
  - Dummy + Logistic + Polynomial + Decision Tree + kNN + Random Forest.
  - Tous évalués avec la **même méthode** (cross-validation + train/test).
- **Comparaison claire** :
  - Tableau récapitulatif.
  - Discussion des résultats.
- **Conclusion argumentée** sur le meilleur modèle.
- **Pas de chiffres “magiques” hard-codés** : tout est calculé par le code.

---

## 6. Comment utiliser ce document

- Au début du projet : pour **planifier** ton notebook.
- Pendant le projet : comme **checklist** pour vérifier que tu as bien couvert toutes les étapes.
- À la fin : comme **grille d’auto-évaluation** avant de rendre le projet.

Tu peux adapter cette guideline au dataset concret que tu choisiras, mais l’idée est de rester **aligné sur la manière de faire du professeur** :
- structuré,
- reproductible,
- bien commenté,
- et rigoureux dans la comparaison des modèles.




