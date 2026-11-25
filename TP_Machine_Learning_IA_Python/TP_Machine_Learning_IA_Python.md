# TP --- Machine Learning et intelligence artificielle avec Python

## Guide de prompts, d'expérimentation et d'optimisation

**Public :** développeurs, analystes de données, Data Scientists et ingénieurs IA disposant des bases de Python.\
**Durée indicative :** deux journées pour le parcours principal ; modules avancés à réaliser en complément.\
**Environnement :** Python, Jupyter Notebook, Git et GitHub. Aucun service cloud obligatoire.

## 1. Objectifs pédagogiques

-   Utiliser une IA générative comme assistant de développement Python, sans considérer ses réponses comme automatiquement correctes.
-   Générer, explorer et préparer des données.
-   Entraîner et comparer des modèles de classification supervisée.
-   Concevoir des expériences reproductibles, mesurer les performances et optimiser les hyperparamètres.
-   Comprendre le surapprentissage, le sous-apprentissage, le déséquilibre des classes et les seuils de décision.
-   Interpréter les prédictions, suivre les expériences, tester la robustesse et déployer une application interactive.

## 2. Scénario métier : prédiction du départ des clients

Une entreprise souhaite identifier les clients susceptibles de résilier leur abonnement. La cible `Churn` vaut 1 en cas de départ et 0 sinon. Il s'agit d'une classification binaire supervisée.

  Variable            Description
  ------------------- -------------------------------
  Age                 Âge du client
  Tenure              Ancienneté en mois
  MonthlyCharges      Facturation mensuelle
  TotalCharges        Facturation cumulée
  ContractType        Type de contrat
  SupportTickets      Nombre de demandes de support
  SatisfactionScore   Satisfaction de 1 à 5
  PaymentMethod       Moyen de paiement
  Churn               Variable cible (0 ou 1)

Les données sont synthétiques : elles ne représentent pas de véritables clients et ne permettent pas de tirer des conclusions sur une population réelle.

## 3. Technologies

  Technologie                   Usage
  ----------------------------- ----------------------------------------
  Python, Jupyter               Développement et expérimentation
  NumPy, Pandas                 Génération et manipulation des données
  Matplotlib, Seaborn, Plotly   Visualisation
  Scikit-learn                  Prétraitement, modèles et évaluation
  XGBoost, LightGBM             Modèles de boosting (optionnels)
  imbalanced-learn              Rééquilibrage
  Optuna                        Optimisation
  SHAP, LIME                    Explicabilité
  TensorFlow / Keras            Réseaux de neurones
  MLflow                        Suivi des expériences
  Joblib, Streamlit             Sauvegarde et application
  pytest, Git                   Tests et versionnement

## 4. Installation et méthode de travail

Créer un environnement virtuel :

``` bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell (à la place de la commande précédente)
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install jupyter pandas numpy matplotlib seaborn scikit-learn joblib pytest
jupyter notebook
```

Installer les dépendances avancées uniquement lorsque nécessaire, dans des versions compatibles avec l'environnement. Demander à l'assistant de produire un fichier `requirements.txt` reproductible et vérifier les versions réellement installées.

Pour chaque prompt : copier le bloc dans l'assistant IA, fournir le code existant si nécessaire, exécuter les cellules, vérifier les résultats, corriger les erreurs et consigner les observations avant de passer à la suite. Conserver le même jeu de test final, inaccessible pendant les choix de modèles. Utiliser des partitions de validation ou des prédictions hors pli pour les expériences. Les scénarios synthétiques alternatifs doivent rester distincts du test final.

## 5. Prompts détaillés

### Prompt 01 --- Génération du projet de référence

``` text
Agis comme un Data Scientist senior, un ingénieur Machine Learning et un formateur Python.
Crée un notebook Jupyter complet, exécutable et pédagogique pour prédire le départ des clients.
Génère 10 000 clients synthétiques avec les variables Age, Tenure, MonthlyCharges, TotalCharges, ContractType, SupportTickets, SatisfactionScore, PaymentMethod et Churn. Introduis des corrélations plausibles, des valeurs manquantes, des valeurs aberrantes et un déséquilibre des classes. Génère la cible de manière probabiliste, sans fuite d'information. Utilise random_state=42.
Sépare les données en ensembles d'entraînement et de test stratifiés avant tout ajustement du prétraitement. Réserve le test final sans l'utiliser pour choisir les modèles.
Réalise une analyse exploratoire sur l'entraînement : statistiques, distributions, valeurs manquantes, corrélations et diagrammes en boîte.
Construis un Pipeline et un ColumnTransformer avec imputation, standardisation et encodage one-hot adapté aux catégories inconnues.
Entraîne un DummyClassifier de référence, une régression logistique, un arbre de décision, une forêt aléatoire, KNN, SVM et Gradient Boosting. Ajoute XGBoost en option.
Compare exactitude, précision, rappel, F1, ROC-AUC, précision moyenne (AP) et temps de calcul par validation croisée stratifiée. Distingue AP de l'aire trapézoïdale sous la courbe précision-rappel.
Produis des tableaux, matrices de confusion, courbes ROC et précision-rappel à partir de prédictions hors pli. Sélectionne un modèle de référence selon le F1 de validation.
Prépare des fonctions réutilisables pour les expériences suivantes, la sauvegarde Joblib et les prédictions sur de nouveaux clients. N'évalue pas encore le test final.
Écris toutes les explications, commentaires et titres en français. Conserve les identifiants Python et les noms d'API usuels. Fournis les cellules dans l'ordre, les dépendances, les exercices et des assertions de vérification. N'invente aucun résultat d'exécution.
```

**Travail :** exécuter le notebook, vérifier les dimensions, la distribution de la cible et la reproductibilité du modèle de référence.

### Prompt 02 --- Génération et qualité des données

``` text
Étends le projet avec un générateur de données paramétrable utilisant NumPy et Pandas.
Permets de choisir 1 000, 5 000, 10 000 ou 50 000 observations, le taux de valeurs manquantes, le taux de valeurs aberrantes, la proportion de clients qui partent, le niveau de bruit et la force des relations entre variables et cible.
Génère la cible à partir d'un mécanisme probabiliste documenté. Vérifie les proportions réellement obtenues et les contraintes métier. Exporte les données en CSV.
Crée des graphiques comparant les distributions, la qualité des données et les performances en validation pour chaque scénario. Modifie un facteur à la fois et conserve le test final intact. Explique les résultats en français.
```

**Travail :** comparer trois tailles et trois niveaux de bruit ; distinguer l'effet de la taille de celui de la qualité.

### Prompt 03 --- Analyse exploratoire avancée

``` text
À partir des données d'entraînement, produis une analyse exploratoire complète avec Pandas, Matplotlib et Seaborn.
Génère des histogrammes, densités, diagrammes en boîte, violons, nuages de points, cartes de corrélation et distributions catégorielles. Limite les graphiques coûteux à un échantillon reproductible.
Compare les clients qui partent et ceux qui restent. Identifie les valeurs manquantes, doublons, anomalies, corrélations fortes et risques de fuite de données.
Construis un tableau de bord visuel et explique en français comment interpréter chaque graphique. Ne confonds pas corrélation et causalité.
```

**Travail :** identifier trois variables potentiellement utiles et formuler des hypothèses à tester.

### Prompt 04 --- Comparaison des algorithmes

``` text
Crée un cadre de comparaison réutilisable pour la régression logistique, les arbres, Random Forest, KNN, SVM, Gradient Boosting, XGBoost, LightGBM et Naive Bayes avec un prétraitement compatible. Inclue DummyClassifier.
Utilise les mêmes plis de validation croisée stratifiée (5 plis) et des pipelines adaptés à chaque algorithme. Mesure exactitude, précision, rappel, F1, ROC-AUC, AP, temps d'entraînement et de prédiction.
Crée un DataFrame avec moyennes et écarts-types, classe les modèles selon le F1 et génère des graphiques comparatifs, courbes ROC, précision-rappel et matrices de confusion hors pli.
Explique les compromis entre performance, coût de calcul et interprétabilité. N'utilise pas le test final pour le classement.
```

**Travail :** justifier le choix d'un modèle selon les besoins métier et les ressources disponibles.

### Prompt 05 --- Impact de la taille d'entraînement

``` text
Entraîne une forêt aléatoire avec 10, 20, 40, 60, 80 et 100 % des données d'entraînement disponibles dans chaque pli. Conserve les hyperparamètres et les partitions de validation comparables.
Répète les expériences avec cinq graines aléatoires. Mesure les scores d'entraînement et de validation, le F1 et le temps de calcul.
Trace les courbes d'apprentissage avec bandes de variabilité (moyenne ± écart-type). Explique si davantage de données semble utile et distingue cette variabilité d'un intervalle de confiance.
```

**Travail :** déterminer à partir de quelle taille le gain marginal devient faible.

### Prompt 06 --- Surapprentissage et sous-apprentissage

``` text
Entraîne des arbres de décision avec max_depth de 1 à 30. Mesure les scores d'entraînement et de validation croisée pour chaque profondeur.
Trace les courbes en fonction de la complexité et identifie les zones de sous-apprentissage et de surapprentissage si elles apparaissent.
Répète l'expérience avec Random Forest en faisant varier min_samples_split, min_samples_leaf et n_estimators. Modifie un paramètre à la fois.
Explique le compromis biais-variance et sélectionne la complexité sur les données de validation uniquement. Ne suppose pas que les courbes suivront nécessairement une forme théorique.
```

**Travail :** expliquer les écarts entre scores d'entraînement et de validation.

### Prompt 07 --- Optimisation des hyperparamètres

``` text
Compare GridSearchCV, RandomizedSearchCV et l'optimisation bayésienne avec Optuna pour Random Forest et XGBoost.
Définis des espaces de recherche propres à chaque modèle : n_estimators, max_depth, min_samples_split, max_features, learning_rate et subsample selon leur compatibilité.
Utilise une validation croisée stratifiée à cinq plis, le F1 comme objectif et des budgets de calcul comparables. Fixe les graines et limite le parallélisme excessif.
Enregistre le nombre de configurations, le temps, les meilleurs paramètres et les scores. Trace l'historique d'optimisation et l'importance des paramètres lorsque disponible.
Compare les gains par validation externe ou validation croisée imbriquée. Explique le biais de sélection du meilleur score de recherche. Garde le test final intact.
```

**Travail :** comparer le gain de F1 et le coût de calcul des trois méthodes.

### Prompt 08 --- Déséquilibre des classes

``` text
Génère des scénarios de proportions 50/50, 70/30, 90/10 et 99/1, en documentant les proportions obtenues.
Compare l'absence de rééquilibrage, class_weight, suréchantillonnage aléatoire, sous-échantillonnage et SMOTE ou SMOTENC selon le type des variables.
Utilise les pipelines imbalanced-learn pour appliquer le rééchantillonnage uniquement dans les plis d'entraînement. Vérifie que chaque pli contient suffisamment de cas minoritaires ; adapte k_neighbors si nécessaire.
Mesure précision, rappel, F1, exactitude équilibrée, ROC-AUC et AP. Trace les matrices de confusion et courbes précision-rappel sur des validations non rééchantillonnées.
Explique pourquoi l'exactitude seule peut être trompeuse et pourquoi les scores entre prévalences différentes doivent être interprétés avec prudence.
```

**Travail :** comparer un classifieur majoritaire à un modèle capable de détecter les départs rares.

### Prompt 09 --- Seuil de décision et coût métier

``` text
Utilise predict_proba ou des probabilités calibrées sur les données d'entraînement. Sur des prédictions hors pli ou une validation dédiée, teste les seuils de 0,1 à 0,9.
Calcule précision, rappel, F1, faux positifs et faux négatifs pour chaque seuil.
Suppose un coût de 500 euros par faux négatif et 50 euros par faux positif. Calcule le coût total = 500 × FN + 50 × FP.
Identifie le seuil minimisant ce coût sur la validation et compare-le au seuil 0,5. Crée un graphique interactif Plotly et une matrice de confusion dynamique.
Explique les compromis métier, la calibration des probabilités et les limites de ces hypothèses de coût. Fige le seuil retenu avant l'évaluation finale.
```

**Travail :** modifier les coûts et observer le déplacement du seuil optimal.

### Prompt 10 --- Création et sélection de variables

``` text
Crée des variables dérivées pertinentes : dépense moyenne, fréquence des tickets et indicateurs d'engagement disponibles avant la prédiction. Gère les divisions par zéro et les valeurs manquantes.
Compare StandardScaler, MinMaxScaler et RobustScaler pour les modèles sensibles à l'échelle.
Teste SelectKBest, information mutuelle, élimination récursive de variables et importance par permutation. Adapte les méthodes aux types de variables et aux modèles.
Intègre les transformations apprises dans les pipelines et compare les résultats avec les mêmes plis. Produis des graphiques d'importance et explique les limites liées aux variables corrélées.
```

**Travail :** vérifier si un modèle plus simple conserve des performances comparables.

### Prompt 11 --- Intelligence artificielle explicable

``` text
Ajoute SHAP et LIME au projet. Sélectionne des méthodes compatibles avec le modèle et le prétraitement ; utilise un échantillon de référence provenant de l'entraînement.
Génère une importance globale, un graphique SHAP beeswarm, des graphiques de dépendance et waterfall, ainsi que des explications locales LIME.
Sélectionne cinq clients de validation avec des probabilités différentes. Associe correctement les noms de variables transformées et les classes aux contributions.
Compare explications globales et locales. Traduis les résultats en langage métier et explique que les contributions ne démontrent pas une causalité. Vérifie la stabilité des explications lorsque possible.
```

**Travail :** expliquer deux prédictions opposées et discuter des limites des recommandations.

### Prompt 12 --- Réseau de neurones

``` text
Implémente un réseau de neurones avec TensorFlow/Keras : couches Dense, ReLU, Dropout et sortie sigmoïde. Utilise binary_crossentropy et Adam.
Ajuste le prétraitement uniquement sur l'entraînement et réserve une validation distincte. Teste plusieurs nombres de couches, neurones, tailles de lots, époques, taux d'apprentissage et taux de dropout.
Utilise EarlyStopping avec restauration des meilleurs poids et ModelCheckpoint. Fixe les graines et documente les limites de reproductibilité.
Trace les courbes de perte et d'exactitude, les courbes ROC et précision-rappel et les matrices de confusion sur validation.
Compare le réseau aux modèles classiques selon un protocole équivalent et explique pourquoi le Deep Learning n'est pas toujours préférable pour les données tabulaires.
```

**Travail :** comparer trois architectures et analyser l'effet du dropout.

### Prompt 13 --- Suivi des expériences avec MLflow

``` text
Intègre MLflow avec une expérience nommée Customer-Churn-Prediction.
Pour chaque exécution, journalise le modèle, la version des données, la graine, la taille d'entraînement, les hyperparamètres, les métriques de validation, le temps et les graphiques.
Enregistre le pipeline entraîné comme artefact MLflow. Réalise au moins 20 configurations avec un budget de calcul raisonnable.
Fournis les commandes pour démarrer l'interface locale MLflow et configure un stockage compatible avec la version installée. Explique comment comparer les exécutions, reproduire un résultat et enregistrer un modèle dans un registre compatible.
N'enregistre pas de secrets ni de données personnelles dans les artefacts.
```

**Travail :** retrouver et reproduire une expérience à partir de ses paramètres.

### Prompt 14 --- Robustesse et dérive des données

``` text
Crée des jeux synthétiques de robustesse distincts du test final : valeurs manquantes, bruit, anomalies, catégories inconnues et changements de distributions.
Utilise des cibles connues générées par un mécanisme documenté. Distingue dérive des variables, changement de prévalence et dérive du concept.
Évalue le modèle sans réentraînement dans chaque scénario. Mesure la dégradation du F1, de l'AP et du coût métier.
Implémente le PSI avec des classes de référence fixes et un traitement des fréquences nulles, ainsi que le test de Kolmogorov-Smirnov pour les variables numériques appropriées.
Trace les distributions avant et après dérive. Explique les limites des seuils de PSI et des tests statistiques, et propose une stratégie de surveillance et de réentraînement.
```

**Travail :** identifier les perturbations les plus dommageables et proposer des alertes.

### Prompt 15 --- Application interactive Streamlit

``` text
Crée une application Streamlit en français utilisant le pipeline et le modèle sélectionnés, sauvegardés avec Joblib (ou le format Keras si nécessaire). Ne charge que des artefacts de confiance.
Ajoute des champs pour les caractéristiques du client, une probabilité de départ, une classe prédite, un niveau de risque et un seuil configurable.
Ajoute des explications locales compatibles, des prédictions par lot à partir d'un CSV, des graphiques de distribution et un export CSV.
Affiche les métriques enregistrées en précisant leur origine (validation ou test final). Valide le schéma, les types, les plages et les catégories ; gère les erreurs de prédiction.
Génère app.py, requirements.txt et les instructions de lancement. Ajoute des tests de prédiction et de validation des entrées. Ne présente pas les scores comme des garanties métier.
```

**Travail :** tester plusieurs profils, un CSV valide et un CSV incorrect.

### Prompt 16 --- Génération du support de formation complet

``` text
Agis comme un ingénieur IA senior et un formateur technique. Transforme les modules précédents en un TP complet intitulé « Machine Learning et intelligence artificielle avec Python : expérimentation, optimisation et déploiement ».
Prépare une version participant avec consignes, cellules TODO et questions, ainsi qu'une version formateur avec solutions complètes.
Pour chaque module, fournis les objectifs, les prérequis, la théorie, les étapes, le code Python, les graphiques, les vérifications, les exercices, les erreurs fréquentes et les défis optionnels.
Génère les notebooks Jupyter, scripts Python, README.md, requirements.txt, tests pytest et le générateur de données reproductible.
Ajoute une étape finale : figer le modèle, le prétraitement et le seuil, réentraîner sur les données de développement selon le protocole retenu et évaluer une seule fois sur le test réservé. Rapporter les métriques, les limites et les résultats réellement obtenus.
Utilise le français pour la documentation, les commentaires et les interfaces. Conserve les noms d'API et les identifiants techniques nécessaires. Ne prétends pas avoir exécuté des cellules non exécutées.
```

## 6. Prompts complémentaires de vérification

### Prompt 17 --- Tests automatisés et revue du code

``` text
Agis comme un ingénieur qualité Machine Learning. Examine le code existant et identifie les erreurs, fuites de données, problèmes de reproductibilité et incompatibilités de versions.
Crée des tests pytest pour le schéma des données, les valeurs manquantes, les catégories inconnues, les dimensions, les probabilités entre 0 et 1, la reproductibilité et la cohérence des prédictions après sauvegarde et rechargement.
Ajoute des tests de non-régression et des tests sur entrées invalides. Vérifie que les transformations apprises ne voient jamais le test final. Fournis les commandes pytest et distingue les tests réellement exécutés des tests proposés.
```

### Prompt 18 --- Rapport d'expérimentation

``` text
À partir uniquement des résultats réellement produits par mes notebooks et fichiers CSV, crée un rapport Markdown en français. Présente les hypothèses, le protocole, les métriques, les tableaux comparatifs, les graphiques, les limites et les recommandations. Identifie les expériences non concluantes et les informations manquantes. Ne fabrique aucun score. Propose les prochaines expériences en justifiant leur intérêt.
```

## 7. Visualisations attendues

  Visualisation                         Interprétation
  ------------------------------------- --------------------------------------
  Carte de corrélation                  Relations entre variables numériques
  Histogrammes et diagrammes en boîte   Distributions et anomalies
  Matrice de confusion                  Vrais et faux positifs/négatifs
  Courbe ROC et précision-rappel        Capacité de discrimination
  Courbes d'apprentissage               Effet de la taille d'entraînement
  Courbes de validation                 Effet de la complexité
  Historique Optuna                     Progression de l'optimisation
  Courbe coût/seuil                     Compromis métier
  Graphiques SHAP                       Contributions aux prédictions
  Courbes de perte Keras                Dynamique d'entraînement
  Graphiques de dérive                  Changements de distributions

Aucun score n'est fourni à titre de résultat réel : les participants doivent produire et interpréter leurs propres mesures.

## 8. Structure proposée du dépôt GitHub

``` text
ai-machine-learning-tp/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── generate_dataset.py
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_pretraitement.ipynb
│   ├── 03_entrainement.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_comparaison.ipynb
│   ├── 06_complexite.ipynb
│   ├── 07_optimisation.ipynb
│   ├── 08_desequilibre.ipynb
│   ├── 09_variables.ipynb
│   ├── 10_explicabilite.ipynb
│   ├── 11_deep_learning.ipynb
│   ├── 12_mlflow.ipynb
│   └── 13_robustesse.ipynb
├── models/
├── experiments/
├── visualizations/
├── tests/
│   ├── test_preprocessing.py
│   └── test_predictions.py
├── student/
├── solutions/
└── app/
    └── app.py
```

Exemple de `.gitignore` :

``` gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
.pytest_cache/
.env
mlruns/
mlartifacts/
*.db
models/*.joblib
models/*.keras
data/*.csv
```

Pour initialiser le dépôt (après avoir enregistré ce guide sous `README.md` et créé le `.gitignore`) :

``` bash
git init
git add README.md .gitignore
git commit -m "Ajout du guide de TP Machine Learning"
git branch -M main
# Remplacer l'adresse par celle de votre dépôt GitHub vide
git remote add origin https://github.com/VOTRE-COMPTE/VOTRE-DEPOT.git
git push -u origin main
```

Créer des commits par expérience, conserver les configurations et les résultats légers, et éviter de versionner les secrets, environnements virtuels et modèles volumineux. Si nécessaire, conserver les corrigés dans un dépôt privé distinct.

## 9. Organisation pédagogique indicative

  Activité                              Durée
  ----------------------------- -------------
  Introduction et prompts              30 min
  Données et exploration               45 min
  Prétraitement                        45 min
  Entraînement                         60 min
  Évaluation et comparaison            45 min
  Complexité                           30 min
  Optimisation                         60 min
  Déséquilibre                         30 min
  Seuil de décision                    30 min
  Création de variables                45 min
  Explicabilité                        45 min
  Deep Learning                        60 min
  MLflow                               60 min
  Robustesse                           45 min
  Streamlit                            60 min
  Tests, défis et restitution          60 min
  **Total indicatif**             **13 h 30**

Prévoir du temps supplémentaire pour les installations et les recherches d'hyperparamètres coûteuses.

## 10. Défis et livrables

-   [ ] Dépasser le F1 du modèle de référence en validation.
-   [ ] Réduire le temps d'entraînement sans perte significative de performance.
-   [ ] Identifier cinq variables importantes et expliquer les limites de cette analyse.
-   [ ] Comparer trois stratégies d'optimisation.
-   [ ] Minimiser le coût métier en ajustant le seuil.
-   [ ] Tester un déséquilibre extrême des classes.
-   [ ] Comparer un réseau de neurones à XGBoost.
-   [ ] Simuler une dérive des données.
-   [ ] Enregistrer et comparer 20 expériences MLflow.
-   [ ] Déployer le modèle dans Streamlit et exécuter les tests.

Pour chaque expérience, renseigner le tableau suivant :

  --------------------------------------------------------------------------------------------------
  ID         Hypothèse     Paramètres modifiés   F1 validation   AP         Temps      Conclusion
  ---------- ------------- --------------------- --------------- ---------- ---------- -------------
  EXP-001    À compléter   À compléter           ---             ---        ---        À compléter

  --------------------------------------------------------------------------------------------------

**Livrables :** notebooks exécutés, code source, configurations, résultats mesurés, graphiques, tests, application et rapport final.

## 11. Limites et bonnes pratiques

Ce document est un guide de prompts, pas un projet Python déjà implémenté. Le code généré doit être revu et exécuté avant utilisation en formation. Les choix de scénario, coûts fictifs, outils et durée sont adaptables. Les données synthétiques servent à apprendre les méthodes, non à valider un système de production. Le test final ne doit pas servir à optimiser le modèle ; toute nouvelle optimisation après consultation du test exige une nouvelle évaluation indépendante.
