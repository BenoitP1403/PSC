# Projet PrimAITE — Apprentissage par renforcement pour la défense réseau

Ce dépôt regroupe l'ensemble des travaux d'apprentissage par renforcement menés sur [**PrimAITE**](https://github.com/Alan-Turing-Institute/PrimAITE), une plateforme de simulation cybersécurité dans laquelle un *Blue Agent* apprend à défendre un réseau contre un *Red Agent* exécutant des campagnes d'attaque (principalement du **data manipulation**).

Le projet explore et compare plusieurs approches RL - algorithmes classiques (Q-Learning, REINFORCE), algorithmes profonds via Stable-Baselines3 (PPO, A2C, DQN, MaskablePPO), une approche **padding** pour topologies dynamiques, et une approche **GNN** qui exploite nativement la structure de graphe du réseau.

## Contenu du dépôt

```
.
├── GNN/                    # Approche GNN (PyTorch Geometric) + curriculum
├── Mouse/                  # Implémentations pédagogiques Q-Learning et REINFORCE
├── Padding/                # Approche padding + MaskablePPO (topologies dynamiques)
├── Parameter optimization/ # Tuning Optuna des hyperparamètres SB3
├── SB3 vanilla/            # Entraînement PPO/DQN/A2C/MaskablePPO de référence
├── Scenarios/              # Scénarios YAML PrimAITE (partagés par tous les modules)
```

### `GNN/`
Approche **Graph Neural Network** qui exploite la structure de graphe du réseau PrimAITE. Utilise PyTorch Geometric (GINEConv / HeteroConv) avec un apprentissage par **curriculum** en 5 étapes (introduction progressive de nouveaux scénarios). Produit l'agent de comparaison directe avec l'approche padding sur les mêmes métriques d'évaluation.

### `Mouse/`
Implémentations **pédagogiques** de Q-Learning (Sarsamax, off-policy) et REINFORCE (policy gradient, on-policy) à partir de NumPy, sur une tâche jouet de navigation en grille. Ne dépend pas de PrimAITE — sert à illustrer les concepts fondamentaux de RL qui sous-tendent les modules SB3/Padding/GNN.

### `Padding/`
Approche **padding** pour environnements à topologie variable : observation et action maintenues à **taille fixe**, avec bits de présence par nœud, masquage dur des actions (`ActionMasker`) et randomisation de domaine par épisode. Entraîne un `MaskablePPO` dans trois modes : `static`, `dynamic`, `randomized`. Sert de contrepartie à l'approche GNN.

### `Parameter optimization/`
Optimisation des hyperparamètres via [**Optuna**](https://optuna.org/) (TPE sampler + Median pruner) pour chaque algorithme SB3 (PPO, MaskablePPO, A2C, DQN). Les meilleurs jeux d'hyperparamètres sont sérialisés en JSON dans `best_params/` et consommés par `SB3 vanilla/train.py`. Permet de garantir que les comparaisons inter-algorithmes se font à hyperparamètres calibrés plutôt qu'arbitraires.

### `SB3 vanilla/`
Entraînement de référence « sans fioritures » : un scénario PrimAITE + un algorithme SB3 (PPO, DQN, A2C, MaskablePPO) avec les hyperparamètres calibrés par `Parameter optimization/`. Sert de **baseline** pour les approches plus avancées (Padding, GNN).

### `Scenarios/`
Contient les fichiers YAML décrivant les scénarios PrimAITE utilisés dans tout le projet (data manipulation, variantes 3-PC / curriculum, etc.). Voir `Scenarios/README.md` pour le détail de chaque scénario.


## Prérequis globaux

- Python ≥ 3.10
- `primaite`, `stable-baselines3`, `sb3-contrib`, `gymnasium`, `optuna`, `pyyaml`, `numpy`, `matplotlib`
- Pour le module GNN : `torch`, `torch-geometric`

## Objectif global

Fournir une **base expérimentale reproductible** pour étudier l'apprentissage par renforcement dans un contexte de cybersécurité réseau, en comparant sur pied d'égalité des approches de complexité croissante : RL from scratch - SB3 vanilla - Padding - GNN.
