# Souris — Q-Learning vs REINFORCE *from scratch*

Ce dossier regroupe l’implémentation et la comparaison de deux algorithmes d’apprentissage par renforcement **implémentés entièrement *from scratch*** (uniquement avec NumPy — pas de Stable Baselines, pas de PyTorch, pas de Gymnasium), appliqués à une tâche de navigation sur grille. Une souris apprend, à partir des seules récompenses, à collecter du fromage sur une grille 10×10 tout en évitant les cases empoisonnées.

## Contenu du dossier

```
Souris/
├── main.py             # Point d'entrée : choix de l'algorithme et lancement de l'entraînement + visualisation
├── config.py           # Hyperparamètres (taille de grille, taux d'apprentissage, discount, itérations…)
├── env.py              # Logique MDP : transitions d'états, fonction de récompense, conditions terminales
├── board.py            # Génération du plateau et gestion de l'état
├── q_learning.py       # Implémentation du Q-Learning (Sarsamax, off-policy)
├── policy_gradient.py  # Implémentation de REINFORCE (gradient de politique, on-policy)
├── visualize.py        # Visualisation PyGame de l'entraînement et de la politique apprise
└── requirements.txt
```

### `main.py`
Point d’entrée : sélectionne l’algorithme à entraîner puis lance la visualisation. Par défaut, le Q-Learning est utilisé ; pour basculer sur REINFORCE, décommenter la ligne correspondante.

### `config.py`
Centralise l’ensemble des hyperparamètres (taille de la grille, nombre d’itérations, taux d’apprentissage, facteur de discount, probabilité de poison, etc.) ainsi que la définition des actions disponibles.

### `env.py` / `board.py`
Modélisent le **MDP** : génération du plateau, placement des fromages et du poison, transitions d’états, fonction de récompense et conditions terminales.

### `q_learning.py`
Implémentation **Q-Learning (Sarsamax, off-policy)** apprenant une Q-table `Q(s, a)` via une politique d’exploration ε-greedy avec décroissance exponentielle (`ε` de 1.0 → 0.01).

### `policy_gradient.py`
Implémentation **REINFORCE (on-policy)** optimisant directement une politique stochastique softmax paramétrée par une table de préférences θ, avec retour Monte Carlo sur trajectoire complète.

### `visualize.py`
Visualisation PyGame de la dynamique d’entraînement et de la politique apprise (déplacements de la souris, consommation du fromage, cases empoisonnées).

## Le problème

Une souris navigue sur une grille 10×10 pour collecter du fromage tout en évitant les cases empoisonnées. L'agent apprend uniquement à partir des récompenses.

| Type de case | Récompense |
|--------------|------------|
| Fromage      | +50        |
| Poison       | −1000 (terminal) |
| Vide         | −5 + bonus de proximité (jusqu'à +5) |

Le bonus de proximité encourage l'exploration vers le fromage sans en révéler directement la position.

## Algorithmes implémentés

### Q-Learning (Sarsamax — off-policy)

Apprend une Q-table `Q(s, a)` estimant la valeur de chaque action dans chaque état. Utilise une **politique d'exploration ε-greedy** avec décroissance exponentielle (`ε` de 1.0 → 0.01).

**Règle de mise à jour (équation de Bellman) :**
$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot [r + \gamma \cdot \max_{a'} Q(s', a') − Q(s, a)]$

Détails d'implémentation clés :
- Les actions invalides (collisions avec les murs) sont masquées avant la sélection de l'action greedy
- Les égalités sont départagées aléatoirement entre actions de valeurs équivalentes pour éviter les boucles déterministes
- L'état du plateau est copié à chaque épisode pour permettre le suivi de la consommation du fromage

### REINFORCE (gradient de politique — on-policy)

Optimise directement une politique stochastique $\pi_\theta(a|s)$ paramétrée par une table de préférences $\theta$. Utilise une **politique softmax** avec masquage des actions invalides (fixées à −1e9 avant le softmax).

**Règle de mise à jour :**
$\theta(s, a) \leftarrow \theta(s, a) + \alpha \cdot G_t \cdot \nabla \log \pi_\theta(a_t | s_t)$

Ce qui se simplifie en :
- $+\alpha \cdot G \cdot (1 − \pi(a))$ pour l'action effectuée
- $+\alpha \cdot G \cdot (−\pi(a'))$ pour toutes les autres actions

Les trajectoires sont déroulées intégralement avant la mise à jour (retour Monte Carlo `G`).

## Prérequis

- Python ≥ 3.10
- `numpy`, `pygame` (voir `requirements.txt`)

Installation :
```bash
pip install -r requirements.txt
```

## Utilisation rapide

```bash
# Lance l'algorithme sélectionné dans main.py (Q-Learning par défaut)
python main.py
```

Pour basculer sur REINFORCE, ouvrir `main.py` et remplacer :
```python
visualize(learn_q_table, greedy_policy)
```
par :
```python
visualize(learn_reinforce, policy)
```

Les hyperparamètres peuvent être ajustés dans `config.py` :
```python
ROWS, COLS = 10, 10       # Taille de la grille
ITERATIONS = 500_000      # Itérations d'entraînement
LEARNING_RATE = 0.0005
DISCOUNT = 0.99
POISON_PROB = 0.1
```

## Objectif global

Ce projet propose un cadre minimal et auto-contenu pour comparer, à partir d’une même tâche de navigation, les comportements d’apprentissage d’une méthode **value-based off-policy** (Q-Learning) et d’une méthode **policy-based on-policy** (REINFORCE), sans l’abstraction d’une bibliothèque RL tierce.
