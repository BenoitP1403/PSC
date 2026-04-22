# Padding — MaskablePPO sur PrimAITE à topologie dynamique

Ce dossier regroupe l'approche **padding** utilisée pour entraîner et évaluer un agent de défense réseau (Blue Agent) dans des environnements PrimAITE dont la topologie varie au cours d'un épisode ou d'un scénario à l'autre. Contrairement à l'approche GNN (voir `../GNN/`), qui exploite nativement la structure de graphe, l'approche padding garde l'espace d'observation et d'action **aplati et de taille fixe** : les nœuds absents ou non encore annoncés sont remplis par du padding et masqués durement via `ActionMasker` (sb3-contrib). Cette approche sert de point de comparaison directe à l'approche GNN, sur les mêmes scénarios et avec les mêmes protocoles d'évaluation.

## Contenu du dossier

```
Padding/
├── structured_obs_wrapper.py   # Bits de présence/annonce par nœud
├── dynamic_node_wrapper.py     # Activation progressive des nœuds (scheduled/random)
├── domain_randomization.py     # Randomisation de topologie par épisode
├── reward_shaping.py           # Récompenses intrinsèques anti no-op
├── universal_obs_wrapper.py    # Alignement vers un espace de taille fixe maximale
├── train_padding.py            # Entraînement MaskablePPO (static/dynamic/randomized)
└── evaluate.py                 # Évaluation et comparaison padding vs GNN
```

### `structured_obs_wrapper.py`
Enveloppe l'environnement PrimAITE de base (observations aplaties) et ajoute deux bits de présence par nœud : `present` et `announced`. L'agent peut ainsi distinguer les nœuds actifs, absents, ou en cours d'annonce, sans modifier la structure interne de PrimAITE.

### `dynamic_node_wrapper.py`
Gère l'activation progressive des nœuds au fil d'un épisode. Chaque nœud dynamique suit trois états : **absent → annoncé → présent**. Deux modes d'activation sont supportés : `scheduled` (déterministe, calendrier fixe) et `random` (probabiliste). Les observations et actions des nœuds inactifs sont automatiquement masquées.

### `domain_randomization.py`
Applique une randomisation de topologie à chaque épisode : le sous-ensemble de nœuds visibles et leurs calendriers d'activation sont tirés aléatoirement selon une `RandomizationConfig`. Cela oblige l'agent à généraliser à des topologies diverses plutôt qu'à mémoriser un scénario fixe.

Configuration utilisée en production (`RAND_CONFIG`) :
```python
RandomizationConfig(
    subnet_range=(1, 1),
    clients_per_subnet_range=(9, 12),  # 9 à 12 nœuds visibles par épisode
    dynamic_prob=0.25,                 # ~25 % des nœuds deviennent dynamiques
    activation_step_range=(10, 60),    # activation dans la première moitié de l'épisode
)
```

### `reward_shaping.py`
Ajoute trois signaux de récompense intrinsèques pour éviter l'effondrement vers la politique no-op :
- **Bonus de détection** — récompense le scan d'un nœud effectivement compromis
- **Bonus d'interruption** — récompense le ralentissement de la chaîne d'attaque
- **Pénalité d'action inutile** — pénalise l'action sur un nœud sain

### `universal_obs_wrapper.py`
Aligne les espaces d'observation et d'action sur une taille fixe maximale, déterminée à partir de l'ensemble des scénarios utilisés (voir `compute_max_dimensions`). Permet à un modèle entraîné sur un scénario donné d'être évalué sur un autre sans aucune modification du modèle — c'est la brique clé qui rend les comparaisons inter-scénarios possibles.

### `train_padding.py`
Fabrique d'environnements et boucle d'entraînement principale. Expose `make_env()` qui assemble la pile complète de wrappers selon le mode choisi, et `train()` / `evaluate()` pour l'entraînement et l'évaluation via `MaskablePPO`. Supporte trois modes d'entraînement : `static`, `dynamic`, `randomized` (voir tableau ci-dessous).

### `evaluate.py`
Module d'évaluation propre à l'approche padding. Fournit les conteneurs de données (`EpisodeResult`, `EvalStats`) et les fonctions d'évaluation (`evaluate_model`, `evaluate_random`, `evaluate_noop`, `compare_approaches`). Le module est totalement indépendant du module GNN : `compare_approaches` accepte les résultats GNN sous forme d'un `EvalStats` pré-calculé passé en paramètre.

## Pile de wrappers

```
PrimaiteGymEnv
  └── StructuredObsWrapper             # bits de présence par nœud
        └── DynamicNodeWrapper         # (mode dynamic / randomized uniquement)
              └── DomainRandomizationWrapper   # (mode randomized uniquement)
                    └── RewardShapingWrapper   # récompense intrinsèque
                          └── ActionMasker     # masquage dur des actions (sb3_contrib)
                                └── UniversalPaddingWrapper   # alignement taille fixe
```

## Modes d'entraînement

| Mode | Description |
|---|---|
| `static` | Scénario unique, topologie fixe, tous les nœuds présents dès le départ |
| `dynamic` | Scénario unique, activation progressive des nœuds selon un calendrier fixe |
| `randomized` | Topologie et calendriers d'activation tirés aléatoirement à chaque épisode |

## Prérequis

- Python ≥ 3.10
- `stable-baselines3`, `sb3-contrib`, `gymnasium`, `numpy`, `matplotlib`, `pyyaml`, `primaite`

```bash
pip install stable-baselines3 sb3-contrib gymnasium numpy matplotlib pyyaml primaite
```

## Utilisation rapide

Les scénarios sont partagés avec le reste du projet — ils se trouvent dans `../Scenarios/`.

```bash
# Depuis le dossier Padding/ (ou le dossier racine selon la config d'import)

# Entraînement — scénario statique
python -m padding.train_padding \
    --config ../Scenarios/data_manipulation_3_pc.yaml \
    --mode static \
    --total-timesteps 100000

# Entraînement — nœuds dynamiques (calendrier fixe)
python -m padding.train_padding \
    --config ../Scenarios/data_manipulation_3_pc.yaml \
    --mode dynamic \
    --total-timesteps 100000

# Entraînement — randomisation de topologie par épisode
python -m padding.train_padding \
    --config ../Scenarios/data_manipulation_3_pc.yaml \
    --mode randomized \
    --total-timesteps 100000

# Évaluation d'un modèle entraîné
python -m padding.train_padding \
    --config ../Scenarios/data_manipulation_3_pc.yaml \
    --mode static \
    --eval-only \
    --model-path models/ppo_static.zip
```

## Objectif global

Ce module fournit l'**approche padding** de référence pour la comparaison avec l'approche GNN. L'idée directrice : rester dans un espace d'observation et d'action de taille fixe, et traiter la variabilité de topologie par un mélange de bits de présence, de masquage dur des actions, et de randomisation de domaine à l'entraînement. Les résultats produits ici (via `evaluate.compare_approaches`) servent de baseline directe à `../GNN/` pour évaluer l'apport réel d'une architecture qui exploite nativement la structure de graphe.
