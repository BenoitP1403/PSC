# SB3 agent - Entraînement PPO/DQN/A2C/MaskablePPO sur PrimAITE

Ce dossier contient le **baseline Stable-Baselines3** pour l’entraînement d’un agent défenseur sur PrimAITE, **sans GNN ni padding** de l’espace d’observation/action. Il sert de **point de comparaison** aux approches plus élaborées du dossier `GNN/` (PPO avec encodeur *Graph Neural Network* hétérogène). L’agent est entraîné avec une simple `MlpPolicy` sur l’observation vectorielle fournie nativement par PrimAITE, en utilisant les hyperparamètres optimisés dans le dossier `Parameter optimization/`.

## Contenu du dossier

```
SB3 agent/
└── train.py     # Script d'entraînement SB3 (PPO, MaskablePPO, A2C, DQN) sur un scénario PrimAITE
```

### `train.py`
Charge un scénario PrimAITE (par défaut `../Scenarios/data_manipulation_3_pc.yaml`), instancie un environnement `PrimaiteGymEnv`, puis entraîne l’algorithme sélectionné en ligne de commande en chargeant ses hyperparamètres optimisés depuis un fichier JSON (`../Parameter optimization/best_params/<ALGO>.json`). Les logs TensorBoard sont écrits dans `SB3 agent/tensorboard_logs/<ALGO>/` et le modèle final est sauvegardé sous `SB3 agent/<ALGO> model.zip`.

Algorithmes disponibles :
- `PPO` - Proximal Policy Optimization (Stable-Baselines3)
- `MPPO` - MaskablePPO (sb3-contrib, gestion des actions invalides par masquage)
- `A2C` - Advantage Actor-Critic
- `DQN` - Deep Q-Network

Argument CLI :
- `--algo {PPO,MPPO,A2C,DQN}` - choix de l’algorithme (obligatoire)

Constantes notables (modifiables directement dans `train.py`) :
- `SCENARIO_PATH` - chemin du YAML de scénario
- `TOTAL_TIMESTEPS = 1_000_000` - nombre total de pas d’entraînement

## Prérequis

- Python ≥ 3.10
- `primaite`, `stable-baselines3`, `sb3-contrib`, `pyyaml`, `tensorboard`

```bash
pip install primaite stable-baselines3 sb3-contrib pyyaml tensorboard
```

Le script résout ses chemins **par rapport à son propre emplacement** (via `__file__`), il peut donc être lancé depuis n’importe quel dossier. Il lit :
- le scénario YAML à `Scenarios/data_manipulation_3_pc.yaml` (à la racine du projet)
- les hyperparamètres optimisés à `Parameter optimization/best_params/<ALGO>.json`

## Utilisation rapide

```bash
python train.py --algo PPO
python train.py --algo MPPO
python train.py --algo A2C
python train.py --algo DQN
```

Suivi en temps réel avec TensorBoard :
```bash
tensorboard --logdir tensorboard_logs/
```

## Objectif global

Ce baseline permet de quantifier le gain apporté par l’approche **PPO + GNN** du dossier `GNN/` en comparant, sur les mêmes scénarios PrimAITE, les performances d’un agent *flat* (observation vectorielle + `MlpPolicy`) à celles d’un agent exploitant explicitement la structure du graphe réseau. Il fournit également un cadre simple pour comparer entre eux plusieurs algorithmes classiques de RL profond (PPO, MaskablePPO, A2C, DQN) à hyperparamètres optimisés équivalents.
