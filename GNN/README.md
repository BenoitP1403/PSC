# GNN - Entraînement PPO sur graphe pour PrimAITE

Ce dossier regroupe les scripts d’entraînement d’une politique **PPO** s’appuyant sur un encodeur **Graph Neural Network (GNN hétérogène, GINE)** pour l’environnement [PrimAITE](https://github.com/Alan-Turing-Institute/PrimAITE). L’objectif est d’apprendre un agent défenseur capable de détecter, contenir et corriger des attaques visant l’intégrité des données dans un système d’information modélisé sous forme de graphe (hôtes, services, liens réseau).

## Contenu du dossier

```
GNN/
├── GNN_single_scenario.py      # Entraînement sur un scénario unique
└── GNN_multiscenario.py        # Entraînement multi-scénarios avec curriculum
```

Les fichiers YAML de scénarios consommés par ces scripts se trouvent désormais à la racine du projet, dans [`../Scenarios/`](../Scenarios/README.md) (partagés avec `SB3 agent/` et `Parameter optimization/`).

### `GNN_single_scenario.py`
Entraîne une politique PPO-GNN sur **un seul** scénario PrimAITE. Il définit l’architecture du modèle (deux couches `GINEConv` via `HeteroConv`, têtes acteur/critique), la boucle de rollout et les mises à jour PPO, la gestion des checkpoints et le tracé des courbes d’entraînement.

Arguments CLI principaux :
- `--train-scenario` : chemin du YAML à utiliser (défaut : `../Scenarios/data_manipulation.yaml`)
- `--checkpoint-dir` : dossier de checkpoints
- `--plot-path` : chemin du graphique d’entraînement
- `--resume` : reprise depuis un checkpoint
- `--total-updates`, `--device`, `--seed`

### `GNN_multiscenario.py`
Entraîne une politique unique sur **plusieurs scénarios** via un **curriculum** en cinq étapes, avec un schéma d’observations et d’actions universel partagé entre tous les scénarios. Chaque étape introduit progressivement de nouveaux scénarios et ajuste les poids d’échantillonnage :

| Étape | Updates | Scénarios ajoutés |
|---|---|---|
| stage_0 | 0–399 | `data_manipulation` |
| stage_1 | 400–799 | + `…_with_different_connections` |
| stage_2 | 800–1199 | + `…_plus_one_client` |
| stage_3 | 1200–1599 | + `…_plus_one_dynamic_client` |
| stage_4 | 1600+ | + `…_plus_two_dynamic_clients` |

## Prérequis

- Python ≥ 3.10
- PyTorch, PyTorch Geometric (`torch_geometric`), Gymnasium, NumPy, Matplotlib, PyYAML, platformdirs
- Une installation de **PrimAITE** accessible (soit via `pip install primaite`, soit via une copie locale placée dans `GNN/PrimAITE/`, auquel cas les scripts ajoutent automatiquement `PrimAITE/src` au `PYTHONPATH`)

## Utilisation rapide

Les chemins par défaut (`../Scenarios/...`) supposent que le script est lancé **depuis le dossier `GNN/`** :

```bash
cd GNN

# Entraînement mono-scénario (scénario par défaut : data_manipulation.yaml)
python GNN_single_scenario.py

# Entraînement mono-scénario sur un autre fichier
python GNN_single_scenario.py --train-scenario ../Scenarios/data_manipulation_plus_one_client.yaml

# Entraînement multi-scénarios avec curriculum
python GNN_multiscenario.py
```

Les sorties (checkpoints, plots, logs) sont écrites dans `checkpoints_*/` et `training_plots*/` relatifs au dossier `GNN/`.

## Scénarios

Les scénarios (fichier YAML de référence + 7 variantes : topologie modifiée, ajout de clients statiques/dynamiques, second attaquant, passage à 8 ou 10 clients) sont documentés en détail dans [`../Scenarios/README.md`](../Scenarios/README.md).
