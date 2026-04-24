# Parameter optimization - Tuning Optuna des hyperparamètres SB3

Ce dossier regroupe les scripts d’**optimisation d’hyperparamètres** pour les algorithmes Stable-Baselines3 (PPO, MaskablePPO, A2C, DQN) appliqués à un scénario PrimAITE. Le tuning utilise [**Optuna**](https://optuna.org/) avec un sampler **TPE** (Tree-structured Parzen Estimator) et un pruner **Median** pour explorer efficacement l’espace de recherche. Les meilleurs hyperparamètres trouvés sont sérialisés en JSON et ensuite consommés par le script `SB3 agent/train.py` pour l’entraînement final.

## Contenu du dossier

```
Parameter optimization/
├── optimize_parameters.py    # Point d'entrée CLI : lance une étude Optuna pour un algorithme donné
├── study_algorithm.py        # Orchestration Optuna : création d'étude, sauvegarde des résultats
├── objective_functions.py    # Fonctions objectif + espaces de recherche par algorithme
└── best_params/
    ├── PPO.json              # Meilleurs hyperparamètres trouvés pour PPO
    ├── MPPO.json             # Idem pour MaskablePPO
    ├── A2C.json              # Idem pour A2C
    └── DQN.json              # Idem pour DQN
```

### `optimize_parameters.py`
Point d’entrée en ligne de commande. Charge le scénario PrimAITE (par défaut `../Scenarios/data_manipulation_3_pc.yaml`), sélectionne la fonction objectif correspondant à l’algorithme choisi, puis lance l’étude Optuna via `study_algorithm`.

Argument CLI :
- `--algo {PPO,MPPO,DQN,A2C}` - choix de l’algorithme à optimiser (obligatoire)

Constantes (modifiables directement dans le fichier) :
- `N_TRIALS = 50` - nombre de trials Optuna
- `TOTAL_TIMESTEPS = 100_000` - durée d’entraînement par trial
- `N_EVAL_EPISODES = 50` - épisodes d’évaluation par seed
- `TUNE_SEEDS = [0, 1, 2]` - seeds utilisées pour chaque trial (moyenne des 3 récompenses)

### `study_algorithm.py`
Gère toute la mécanique Optuna :
- création de l’étude avec **`TPESampler(seed=42)`** et **`MedianPruner`** (prune un trial si sa récompense moyenne courante est en dessous de la médiane des trials précédents au même step)
- persistance de l’étude dans une base SQLite locale (`best_params/optuna_studies.db`) - permet la reprise d’une étude interrompue
- sauvegarde du meilleur trial au format JSON (`best_params/<ALGO>.json`) avec métadonnées (study name, best_value, best_params, seed rewards, timestamp UTC)

Expose aussi `load_env_config` et `make_env` utilisés par les fonctions objectif.

### `objective_functions.py`
Une fonction objectif par algorithme. Chacune définit un **espace de recherche** (via `trial.suggest_*`) puis entraîne un modèle sur chacune des `TUNE_SEEDS`, évalue la récompense moyenne, et renvoie la moyenne inter-seed. Le pruning intermédiaire est activé après chaque seed.

Hyperparamètres explorés :

| Algo | Hyperparamètres recherchés |
|---|---|
| **PPO** / **MaskablePPO** | `learning_rate`, `gamma`, `n_steps`, `batch_size`, `ent_coef`, `clip_range`, `n_epochs`, `gae_lambda` |
| **A2C** | `learning_rate`, `gamma`, `n_steps`, `ent_coef`, `gae_lambda`, `vf_coef` |
| **DQN** | `learning_rate`, `gamma`, `buffer_size`, `batch_size`, `tau`, `train_freq`, `gradient_steps`, `exploration_fraction`, `exploration_final_eps`, `target_update_interval` |

Les combinaisons incohérentes (par ex. `batch_size > n_steps` ou `n_steps % batch_size ≠ 0` pour PPO/MPPO) sont prunées automatiquement.

### `best_params/`
Stocke deux types d’artefacts :
- **`<ALGO>.json`** - meilleurs hyperparamètres trouvés, sérialisés avec le nom de l’étude, la meilleure valeur, les récompenses par seed, et des métadonnées (timesteps d’entraînement, nombre d’épisodes d’éval, seeds utilisées, timestamp). Consommés par `SB3 agent/train.py`.
- **`optuna_studies.db`** - base SQLite Optuna contenant l’intégralité des trials pour chaque étude (utile pour reprise, inspection, visualisation).

## Prérequis

- Python ≥ 3.10
- `optuna`, `primaite`, `stable-baselines3`, `sb3-contrib`, `pyyaml`

```bash
pip install optuna primaite stable-baselines3 sb3-contrib pyyaml
```

## Utilisation rapide

```bash
# Depuis le dossier Parameter optimization/
python optimize_parameters.py --algo PPO
python optimize_parameters.py --algo MPPO
python optimize_parameters.py --algo A2C
python optimize_parameters.py --algo DQN
```

Chaque exécution :
1. reprend l’étude existante (`load_if_exists=True`) ou en crée une nouvelle,
2. lance jusqu’à `N_TRIALS` nouveaux trials,
3. écrase `best_params/<ALGO>.json` avec le meilleur résultat final.

Inspection interactive d’une étude :
```python
import optuna
study = optuna.load_study(
    study_name="PPO",
    storage="sqlite:///best_params/optuna_studies.db",
)
print(study.best_params, study.best_value)
```

## Objectif global

Ce module fournit une base d’**hyperparamètres optimisés reproductibles** pour chaque algorithme SB3 utilisé dans le projet. Il permet de s’assurer que les comparaisons entre algorithmes (PPO vs A2C vs DQN…) et entre approches (baseline SB3 vs PPO+GNN) se font à régimes d’hyperparamètres calibrés, et non à configurations arbitraires.
