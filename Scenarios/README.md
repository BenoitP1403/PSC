# Scenarios - Fichiers YAML de scénarios PrimAITE

Ce dossier regroupe l’ensemble des fichiers de configuration YAML décrivant les scénarios de cybersécurité utilisés par les différents modules du projet (`GNN/`, `SB3 agent/`, `Parameter optimization/`). Ces scénarios sont centrés sur le **data manipulation** : ils servent à étudier la résilience opérationnelle d’un système d’information face à des attaques visant l’intégrité des données, tout en prenant en compte la continuité de service.

## Contenu du dossier

```
Scenarios/
├── data_manipulation.yaml                             # Scénario de référence
├── data_manipulation_with_different_connections.yaml  # Variante topologique du réseau
├── data_manipulation_plus_one_client.yaml             # + 1 client statique
├── data_manipulation_plus_one_dynamic_client.yaml     # + 1 client dynamique
├── data_manipulation_plus_two_dynamic_clients.yaml    # + 2 clients dynamiques
├── data_manipulation_two_attackers.yaml               # + 1 second attaquant
├── data_manipulation_plus_8_clients.yaml              # Passage à 8 clients
└── data_manipulation_dynamic_10_clients.yaml          # Passage à 10 clients avec activations dynamiques
```

## Scénario de référence

### `data_manipulation.yaml`

Ce scénario considère un système d’information de taille réduite comprenant :

- deux postes clients ;
- un serveur web ;
- un serveur de base de données ;
- un serveur de sauvegarde ;
- les équipements réseau associés.

Deux agents utilisateurs y génèrent une activité légitime, tandis qu’un agent attaquant cherche à altérer périodiquement les données de la base. En parallèle, un agent défenseur dispose de capacités d’observation et d’action sur les hôtes, les services, les fichiers et certains mécanismes de contrôle réseau afin de détecter, contenir et corriger les effets de l’attaque.

Ce scénario sert de **base de référence** pour l’ensemble des variantes décrites ci-dessous.

## Variantes du scénario

### `data_manipulation_with_different_connections.yaml`

Cette variante reprend le scénario de référence avec le même environnement, les mêmes agents, les mêmes objectifs et les mêmes capacités d’action pour l’attaquant comme pour le défenseur.

La différence principale porte sur la **topologie réseau** : la connexion de `client_1` au commutateur `switch_2` est déplacée vers un autre port, sans modification du comportement global du scénario.

Cette version permet d’évaluer dans quelle mesure les performances observées restent stables lorsque la structure des connexions réseau est légèrement modifiée.

---

### `data_manipulation_plus_one_client`

Cette variante conserve la même architecture générale, le même type d’attaque et les mêmes capacités de défense, mais ajoute :

- un troisième poste client ;
- un troisième agent utilisateur légitime.

L’attaquant peut ainsi opérer depuis un ensemble de points d’entrée plus large, tandis que le défenseur doit prendre en compte un environnement légèrement plus étendu.

Cette version permet d’étudier l’effet d’une **augmentation modérée de la taille du réseau** et de l’activité légitime sur la dynamique d’attaque et de défense.

---

### `data_manipulation_plus_one_dynamic_client`

Cette variante reprend le scénario de référence, mais ajoute un **troisième poste client dynamique**.

Contrairement au scénario initial, ce nouveau client n’est pas actif dès le départ : il apparaît plus tard dans l’exécution. Cela élargit progressivement :

- l’environnement observé par le défenseur ;
- les points d’entrée potentiels pour l’attaquant.

Cette version permet d’étudier le comportement des agents dans un cadre plus évolutif, où la surface du système n’est pas entièrement fixe dans le temps.

---

### `data_manipulation_plus_two_dynamic_clients`

Cette variante étend le scénario de référence avec **deux postes clients supplémentaires**, activés dynamiquement au cours de l’épisode.

Ces nouveaux clients ne sont pas présents dès le début et deviennent actifs plus tard, ce qui fait évoluer progressivement :

- la taille de l’environnement ;
- le nombre d’utilisateurs légitimes ;
- les points d’entrée potentiels pour l’attaquant.

Cette version permet d’étudier le comportement des agents dans un cadre plus évolutif, tout en restant cohérente avec la logique du scénario de base.

---

### `data_manipulation_two_attackers`

Cette variante conserve la même architecture, les mêmes utilisateurs légitimes et les mêmes capacités de défense, mais introduit **deux agents attaquants** au lieu d’un seul.

La pression offensive est donc renforcée, puisque plusieurs sources d’attaque peuvent agir dans le même environnement et viser simultanément l’intégrité des données.

Cette version permet d’évaluer le comportement du défenseur face à une menace plus soutenue, dans un cadre structurellement très proche du scénario de référence.

---

### `data_manipulation_plus_8_clients`

Cette variante conserve l’architecture générale, le type d’attaque et les mécanismes de défense du scénario de référence, mais augmente fortement l’échelle du réseau côté utilisateurs avec :

- huit postes clients ;
- huit agents légitimes.

L’environnement devient plus dense et l’activité utilisateur plus variée, ce qui accroît à la fois :

- les points d’entrée potentiels pour l’attaquant ;
- la charge d’observation et de contrôle pour le défenseur.

Cette version permet d’étudier le passage à une configuration plus large, tout en restant fidèle à la logique du scénario de base.

---

### `data_manipulation_dynamic_10_clients`

Cette variante conserve le même principe d’attaque contre l’intégrité des données et les mêmes mécanismes généraux de défense, mais augmente fortement l’échelle de l’environnement.

Le système comprend ici :

- dix postes clients ;
- une topologie plus étendue ;
- plusieurs clients activés dynamiquement au cours de l’épisode.

Par rapport au scénario de référence, cette version combine :

- une montée en taille du réseau ;
- une augmentation du nombre d’utilisateurs légitimes ;
- une évolution progressive de la surface du système en cours d’exécution.

Elle permet d’étudier le comportement des agents dans un cadre plus large, plus distribué et moins statique que le scénario de base.

## Résumé des variantes

| Fichier | Évolution principale |
|---|---|
| `data_manipulation.yaml` | Scénario de référence |
| `data_manipulation_with_different_connections.yaml` | Variante topologique du réseau |
| `data_manipulation_plus_one_client` | Ajout d’un client statique |
| `data_manipulation_plus_one_dynamic_client` | Ajout d’un client dynamique |
| `data_manipulation_plus_two_dynamic_clients` | Ajout de deux clients dynamiques |
| `data_manipulation_two_attackers` | Ajout d’un second attaquant |
| `data_manipulation_plus_8_clients` | Passage à huit clients |
| `data_manipulation_dynamic_10_clients` | Passage à dix clients avec activations dynamiques |

## Utilisation

Ces fichiers sont consommés par plusieurs modules du projet :

- `GNN/GNN_single_scenario.py` et `GNN/GNN_multiscenario.py` - entraînement PPO-GNN
- `SB3 agent/train.py` - baseline SB3 (PPO, MaskablePPO, A2C, DQN)
- `Parameter optimization/optimize_parameters.py` - optimisation Optuna des hyperparamètres

Les chemins dans ces scripts pointent vers ce dossier à la racine du projet (`Scenarios/`).

## Objectif global

L’ensemble de ces scénarios offre un cadre d’expérimentation cohérent pour analyser la robustesse d’un défenseur face à des menaces visant l’intégrité des données, dans des environnements de complexité croissante.
