from policy_gradient import learn_reinforce, policy as pg_policy
from q_learning import learn_q_table, greedy_policy
from visualize import visualize
from save_load import train_and_save, load_only, list_saves, print_save_info

ALGOS = {
    "q_table":   (learn_q_table,   greedy_policy, "Q-Learning"),
    "reinforce": (learn_reinforce, pg_policy,      "REINFORCE"),
}


def menu():
    saves = list_saves()

    print("\n=== Mouse RL ===")
    print("1. Entraîner  —  Q-Learning")
    print("2. Entraîner  —  REINFORCE")

    if saves:
        print("\nFichiers disponibles :")
        for i, entry in enumerate(saves, start=3):
            label = ALGOS.get(entry["algo_name"], (None, None, entry["algo_name"]))[2]
            print(f"\n{i}. Charger  —  {label}")
            print_save_info(entry["filename"], entry["meta"])

    print()
    choice = input("Choix : ").strip()

    if choice == "1":
        name = "q_table"
        learn_fn, pol, _ = ALGOS[name]
        visualize(train_and_save(learn_fn, name), pol)
    elif choice == "2":
        name = "reinforce"
        learn_fn, pol, _ = ALGOS[name]
        visualize(train_and_save(learn_fn, name), pol)
    else:
        idx = int(choice) - 3 if choice.isdigit() else -1
        if 0 <= idx < len(saves):
            entry = saves[idx]
            _, pol, _ = ALGOS.get(entry["algo_name"], (None, greedy_policy, None))
            visualize(load_only(entry), pol)
        else:
            print("Choix invalide.")


menu()
