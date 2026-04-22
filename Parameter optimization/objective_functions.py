from optuna.exceptions import TrialPruned
from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import MaskablePPO
from stable_baselines3.common.evaluation import evaluate_policy as evaluate_PPO_DQN_A2C
from sb3_contrib.common.maskable.evaluation import evaluate_policy as evaluate_MPPO
from study_algorithm import make_env


def create_objective_ppo(env_config, total_timesteps, n_eval_episodes, tune_seeds):
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.9999)
        n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        ent_coef = trial.suggest_float("ent_coef", 1e-6, 1e-2, log=True)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
        n_epochs = trial.suggest_int("n_epochs", 3, 15)
        gae_lambda = trial.suggest_float("gae_lambda", 0.90, 1.0)

        if batch_size > n_steps or (n_steps % batch_size != 0):
            raise TrialPruned()

        seed_rewards = []
        for idx, seed in enumerate(tune_seeds, start=1):
            train_env = make_env(env_config)
            eval_env = make_env(env_config)
            try:
                model = PPO(
                    "MlpPolicy",
                    train_env,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    ent_coef=ent_coef,
                    clip_range=clip_range,
                    n_epochs=n_epochs,
                    gae_lambda=gae_lambda,
                    seed=seed,
                    verbose=0,
                )
                model.learn(total_timesteps=total_timesteps)
                mean_reward, _ = evaluate_PPO_DQN_A2C(
                    model,
                    eval_env,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,
                )
                seed_rewards.append(float(mean_reward))

                running_mean = sum(seed_rewards) / len(seed_rewards)
                trial.report(running_mean, step=idx)
                if trial.should_prune():
                    raise TrialPruned()
            finally:
                train_env.close()
                eval_env.close()

        trial.set_user_attr("seed_rewards", seed_rewards)
        return sum(seed_rewards) / len(seed_rewards)

    return objective


def create_objective_mppo(env_config, total_timesteps, n_eval_episodes, tune_seeds):
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.9999)
        n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        ent_coef = trial.suggest_float("ent_coef", 1e-6, 1e-2, log=True)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
        n_epochs = trial.suggest_int("n_epochs", 3, 15)
        gae_lambda = trial.suggest_float("gae_lambda", 0.90, 1.0)

        if batch_size > n_steps or (n_steps % batch_size != 0):
            raise TrialPruned()

        seed_rewards = []
        for idx, seed in enumerate(tune_seeds, start=1):
            train_env = make_env(env_config)
            eval_env = make_env(env_config)
            try:
                model = MaskablePPO(
                    "MlpPolicy",
                    train_env,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    ent_coef=ent_coef,
                    clip_range=clip_range,
                    n_epochs=n_epochs,
                    gae_lambda=gae_lambda,
                    seed=seed,
                    verbose=0,
                )
                model.learn(total_timesteps=total_timesteps)
                mean_reward, _ = evaluate_MPPO(
                    model,
                    eval_env,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,
                    use_masking=True,
                )
                seed_rewards.append(float(mean_reward))

                running_mean = sum(seed_rewards) / len(seed_rewards)
                trial.report(running_mean, step=idx)
                if trial.should_prune():
                    raise TrialPruned()
            finally:
                train_env.close()
                eval_env.close()

        trial.set_user_attr("seed_rewards", seed_rewards)
        return sum(seed_rewards) / len(seed_rewards)

    return objective


def create_objective_dqn(env_config, total_timesteps, n_eval_episodes, tune_seeds):
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.9999)
        buffer_size = trial.suggest_categorical(
            "buffer_size", [50_000, 100_000, 200_000]
        )
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        tau = trial.suggest_float("tau", 0.001, 1.0, log=True)
        train_freq = trial.suggest_categorical("train_freq", [1, 4, 8])
        gradient_steps = trial.suggest_categorical("gradient_steps", [1, 4, 8])
        exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
        exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.2)
        target_update_interval = trial.suggest_categorical(
            "target_update_interval", [500, 1000, 5000]
        )

        seed_rewards = []
        for idx, seed in enumerate(tune_seeds, start=1):
            train_env = make_env(env_config)
            eval_env = make_env(env_config)
            try:
                model = DQN(
                    "MlpPolicy",
                    train_env,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    buffer_size=buffer_size,
                    batch_size=batch_size,
                    tau=tau,
                    train_freq=train_freq,
                    gradient_steps=gradient_steps,
                    exploration_fraction=exploration_fraction,
                    exploration_final_eps=exploration_final_eps,
                    target_update_interval=target_update_interval,
                    seed=seed,
                    verbose=0,
                )
                model.learn(total_timesteps=total_timesteps)
                mean_reward, _ = evaluate_PPO_DQN_A2C(
                    model,
                    eval_env,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,
                )
                seed_rewards.append(float(mean_reward))

                running_mean = sum(seed_rewards) / len(seed_rewards)
                trial.report(running_mean, step=idx)
                if trial.should_prune():
                    raise TrialPruned()
            finally:
                train_env.close()
                eval_env.close()

        trial.set_user_attr("seed_rewards", seed_rewards)
        return sum(seed_rewards) / len(seed_rewards)

    return objective


def create_objective_a2c(env_config, total_timesteps, n_eval_episodes, tune_seeds):
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.9999)
        n_steps = trial.suggest_categorical("n_steps", [5, 10, 20, 50, 100])
        ent_coef = trial.suggest_float("ent_coef", 1e-6, 1e-2, log=True)
        gae_lambda = trial.suggest_float("gae_lambda", 0.90, 1.0)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)

        seed_rewards = []
        for idx, seed in enumerate(tune_seeds, start=1):
            train_env = make_env(env_config)
            eval_env = make_env(env_config)
            try:
                model = A2C(
                    "MlpPolicy",
                    train_env,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    n_steps=n_steps,
                    ent_coef=ent_coef,
                    gae_lambda=gae_lambda,
                    vf_coef=vf_coef,
                    seed=seed,
                    verbose=0,
                )
                model.learn(total_timesteps=total_timesteps)
                mean_reward, _ = evaluate_PPO_DQN_A2C(
                    model,
                    eval_env,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,
                )
                seed_rewards.append(float(mean_reward))

                running_mean = sum(seed_rewards) / len(seed_rewards)
                trial.report(running_mean, step=idx)
                if trial.should_prune():
                    raise TrialPruned()
            finally:
                train_env.close()
                eval_env.close()

        trial.set_user_attr("seed_rewards", seed_rewards)
        return sum(seed_rewards) / len(seed_rewards)

    return objective
