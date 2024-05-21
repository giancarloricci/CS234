import argparse
import pathlib
import time
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common.evaluation import evaluate_policy
from my_lander import LunarLander, WindMode
from util import export_plot

def evaluate(env, policy):
    model_return = 0
    obs, _ = env.reset()
    T = 1000
    for _ in range(T):
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        model_return += reward
        if done:
            break
    return model_return


class EvalCallback(sb3.common.callbacks.BaseCallback):
    def __init__(self, eval_period, num_episodes, env, policy):
        super().__init__()
        self.eval_period = eval_period
        self.num_episodes = num_episodes
        self.env = env
        self.policy = policy

        self.returns = []

    def _on_step(self):
        if self.n_calls % self.eval_period == 0:
            print(f"Evaluating after {self.n_calls} steps")
            model_returns = []
            for _ in range(self.num_episodes):
                model_returns.append(evaluate(self.env, self.policy))
            self.returns.append(np.mean(model_returns))
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rl-steps",
        type=int,
        help="The number of learning iterations",
        default=1000000,
    )
    parser.add_argument("--seed", default=0)

    parser.add_argument(
        "--wind-mode",
        type=int,
        help="The type of wind",
        default=0,
    )
    args = parser.parse_args()

    output_path = pathlib.Path(__file__).parent.joinpath(
        "results",
        f"lunar={args.seed},wind-mode={args.wind_mode}",
    )

    model_output = output_path.joinpath("model.zip")
    log_path = output_path.joinpath("log.txt")
    scores_output = output_path.joinpath("scores.npy")
    plot_output = output_path.joinpath("scores.png")
    env = LunarLander(render_mode="human", wind_mode=WindMode(args.wind_mode))

    agent = sb3.PPO("MlpPolicy", env, verbose=1)
    eval_callback = EvalCallback(
       args.rl_steps // 100,
       10,
       env,
       lambda obs: agent.predict(obs)[0],
    )

    # train the model
    start = time.perf_counter()
    agent.learn(args.rl_steps, callback=eval_callback)
    end = time.perf_counter()

    # Log the results
    returns = eval_callback.returns
    if not output_path.exists():
        output_path.mkdir(parents=True)
    agent.save(model_output)
    with open(log_path, "w") as f:
        f.write(f"Wall time elapsed: {end-start:.2f}s\n")
    np.save(scores_output, returns)
    export_plot(returns, "Returns", "lunar", plot_output)

    # get stats 
    mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes=10)
    print(f"mean reward: {mean_reward}, std_reward: {std_reward}")

    # observe policy
    # print("rendering leanred policy")
    # env = agent.get_env()
    # obs = env.reset()
    # while True:
    #     action, _states = agent.predict(obs, deterministic=True)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render("human")