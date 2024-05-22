import os
from datetime import datetime
import torch
import numpy as np
from my_lander import LunarLander, WindMode
from simple_ppo import PPO

import os
from datetime import datetime
import torch
import numpy as np
from my_lander import LunarLander, WindMode
from simple_ppo import PPO
import argparse

class TrainingConfig:
    def __init__(self, args):
        # Environment hyperparameters
        self.max_ep_len = 400
        self.max_training_timesteps = int(1e5)
        self.print_freq = self.max_ep_len * 4
        self.log_freq = self.max_ep_len * 2
        self.save_model_freq = int(2e4)
        self.action_std = None
        self.wind_interval = args.wind_interval
        self.wind_mode = args.wind_mode
        self.enable_wind = args.enable_wind
        self.l2_init = args.l2_init
        self.save_weights = args.save_weights
        self.load_pretrained = args.load_pretrained

        self.env_name = f"crelu={args.crelu}, wind={args.enable_wind}, l2_init={args.l2_init}"

        # PPO hyperparameters
        self.update_timestep = self.max_ep_len * 3
        self.K_epochs = 30
        self.eps_clip = 0.2
        self.gamma = 0.99
        self.lr_actor = 0.0003
        self.lr_critic = 0.001
        self.use_crelu = args.crelu
        self.random_seed = 0

class Trainer:
    def __init__(self, config):
        self.config = config

    def train(self):
        env = self._create_environment()
        ppo_agent = self._initialize_agent(env)
        log_f_name, checkpoint_path = self._setup_logging_and_checkpointing()
        self._print_hyperparameters(env)
        
        start_time = datetime.now().replace(microsecond=0)
        log_f = open(log_f_name, "w+")
        log_f.write('episode,timestep,reward\n')
        
        print_running_reward, print_running_episodes, log_running_reward, log_running_episodes = 0, 0, 0, 0
        time_step, i_episode = 0, 0

        while time_step <= self.config.max_training_timesteps:
            state, _ = env.reset()
            current_ep_reward = 0

            for _ in range(1, self.config.max_ep_len + 1):
                action = ppo_agent.select_action(state)
                state, reward, done, _, _ = env.step(action)

                self._update_buffer(ppo_agent, reward, done)
                time_step, current_ep_reward = self._update_episode(time_step, reward, current_ep_reward)

                if time_step % self.config.update_timestep == 0:
                    ppo_agent.update()

                self._log_progress(log_f, time_step, i_episode, log_running_reward, log_running_episodes)
                self._print_progress(time_step, i_episode, print_running_reward, print_running_episodes)

                if time_step % self.config.save_model_freq == 0:
                    self._save_model(ppo_agent, checkpoint_path, start_time)

                if done:
                    break

            print_running_reward, print_running_episodes = self._update_print_variables(print_running_reward, print_running_episodes, current_ep_reward)
            log_running_reward, log_running_episodes = self._update_log_variables(log_running_reward, log_running_episodes, current_ep_reward)
            i_episode += 1

        self._finalize_training(env, log_f, start_time)

    def _create_environment(self):
        env = LunarLander(render_mode="human", wind_mode=WindMode(self.config.wind_mode), wind_interval=self.config.wind_interval, enable_wind=self.config.enable_wind)
        return env

    def _initialize_agent(self, env):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        ppo_agent = PPO(state_dim, action_dim, self.config.lr_actor, self.config.lr_critic, self.config.gamma, self.config.K_epochs, self.config.eps_clip, self.config.use_crelu, self.config.l2_init, self.config.save_weights)
        if self.config.load_pretrained:
          str_crelu = "True" if self.config.use_crelu else "False"
          checkpoint_path = "crelu="+str_crelu
          print("loading network from : " + checkpoint_path)
          ppo_agent.load(checkpoint_path)
     
        # preTrained weights directory

        random_seed = 0             #### set this to load a particular checkpoint trained on random seed
        run_num_pretrained = 0      #### set this to load a particular checkpoint num

        env_name = "LunarLander-v2"
        directory = "PPO_preTrained" + '/' + env_name + '/'
        checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
        print("loading network from : " + checkpoint_path)
        ppo_agent.load(checkpoint_path)
        return ppo_agent

    def _setup_logging_and_checkpointing(self):
        env_name = self.config.env_name
        log_dir = "PPO_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = os.path.join(log_dir, env_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)
        log_f_name = os.path.join(log_dir, f'PPO_{env_name}_log_{run_num}.csv')

        directory = "PPO_preTrained"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = os.path.join(directory, env_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        checkpoint_path = os.path.join(directory, f'PPO_{env_name}_{self.config.random_seed}.pth')
        return log_f_name, checkpoint_path

    def _print_hyperparameters(self, env):
        print("--------------------------------------------------------------------------------------------")
        print("max training timesteps : ", self.config.max_training_timesteps)
        print("max timesteps per episode : ", self.config.max_ep_len)
        print("model saving frequency : " + str(self.config.save_model_freq) + " timesteps")
        print("log frequency : " + str(self.config.log_freq) + " timesteps")
        print("printing average reward over episodes in last : " + str(self.config.print_freq) + " timesteps")
        print("--------------------------------------------------------------------------------------------")
        print("state space dimension : ", env.observation_space.shape[0])
        print("action space dimension : ", env.action_space.n)
        print("--------------------------------------------------------------------------------------------")
        print("Use CReLU activation function: ", self.config.use_crelu)
        print("Use L2-init: ", self.config.l2_init)
        print("Saving initial weights: ", self.config.save_weights)
        print("Loading pre-trained model: ", self.config.load_pretrained)
        print("Wind mode: ", self.config.wind_mode)
        print("Wind interval: ", self.config.wind_interval)
        print("Enable wind: ", self.config.enable_wind)
        print("--------------------------------------------------------------------------------------------")
        print("PPO update frequency : " + str(self.config.update_timestep) + " timesteps")
        print("PPO K epochs : ", self.config.K_epochs)
        print("PPO epsilon clip : ", self.config.eps_clip)
        print("discount factor (gamma) : ", self.config.gamma)
        print("--------------------------------------------------------------------------------------------")
        print("optimizer learning rate actor : ", self.config.lr_actor)
        print("optimizer learning rate critic : ", self.config.lr_critic)
        if self.config.random_seed:
            print("--------------------------------------------------------------------------------------------")
            print("setting random seed to ", self.config.random_seed)
            torch.manual_seed(self.config.random_seed)
            env.seed(self.config.random_seed)
            np.random.seed(self.config.andom_seed)

    def _update_buffer(self, agent, reward, done):
        agent.buffer.rewards.append(reward)
        agent.buffer.is_terminals.append(done)

    def _update_episode(self, time_step, reward, current_ep_reward):
        time_step += 1
        current_ep_reward += reward
        return time_step, current_ep_reward

    def _log_progress(self, log_f, time_step, i_episode, log_running_reward, log_running_episodes):
        if time_step % self.config.log_freq == 0:
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)
            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f.flush()
            log_running_reward = 0
            log_running_episodes = 0

    def _print_progress(self, time_step, i_episode, print_running_reward, print_running_episodes):
        if time_step % self.config.print_freq == 0:
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)
            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
            print_running_reward = 0
            print_running_episodes = 0

    def _save_model(self, agent, checkpoint_path, start_time):
        print("--------------------------------------------------------------------------------------------")
        print("saving model at : " + checkpoint_path)
        agent.save(checkpoint_path)
        print("model saved")
        print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
        print("--------------------------------------------------------------------------------------------")

    def _update_print_variables(self, print_running_reward, print_running_episodes, current_ep_reward):
        print_running_reward += current_ep_reward
        print_running_episodes += 1
        return print_running_reward, print_running_episodes

    def _update_log_variables(self, log_running_reward, log_running_episodes, current_ep_reward):
        log_running_reward += current_ep_reward
        log_running_episodes += 1
        return log_running_reward, log_running_episodes

    def _finalize_training(self, env, log_f, start_time):
        log_f.close()
        env.close()
        end_time = datetime.now().replace(microsecond=0)
        print("============================================================================================")
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print("============================================================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PPO Training Configuration")
    parser.add_argument("--crelu", type=str, help="Use CReLU", default="False")
    parser.add_argument("--wind-mode", type=int, help="The type of wind", default=3)
    parser.add_argument("--wind-interval", type=int, help="The duration of wind flips", default=100000)
    parser.add_argument("--enable-wind", type=str, help="enable wind", default="True")
    parser.add_argument("--l2-init", type=str, help="enable l2_init", default="False")
    parser.add_argument("--save_weights", type=str, help="save defualt weights (for use in l2 init)", default="False")
    parser.add_argument("--load_pretrained", type=str, help="load default model", default="False")

    args = parser.parse_args()
    args.crelu = args.crelu.lower() == "true"
    args.enable_wind = args.enable_wind.lower() == "true"
    args.l2_init = args.l2_init.lower() == "true"
    args.save_weights= args.save_weights.lower() == "true"
    args.load_pretrained = args.load_pretrained.lower() == "true"

    config = TrainingConfig(args)
    trainer = Trainer(config)
    trainer.train()
        