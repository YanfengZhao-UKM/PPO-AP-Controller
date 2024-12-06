import os

from simglucose.simulation.env import T1DSimEnv
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
from simglucose.simulation.rendering import Viewer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gym
from tqdm import tqdm
import numpy as np
from collections import deque
import argparse
from collections import namedtuple
from datetime import timedelta
from datetime import datetime
from matplotlib import pyplot as plt

from algorithms.policies import ContinuousPolicy
from algorithms.utils import compute_loss
from algorithms.utils import CVGA_score
from algorithms.APState import APState
from algorithms.replaybuffer import ReplayBuffer
from algorithms.ppo_clip_improved import PPOClip
from algorithms.normalization import Normalization, RewardScaling
from env.simglucose_gym_env import T1DSimHistoryEnv
from reward.custom_rewards import no_negativityV2, simple_reward,no_negativity

def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        episode_steps=0
        while not done and episode_steps < args.max_episode_steps:
            episode_steps += 1
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            insulin = a
            s_, r, done, _ = env.step(insulin)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
            #print('s:{},r:{},done:{}'.format(s[-1], r, done))
        evaluate_reward += episode_reward

    return evaluate_reward / times

def render(days, bghist, insulinhist, rewardhist):

    x_label=np.arange(days*480)
    fig, axes = plt.subplots(3)

    axes[0].set_ylabel('BG (mg/dL)')
    axes[1].set_ylabel('Insulin (U/min)')
    axes[2].set_ylabel('Reward')

    axes[0].plot(x_label, bghist, label='CGM')
    axes[1].plot(x_label, insulinhist, label='Insulin')
    axes[2].plot(x_label, rewardhist, label='Reward')

    for ax in axes:
        ax.legend()

    # Plot zone patches
    axes[0].axhspan(70, 180, alpha=0.3, color='limegreen', lw=0)
    axes[0].axhspan(50, 70, alpha=0.3, color='red', lw=0)
    axes[0].axhspan(39, 50, alpha=0.3, color='darkred', lw=0)
    axes[0].axhspan(180, 250, alpha=0.3, color='red', lw=0)
    # axes[0].axhspan(300, 600, alpha=0.3, color='darkred', lw=0)

    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(labelbottom=False)
    #axes[2].tick_params(labelbottom=False)
    plt.show()

    #axes[0].set_title(args.patient_name)

def main(args, patient_name, seed):

    # create simulation environment with patient_name
    env = T1DSimHistoryEnv(patient_name, reward_fun=no_negativityV2, seed=seed, number_of_last_obs=20)
    env_evaluate = T1DSimHistoryEnv(patient_name, reward_fun=no_negativityV2, seed=seed, number_of_last_obs=20)

    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = 20
    args.action_dim = 1
    args.max_action = 1
    args.max_episode_steps = 480 * 10  # Maximum number of steps per episode

    replay_buffer = ReplayBuffer(args)
    agent = PPOClip(args)
    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/PPO_Clip/env_{}_{}_seed_{}'
                 .format(patient_name, args.policy_dist,seed))

    total_steps=0    # Record the total steps during the training
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating

    if args.use_state_norm:
        state_norm = Normalization(shape=args.state_dim)

    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    # bg_hist = []
    # insulin_hist = []
    # reward_hist = []
    # rewardlist=[]

    while total_steps < args.max_train_steps:
        observation = env.reset()
        if args.use_state_norm:
            observation = state_norm(observation)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        # epi_reward=0
        while not done and episode_steps < args.max_episode_steps:
            episode_steps += 1
            a, a_logprob = agent.choose_action(observation)  # Action and the corresponding log probability
            #print(a,a_logprob)
            insulin = a
            observation_, reward, done, _ = env.step(insulin)

            if args.use_reward_norm:
                reward = reward_norm(reward)
            elif args.use_reward_scaling:
                reward = reward_scaling(reward)
            if args.use_state_norm:
                observation_ = state_norm(observation_)

            if done and episode_steps < args.max_episode_steps:
                dw = True
                print('病人在第{}步发生紧急情况->BG:{},insulin:{}'.format(total_steps,observation_[-1],insulin))
            else:
                dw = False

            # bg_hist.append(observation_[-1])
            # insulin_hist.append(insulin)
            # reward_hist.append(reward)

            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(observation, a, a_logprob, reward, observation_, dw, done)
            observation = observation_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_{}'.format(patient_name), evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    file_path = 'policies/PPOClip_{}_{}_seed_{}.npy'.format(
                        args.policy_dist, patient_name, seed)
                    # 检查目录是否存在，如果不存在则创建e:
                    E
                    directory = os.path.dirname(file_path)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    # 保存数据到文件
                    np.save(file_path, evaluate_rewards)

    x = np.arange(len(evaluate_rewards))
    plt.plot(x,evaluate_rewards)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(9600),
                                        help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=480,
                                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=320, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=20, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=32, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-5, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-5, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.1, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=20, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.02, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=False, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    patient = ['adolescent#001','adolescent#002','adolescent#003','adolescent#004','adolescent#005','adolescent#006','adolescent#007','adolescent#008','adolescent#009','adolescent#010',
                    'adult#001','adult#002','adult#003','adult#004','adult#005','adult#006','adult#007','adult#008','adult#009','adult#010',
                    'child#001','child#002','child#003','child#004','child#005','child#006','child#007','child#008','child#009','child#010']
    patient_idx = 16

    main(args, patient_name=patient[patient_idx-1], seed=1)




