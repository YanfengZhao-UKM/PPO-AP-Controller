##IMPORTANT##!!!need set environment according to the training env
from simglucose.envs.simglucose_gym_env import T1DSimEnv,T1DSimHistoryEnv,T1DSimMergeStateEnv,T1DCHOObsSimEnv
import os
import torch
import numpy as np
import argparse
from matplotlib import pyplot as plt
from algorithms.replaybuffer import ReplayBuffer
from algorithms.ppo_clip_improved_totaltimesteps import Actor_Beta
from reward.custom_rewards import myreward
from simglucose.analysis.risk import risk_index

# Calculate the evaluation metrics
def get_returns(episode_cgm_hist):
    cgm_num =  len(episode_cgm_hist)
    TAR_S = len([p for p in episode_cgm_hist if p > 300]) / cgm_num
    TAR = len([p for p in episode_cgm_hist if 300 >= p > 180]) / cgm_num
    TIR = len([p for p in episode_cgm_hist if 180 >= p >= 70]) / cgm_num
    TBR = len([p for p in episode_cgm_hist if 70 > p >= 50]) / cgm_num
    TBR_S = len([p for p in episode_cgm_hist if p < 50]) / cgm_num

    fBG = 1.509 * (np.log(episode_cgm_hist) ** 1.084 - 5.381)
    rl = 10 * fBG[fBG < 0] ** 2
    rh = 10 * fBG[fBG > 0] ** 2
    LBGI = np.nan_to_num(np.mean(rl))
    HBGI = np.nan_to_num(np.mean(rh))
    RI = LBGI + HBGI

    return TAR_S, TAR, TIR, TBR, TBR_S, LBGI, HBGI, RI

# For evaluating the policy
def evaluate_policy(args, env, agent, render=False):
    episodes = 2
    returns = []
    for epi in range(episodes):
        observation = env.reset()
        episode_reward = 0
        episode_cgm_hist = []
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            observation = torch.unsqueeze(torch.tensor(observation, dtype=torch.float), 0).to("cuda")
            with torch.no_grad():
                dist = agent.get_dist(observation)
                a = dist.sample()
                insulin = torch.clamp(a, 0, args.max_action).cpu().detach().numpy().flatten()  # [-max,max] 改为 [0,max]
                observation, reward, done, _ = env.step(insulin)
            if render and epi==0:
                env.render()
            episode_reward += reward
            episode_cgm_hist.append(observation[0])
        returns.append(np.insert(get_returns(episode_cgm_hist), 0, episode_reward))

        mean = np.mean(returns, axis=0)
        max = np.max(returns, axis=0)
        min = np.min(returns, axis=0)
        std = np.max([max-mean, mean-min], axis=0)
    return np.round(mean,2), np.round(std,2)

def render(measure_metrics_records):

    # Generate data from the measure_metrics_records
    x = np.arange(int(len(measure_metrics_records) / 4))
    TAR_S = measure_metrics_records[0::4, 1].flatten() * 100
    TAR = measure_metrics_records[0::4, 2].flatten() * 100
    TIR = measure_metrics_records[0::4, 3].flatten() * 100
    TBR = measure_metrics_records[0::4, 4].flatten() * 100
    TBR_S = measure_metrics_records[0::4, 5].flatten() * 100
    TAR_S_mean = round(np.mean(TAR_S), 2)
    TAR_mean = round(np.mean(TAR), 2)
    TIR_mean = round(np.mean(TIR), 2)
    TBR_mean = round(np.mean(TBR), 2)
    TBR_S_mean = round(np.mean(TBR_S), 2)
    TAR_S_max = measure_metrics_records[2::4, 1].flatten() * 100
    TAR_max = measure_metrics_records[2::4, 2].flatten() * 100
    TIR_max = measure_metrics_records[2::4, 3].flatten() * 100
    TBR_max = measure_metrics_records[2::4, 4].flatten() * 100
    TBR_S_max = measure_metrics_records[2::4, 5].flatten() * 100
    TAR_S_min = measure_metrics_records[2::4, 1].flatten() * 100
    TAR_min = measure_metrics_records[3::4, 2].flatten() * 100
    TIR_min = measure_metrics_records[3::4, 3].flatten() * 100
    TBR_min = measure_metrics_records[3::4, 4].flatten() * 100
    TBR_S_min = measure_metrics_records[3::4, 5].flatten() * 100

    Reward = measure_metrics_records[0::4, 0].flatten()
    Reward_mean = round(np.mean(Reward), 2)
    Reward_max = measure_metrics_records[2::4, 0].flatten()
    Reward_min = measure_metrics_records[3::4, 0].flatten()
    LBGI = measure_metrics_records[0::4, 6].flatten()
    HBGI = measure_metrics_records[0::4, 7].flatten()
    RISK = measure_metrics_records[0::4, 8].flatten()
    LBGI_mean = round(np.mean(LBGI), 2)
    HBGI_mean = round(np.mean(HBGI), 2)
    RISK_mean = round(np.mean(RISK), 2)
    LBGI_max = measure_metrics_records[2::4, 6].flatten()
    HBGI_max = measure_metrics_records[2::4, 7].flatten()
    RISK_max = measure_metrics_records[2::4, 8].flatten()
    LBGI_min = measure_metrics_records[3::4, 6].flatten()
    HBGI_min = measure_metrics_records[3::4, 7].flatten()
    RISK_min = measure_metrics_records[3::4, 8].flatten()

    # Create the time of the glucose range axis
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 6))

    # 左侧子图占50%
    ax1.set_position([0.1, 0.1, 0.4, 0.8])

    # 右侧上下两个子图各占25%
    ax2.set_position([0.6, 0.6, 0.35, 0.35])
    ax3.set_position([0.6, 0.1, 0.35, 0.35])

    # Plot the time of the glucose range lines
    ax1.plot(x, TAR_S, label='Severe Hyper:'+ str(TAR_S_mean))
    ax1.fill_between(x, TAR_max, TAR_S_min, alpha=0.2)

    ax1.plot(x, TAR, label='Hyper:'+ str(TAR_mean))
    ax1.fill_between(x, TAR_max, TAR_min, alpha=0.2)

    ax1.plot(x, TIR, label='Normal:'+str(TIR_mean))
    ax1.fill_between(x, TIR_max, TIR_min, alpha=0.2)

    ax1.plot(x, TBR, label='Hypo:'+str(TBR_mean))
    ax1.fill_between(x, TBR_max, TBR_min, alpha=0.2)

    ax1.plot(x, TBR_S, label='Severe Hypo:'+str(TBR_S_mean))
    ax1.fill_between(x, TBR_S_max, TBR_S_min, alpha=0.2)

    ax1.set_title('Glucose range time percent')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Percent')
    ax1.legend()

    # Create the reward and risk axes
    #fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(10, 8))

    ax2.plot(x, Reward, label='Reward:'+ str(Reward_mean))
    ax2.fill_between(x, Reward_max, Reward_min, alpha=0.2)

    ax2.set_title('Reward Plot')
    #ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.legend()

    #fig, ax3 = plt.subplots(figsize=(4, 8))

    ax3.plot(x, HBGI, label='HBGI:'+ str(HBGI_mean))
    ax3.fill_between(x, HBGI_max, HBGI_min, alpha=0.2)

    ax3.plot(x, LBGI, label='LBGI:'+ str(LBGI_mean))
    ax3.fill_between(x, LBGI_max, LBGI_min, alpha=0.2)

    ax3.plot(x, RISK, label='RISK:' + str(RISK_mean))
    ax3.fill_between(x, RISK_max, RISK_min, alpha=0.2)

    ax3.set_title('RISK Plot')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('RISK')
    ax3.legend()

    plt.show()

def main(args, patient_name, seed):

    # Create simulation environment with patient_name
    env = T1DSimHistoryEnv(patient_name=patient_name, reward_fun=myreward,seed=seed, max_episode_steps=args.max_episode_steps)
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = env.action_space.high[0]

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Obtain the dimensions of the state space abd the action space, and the maximum value of the action space
    # args.state_dim = env.observation_space.shape[0]
    # args.action_dim = env.action_space.shape[0]
    # args.max_action = env.action_space.high[0]

    # Load the best model
    best_folder = './best_model//{}_{}_seed{}_{}/'.format(patient_name, args.policy_dist, seed, env.__class__.__name__)
    actor_model_path = os.path.join(best_folder, 'best_actor_model.pth')
    actor_model = Actor_Beta(args)
    actor_model.load_state_dict(torch.load(actor_model_path))
    actor_model.to("cuda")

    measure_metrics_records = np.vstack(
        (np.arange(9), np.arange(10, 19)))  # Record the measure metrics of every episode

    evaluate_num = 30
    for num in range(evaluate_num):
        mean,std = evaluate_policy(args, env, actor_model)
        measure_metrics = np.vstack((mean, std))
        measure_metrics_records = np.vstack((measure_metrics_records, measure_metrics))
        print(f"{(num+1)/evaluate_num*100}% evaluation process is finished.")
    render(measure_metrics_records[2:])

# Set the input parameters of the main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(4800), help="Maximum number of training days")
    parser.add_argument("--max_episode_steps", type=int, default=480, help="Maximum number of steps in a episode, and evaluate the policy every episode")
    parser.add_argument("--eval_freq", type=int, default=480, help="Evaluate frequency")
    parser.add_argument("--save_freq", type=int, default=960, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Beta", help="Beta or Gaussian")
    parser.add_argument("--dist_init_param1", type=float, default=1.1, help="Initial_mean for Gaussian OR Init_alpha for Beta")
    parser.add_argument("--dist_init_param2", type=float, default=79, help="Initia_log_std for Gaussian OR Init_beta for Beta")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=16, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=2e-5, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=2e-5, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.1, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=20, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--normed_state", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 3:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.02, help="Trick 4: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 5:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 6: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 7: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 8: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=False, help="Trick 9: tanh activation function")

    args = parser.parse_args()

    patient = ['adolescent#001','adolescent#002','adolescent#003','adolescent#004','adolescent#005','adolescent#006','adolescent#007','adolescent#008','adolescent#009','adolescent#010',
                    'adult#001','adult#002','adult#003','adult#004','adult#005','adult#006','adult#007','adult#008','adult#009','adult#010',
                    'child#001','child#002','child#003','child#004','child#005','child#006','child#007','child#008','child#009','child#010']
    patient_idx = 1
    main(args, patient_name=patient[patient_idx-1], seed=1)





