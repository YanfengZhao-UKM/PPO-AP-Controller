import sys
from tqdm import tqdm
from simglucose.envs.simglucose_gym_env import T1DSimEnv,T1DSimHistoryEnv,T1DSimMergeStateEnv,T1DCHOObsSimEnv
from simglucose.simulation.scenario import CustomScenario
import os
import torch
import numpy as np
import argparse
from matplotlib import pyplot as plt
from algorithms.replaybuffer import ReplayBuffer
from algorithms.ppo_clip_improved_totaltimesteps import PPOClip
from reward.custom_rewards import custom_reward, myreward,simple_reward,risk_diff,shaped_reward_around_normal_bg,no_negativity,neg_risk
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
    episodes = 3
    returns = []
    for epi in range(episodes):
        observation = env.reset()
        episode_reward = 0
        episode_cgm_hist = []
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, _ = agent.choose_action(observation)
            insulin = a * get_insulin_coef(env, observation)
            observation, reward, done, _ = env.step(insulin)
            if render and epi==0:
                env.render()
            episode_reward += reward
            cgm_value = observation[0]
            if args.normed_state:
                cgm_value *= 600
            episode_cgm_hist.append(cgm_value)
        returns.append(np.insert(get_returns(episode_cgm_hist), 0, episode_reward))

        mean = np.mean(returns, axis=0)
        max = np.max(returns, axis=0)
        min = np.min(returns, axis=0)
        err = np.max([max-mean, mean-min], axis=0)
    return np.round(mean,2), np.round(err,2), np.round(max,2), np.round(min,2)

# The Safety Adjustment Module to calculate insulin coefficients to avoid postprandial hyperglycaemia and nocturnal hypoglycaemia
def get_insulin_coef(env,obs):
    coef = 1        # default value

    #preventing nocturnal hypoglycaemia
    if (env.env.time.hour<=3):
        coef = coef * 0.6
    if (env.env.time.hour>=20):
        coef =  coef * 0.8

    # preventing postprandial hyperglycaemia
    coef = (obs[0]/112.517) * (obs[0]/obs[1])**2 * coef
    # if (obs[0] >170):
    #     coef = coef * 2
    # Suspension of insulin infusion when risk of hypoglycaemia is predicted
    if (obs[0] < 80):
        coef = 0
    return coef

def main(args, patient_name, seed):

    # Create simulation environment with patient_name
    env = T1DSimHistoryEnv(patient_name=patient_name, reward_fun=myreward, seed=seed, max_episode_steps=args.max_episode_steps)
    eval_env = T1DSimHistoryEnv(patient_name=patient_name, reward_fun=myreward,  seed=seed, max_episode_steps=args.max_episode_steps)

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Obtain the dimensions of the state space abd the action space, and the maximum value of the action space
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = env.action_space.high[0]

    # Set the agent of the PPO-clip algorithm and its replay_buffer
    agent = PPOClip(args)
    replay_buffer = ReplayBuffer(args)

    # Set the saved folder and the best model folder
    saved_folder = './training_model/{}_{}_seed{}_{}/'.format(patient_name, args.policy_dist, seed, env.__class__.__name__)
    best_folder = './best_model//{}_{}_seed{}_{}/'.format(patient_name, args.policy_dist, seed, env.__class__.__name__)
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    if not os.path.exists(best_folder):
        os.makedirs(best_folder)

    # Initiate the time steps and set evaluate parameters
    evaluate_num = 0
    evaluate_rewards = []  # Record the average rewards during the evaluating
    max_evaluate_reward = 0   #Record the maxiumum evaluate reward
    episode_rewards = []   # Record the cumulative reward of every episode
    measure_metrics_records = np.vstack((np.arange(9),np.arange(10,19)))  #Record the measure metrics of every episode

    # Training
    timesteps=0
    #totalmeal=0
    #pbar = tqdm(total=args.max_train_steps, desc='Training Progress') #set the progress bar
    while timesteps < args.max_train_steps:
        # Initiate the environment and the episode parameters
        observation = env.reset()
        episode_steps = 0
        episode_reward = 0
        done = False
        #eat=0
        while not done:
            # update the progress bar
            if timesteps % 4800 == 0:
                print('Training Progress:', timesteps / args.max_train_steps * 100, '%')
            episode_steps += 1

            a, a_logprob = agent.choose_action(observation)  # Select an action and the corresponding log probability
            insulin = a * get_insulin_coef(env, observation)

            ## test the total CHO intake for every meal
            # if (env.patient.planned_meal>0):
            #     eat=1
            #     totalmeal+=env.patient.planned_meal
            # if(eat==1 and env.patient.planned_meal==0):
            #     eat=0
            #     totalmeal = 0
            #     print(totalmeal)

            observation_, reward, done, _ = env.step(insulin) # Interact with the environment in a time step by injecting insulin
            #env.render()  #Animation tracking
            episode_reward += reward # Cumulate the episode reward

            #trick 3: Use failure indicator
            if args.use_failure_indicator:
                # When reaching the max_episode_steps or the patient is in an emergency condition, done will be Ture, we need to distinguish them;
                # dw means there is an emergency event,there is no next state observation_;
                # but when reaching the max_episode_steps,there is a next observation_ actually.
                if done and episode_steps <= args.max_episode_steps:
                    dw = True
                else:
                    dw = False
                # Take the 'action'，but store the original 'a'（especially for Beta）
                replay_buffer.store(observation, a, a_logprob, reward, observation_, dw, done)
            else:
                replay_buffer.store(observation, a, a_logprob, reward, observation_, False, done)

            # Prepare for next episode
            observation = observation_
            timesteps += 1
    #
            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, timesteps)
                replay_buffer.count = 0

            # When reaching the eval_freq
            if timesteps % args.eval_freq == 0:
                mean, err, max, min = evaluate_policy(args, eval_env, agent, render = False)
                evaluate_reward = mean[0]
                evaluate_rewards.append(evaluate_reward)
                evaluate_num += 1
                print("evaluate_num:{} \t evaluate_reward:{} +/- {} \t".format(evaluate_num, evaluate_reward, err[0]))
                if evaluate_reward > max_evaluate_reward:
                    max_evaluate_reward = evaluate_reward
                    print("The new best evaluate reward!")
                    # Save the best model
                    actor_model_path = os.path.join(best_folder,'best_actor_model.pth')
                    critic_model_path = os.path.join(best_folder,'best_critic_model.pth')
                    torch.save(agent.actor.state_dict(), actor_model_path)
                    torch.save(agent.actor.state_dict(), critic_model_path)

                measure_metrics = np.vstack((mean, err, max, min))
                measure_metrics_records = np.vstack((measure_metrics_records, measure_metrics))



            ## When reaching the save_freq
            #if timesteps % args.save_freq == 0:
            #    actor_model_path = os.path.join(saved_folder,'actor_model_{}_steps.pth'.format(timesteps))
            #    critic_model_path = os.path.join(saved_folder,'critic_model_{}_steps.pth'.format(timesteps))
            #    torch.save(agent.actor.state_dict(), actor_model_path)
            #    torch.save(agent.actor.state_dict(), critic_model_path)

        # Record the cumulative reward of each episode
        episode_rewards.append(episode_reward)


    # Save the measure matrics to file on evaluation mode
    measure_metrics_path = os.path.join(saved_folder, 'mesure_metrics.npy')
    np.save(measure_metrics_path, measure_metrics_records)

    # Save the measure matrics to file on training mode
    # episode_rewards_path = os.path.join(saved_folder, 'No_policy_entropy.npy')
    # np.save(episode_rewards_path, episode_rewards)

    # #  Plot the episode rewards to check convergence
    # x = np.arange(len(episode_rewards))
    # plt.plot(x,episode_rewards)
    # plt.title('Reward Convergence Curve')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # image_path="images/reward_convergence/"
    # if not os.path.exists(image_path):
    #     os.makedirs(image_path)
    # plt.savefig('images/reward_convergence/{}_{}_seed{}_{}.png'.format(patient_name,
    #                     args.policy_dist, seed, env.__class__.__name__), dpi=300, bbox_inches='tight')
    # plt.show()

# Set the input parameters of the main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=4800, help="Maximum number of training days")
    parser.add_argument("--max_episode_steps", type=int, default=480, help="Maximum number of steps in a episode, and evaluate the policy every episode")
    parser.add_argument("--eval_freq", type=int, default=480, help="Evaluate frequency")
    parser.add_argument("--save_freq", type=int, default=960, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Beta", help="Trick 1: Beta or Gaussian PDF")
    parser.add_argument("--dist_init_param1", type=float, default=1.1, help="Initial parameter for Beta dist")
    parser.add_argument("--dist_init_param2", type=float, default=79, help="Initial parameter for Beta dist")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=16, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=2e-5, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=2e-5, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.1, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=20, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 2:advantage normalization")
    parser.add_argument("--use_failure_indicator", type=bool, default=True, help="Trick 3:treatment failure indicator")
    parser.add_argument("--normed_state", type=bool, default=False, help="Trick 4:state normalization")
    parser.add_argument("--entropy_coef", type=float, default=0.02, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 7:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 7:reward scaling")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 8: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 9: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 10: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=False, help="Trick 11: tanh activation function")

    args = parser.parse_args()

    patient = ['adolescent#001','adolescent#002','adolescent#003','adolescent#004','adolescent#005','adolescent#006','adolescent#007','adolescent#008','adolescent#009','adolescent#010',
                    'adult#001','adult#002','adult#003','adult#004','adult#005','adult#006','adult#007','adult#008','adult#009','adult#010',
                    'child#001','child#002','child#003','child#004','child#005','child#006','child#007','child#008','child#009','child#010']
    patient_idx=1
    main(args, patient_name=patient[patient_idx-1], seed=1)




