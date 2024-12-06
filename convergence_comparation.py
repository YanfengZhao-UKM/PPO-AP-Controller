import numpy as np
from matplotlib import pyplot as plt

saved_folder="./training_model/Convergence/"


#Get convergence data for all_tricks
Max_path=saved_folder + "All_tricks.npy"
episode_rewards_Max = np.load(Max_path)
epinum_Max = len(episode_rewards_Max)
#For visual comparison the first 8000 episodes are taken uniformly at intervals of 20
x_Max = np.arange(0,9000,30)
y_Max = episode_rewards_Max[:9000:30]

#Get convergence data for No_reward_norm
No_reward_norm_path=saved_folder + "No_reward_norm.npy"
episode_No_reward_norm= np.load(No_reward_norm_path)
#epinum_No_tricks = len(episode_No_tricks)
#For visual comparison the first 8000 episodes are taken uniformly at intervals of 20
x_No_reward_norm = np.arange(0,9000,30)
y_No_reward_norm = episode_No_reward_norm[:9000:30]

#Get convergence data for No_obs_norm
No_obs_norm_path=saved_folder + "No_obs_norm.npy"
episode_No_obs_norm= np.load(No_obs_norm_path)
#epinum_No_tricks = len(episode_No_tricks)
#For visual comparison the first 8000 episodes are taken uniformly at intervals of 20
x_No_obs_norm = np.arange(0,9000,30)
y_No_obs_norm = episode_No_obs_norm[:9000:30]

#Get convergence data for No_adv_norm
No_adv_norm_path=saved_folder + "No_adv_norm.npy"
episode_No_adv_norm = np.load(No_adv_norm_path)
#epinum_No_tricks = len(episode_No_tricks)
#For visual comparison the first 8000 episodes are taken uniformly at intervals of 20
x_No_adv_norm = np.arange(0,9000,30)
y_No_adv_norm = episode_No_adv_norm[:9000:30]

#Get convergence data for No_Beta_PDF
No_Beta_PDF_path=saved_folder + "No_Beta_PDF.npy"
episode_No_Beta_PDF = np.load(No_Beta_PDF_path)
#epinum_No_tricks = len(episode_No_tricks)
#For visual comparison the first 8000 episodes are taken uniformly at intervals of 20
x_No_Beta_PDF = np.arange(0,9000,30)
y_No_Beta_PDF = episode_No_Beta_PDF[:9000:30]

#Get convergence data for No_fail_indicator
No_fail_indicator_path=saved_folder + "No_fail_indicator.npy"
episode_No_fail_indicator = np.load(No_fail_indicator_path)
#epinum_No_tricks = len(episode_No_tricks)
#For visual comparison the first 8000 episodes are taken uniformly at intervals of 20
x_No_fail_indicator = np.arange(0,9000,30)
y_No_fail_indicator = episode_No_fail_indicator[:9000:30]

#Get convergence data for No_entropy
No_entropy_path=saved_folder + "No_entropy.npy"
episode_No_entropy = np.load(No_entropy_path)
#epinum_No_tricks = len(episode_No_tricks)
#For visual comparison the first 8000 episodes are taken uniformly at intervals of 20
x_No_entropy = np.arange(0,9000,30)
y_No_entropy = episode_No_entropy[:9000:30]

#Get convergence data for No_learn_decay
No_learn_decay_path=saved_folder + "No_learn_decay.npy"
episode_No_learn_decay = np.load(No_learn_decay_path)
#epinum_No_tricks = len(episode_No_tricks)
#For visual comparison the first 8000 episodes are taken uniformly at intervals of 20
x_No_learn_decay = np.arange(0,9000,30)
y_No_learn_decay = episode_No_learn_decay[:9000:30]

#Get convergence data for No_grad_clip
No_grad_clip_path=saved_folder + "No_grad_clip.npy"
episode_No_grad_clip = np.load(No_grad_clip_path)
#epinum_No_tricks = len(episode_No_tricks)
#For visual comparison the first 8000 episodes are taken uniformly at intervals of 20
x_No_grad_clip = np.arange(0,9000,30)
y_No_grad_clip = episode_No_grad_clip[:9000:30]

#Get convergence data for No_orth_init
No_orth_init_path=saved_folder + "No_orth_init.npy"
episode_No_orth_init = np.load(No_orth_init_path)
#epinum_No_tricks = len(episode_No_tricks)
#For visual comparison the first 8000 episodes are taken uniformly at intervals of 20
x_No_orth_init = np.arange(0,9000,30)
y_No_orth_init = episode_No_orth_init[:9000:30]

#Get convergence data for No_Adam_param
No_Adam_param_path=saved_folder + "No_Adam_param.npy"
episode_No_Adam_param = np.load(No_Adam_param_path)
#epinum_No_tricks = len(episode_No_tricks)
#For visual comparison the first 8000 episodes are taken uniformly at intervals of 20
x_No_Adam_param = np.arange(0,9000,30)
y_No_Adam_param = episode_No_Adam_param[:9000:30]

#Get convergence data for no_tricks
No_path=saved_folder + "No_tricks.npy"
episode_No_tricks = np.load(No_path)
#epinum_No_tricks = len(episode_No_tricks)
#For visual comparison the first 8000 episodes are taken uniformly at intervals of 20
x_No = np.arange(0,9000,30)
y_No = episode_No_tricks[:9000:30]

plt.title('Reward Convergence Curve')
plt.xlabel('Episode')
plt.ylabel('Reward')
#plt.plot(x_withtips,y_withtips,label='Improved Algorithm')
plt.plot(x_Max,y_Max,color='#FF0000',linewidth=2,label='All_tricks')
plt.plot(x_No_orth_init,y_No_orth_init,label='No_orthogonal_init')
plt.plot(x_No_obs_norm,y_No_obs_norm,label='No_state_normalization')
plt.plot(x_No_Beta_PDF,y_No_Beta_PDF,label='No_Beta_distribution')
plt.plot(x_No_fail_indicator,y_No_fail_indicator,label='No_failure_indicator')
plt.plot(x_No_learn_decay,y_No_learn_decay,label='No_learning_rate_decay')
plt.plot(x_No_reward_norm,y_No_reward_norm,label='No_reward_normalization')
plt.plot(x_No_Adam_param,y_No_Adam_param,label='No_Adam_parameter')
plt.plot(x_No_adv_norm,y_No_adv_norm,label='No_adv_normalization')
plt.plot(x_No_grad_clip,y_No_grad_clip,label='No_gradient_clip')
plt.plot(x_No_entropy,y_No_entropy,label='No_policy_entropy')
plt.plot(x_No,y_No,color='#C0C0C0',label='Without_tricks')
plt.legend()
plt.show()








