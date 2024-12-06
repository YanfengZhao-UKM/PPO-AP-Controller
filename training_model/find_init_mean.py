import os

from simglucose.envs.simglucose_gym_env import T1DSimEnv,T1DSimHistoryEnv,T1DSimMergeStateEnv,T1DCHOObsSimEnv
from reward.custom_rewards import myreward
import numpy as np
from simglucose.controller.base import Action

patient = ['adolescent#001','adolescent#002','adolescent#003','adolescent#004','adolescent#005','adolescent#006','adolescent#007','adolescent#008','adolescent#009','adolescent#010',
                    'adult#001','adult#002','adult#003','adult#004','adult#005','adult#006','adult#007','adult#008','adult#009','adult#010',
                    'child#001','child#002','child#003','child#004','child#005','child#006','child#007','child#008','child#009','child#010']

for patient_name in patient:
    env = T1DSimEnv(patient_name,myreward,1,480)
    episodes = 10
    means=np.arange(0.0,1,0.005)
    max_reward = [patient_name, 0, 0]
    for mean in means:
        total_reward = 0
        for epi in range(episodes):
            done = False
            while not done:
                _, reward, done, _ = env.step(mean)
                total_reward += reward
        if total_reward > max_reward[1]:
            max_reward[1] = total_reward
            max_reward[2] = mean
    print(max_reward)
    directory = 'simglucose/envs/'
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, 'best_means.txt'), 'a') as file:
        file.write(f'{max_reward}\n')







