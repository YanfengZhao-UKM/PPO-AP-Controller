import random

import gymnasium
from gymnasium.envs.registration import register
from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action
import pandas as pd
import numpy as np
import pkg_resources
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from datetime import datetime
from simglucose.envs.APState import APState

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')

patient_params = pd.read_csv(PATIENT_PARA_FILE)
patient_names = patient_params['Name'].values

# Wrap the original T1DsimEnv with the gym, it can randomly choose a patient and
# assign specific reward function and seed, be added an param of max edpisode steps
class T1DSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None, max_episode_steps = 480):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            # patient_name = 'adolescent#001'
            patient_name = random.choice(patient_names)
            print(patient_name)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.env, _, _, _ = self._create_env_from_random_state()
        self.max_episode_steps=max_episode_steps
        self.current_step=0

    def step(self, action):
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            observation, reward, done, info = self.env.step(act)
        else:
            observation, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
        self.current_step += 1
        done = done or self.current_step >= self.max_episode_steps
        return observation, reward, done, info

    def reset(self):
        self.current_step = 0
        self.env, _, _, _ = self._create_env_from_random_state()
        obs, _, _, _ = self.env.reset()
        return obs

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = hash(self.np_random.integers(0, 1000)) % 2 ** 31
        seed3 = hash(seed2 + 1) % 2 ** 31
        seed4 = hash(seed3 + 1) % 2 ** 31

        hour = self.np_random.integers(low=0.0, high=24.0)
        current_date = datetime.today()
        year = current_date.year
        month = current_date.month
        day = current_date.day
        start_time = datetime(year, month, day, hour, 0, 0)
        self.patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(self.patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(1,))

# State space include the difference between the current and the previous CGM value
class T1DSimDiffEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            # patient_name = 'adolescent#001'
            patient_name = random.choice(patient_names)
            print(patient_name)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.env, _, _, _ = self._create_env_from_random_state()
        self.prev_cgm = None

    def step(self, action):
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        observation, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
        observation = self.update_to_full_obs(observation)
        return observation, reward, done, info

    def reset(self):
        self.prev_cgm = None
        self.env, _, _, _ = self._create_env_from_random_state()
        par_obs, _, _, _ = self.env.reset()
        return self.update_to_full_obs(par_obs)

    def update_to_full_obs(self, partial_obs):
        diff = self.calculate_cgm_diff(partial_obs[0])
        self.prev_cgm = partial_obs[0]
        return [partial_obs[0], diff]

    def calculate_cgm_diff(self, current_cgm):
        if self.prev_cgm is None:
            self.prev_cgm = current_cgm
        diff = current_cgm - self.prev_cgm

        return diff

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = hash(self.np_random.integers(0, 1000)) % 2 ** 31
        seed3 = hash(seed2 + 1) % 2 ** 31
        seed4 = hash(seed3 + 1) % 2 ** 31

        hour = self.np_random.integers(low=0.0, high=24.0)
        current_date = datetime.today()
        year = current_date.year
        month = current_date.month
        day = current_date.day
        start_time = datetime(year, month, day, hour, 0, 0)
        self.patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(self.patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(2,))


# State space include the past n time steps CGM value
class T1DSimHistoryEnv(gym.Env):
    '''
        A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
        '''
    metadata = {'render_modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None, number_of_last_obs=15, max_episode_steps=480):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            # patient_name = 'adolescent#001'
            patient_name = random.choice(patient_names)
            print(patient_name)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.env, _, _, _ = self._create_env_from_random_state()
        self.number_of_last_obs = number_of_last_obs
        self.last_n_observations = np.ones([self.number_of_last_obs, ])
        # self.seed = seed
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

    def step(self, action: object) -> object:
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        observation, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
        observation = self.update_obs_history(observation[0])
        self.current_step += 1
        done = done or self.current_step >= self.max_episode_steps
        # if done and self.current_step < self.max_episode_steps:
        #      reward = -5e8
        return observation, reward, done, info

    def update_obs_history(self, observation):
        self.last_n_observations = np.roll(self.last_n_observations, 1)
        self.last_n_observations[0] = observation
        return self.last_n_observations

    def reset(self):
        self.current_step = 0
        self.env, _, _, _ = self._create_env_from_random_state()
        obs, _, _, _ = self.env.reset()
        self.last_n_observations[:] = obs
        return self.last_n_observations

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def delta_time(self):
        return random.uniform(-0.5, 0.5)

    def delta_cho(self):
        return random.uniform(-0.2,0.2)
    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = hash(self.np_random.integers(0, 1000)) % 2 ** 31
        seed3 = hash(seed2 + 1) % 2 ** 31
        seed4 = hash(seed3 + 1) % 2 ** 31

        hour = self.np_random.integers(low=0.0, high=24.0)
        current_date = datetime.today()
        year = current_date.year
        month = current_date.month
        day = current_date.day
        start_time = datetime(year, month, day, hour, 0, 0)
        self.patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        # Configure custom scenarios to standardise patient diets, this one is for adults
        # adult  # 001 102 [94, 113, 94, 75]
        # adult  # 002 111 [101, 121, 101, 81]
        # adult  # 003 82 [89, 107, 89, 71]
        # adult  # 004 63 [73, 87, 73, 58]
        # adult  # 005 94 [88, 106, 88, 70]
        # adult  # 006 66 [75, 90, 75, 60]
        # adult  # 007 91 [86, 103, 86, 69]
        # adult  # 008 103 [95, 114, 95, 76]
        # adult  # 009 75 [83, 99, 83, 66]
        # adult  # 010 74 [82, 99, 82, 66]
        # scenario = CustomScenario(start_time=datetime(year, month, day,hour=0, minute=0, second=0),
        #                           scenario=([7 + self.delta_time(), 75 * (1 + self.delta_cho())],
        #                                     [12 + self.delta_time(),90 * (1 + self.delta_cho())],
        #                                     [18 + self.delta_time(),75 * (1 + self.delta_cho())],
        #                                     [22 + self.delta_time(),60 * (1 + self.delta_cho())]))
        #Custom scenarios for children
        # child  # 001 35 [42, 21, 52, 21, 42, 31]
        # child  # 002 29 [37, 18, 46, 18, 37, 28]
        # child  # 003 41 [47, 23, 59, 23, 47, 35]
        # child  # 004 36 [42, 21, 53, 21, 42, 32]
        # child  # 005 38 [44, 22, 55, 22, 44, 33]
        # child  # 006 41 [47, 23, 59, 23, 47, 35]
        # child  # 007 46 [50, 25, 63, 25, 50, 38]
        # child  # 008 24 [33, 16, 41, 16, 33, 25]
        # child  # 009 36 [42, 21, 53, 21, 42, 32]
        # child  # 010 35 [42, 21, 53, 21, 42, 32]
        # adolescent  # 001 69 [62, 31, 78, 31, 62, 47]
        # adolescent  # 002 51 [55, 27, 69, 27, 55, 41]
        # adolescent  # 003 45 [50, 25, 62, 25, 50, 37]
        # adolescent  # 004 50 [54, 27, 67, 27, 54, 40]
        # adolescent  # 005 47 [52, 26, 65, 26, 52, 39]
        # adolescent  # 006 45 [50, 25, 63, 25, 50, 38]
        # adolescent  # 007 38 [44, 22, 55, 22, 44, 33]
        # adolescent  # 008 41 [47, 23, 59, 23, 47, 35]
        # adolescent  # 009 44 [49, 25, 61, 25, 49, 37]
        # adolescent  # 010 47 [52, 26, 65, 26, 52, 39]
        scenario = CustomScenario(start_time=datetime(year, month, day, hour=0, minute=0, second=0),
                                 scenario=([6 + self.delta_time(), 60 + self.delta_cho()], [9 + self.delta_time(), 30 + self.delta_cho()],
                                 [12 + self.delta_time(), 80 + self.delta_cho()], [16 + self.delta_time(), 30 + self.delta_cho()],
                                 [19 + self.delta_time(), 60 + self.delta_cho()], [22 +self.delta_time(),50 + self.delta_cho()]))

        #Use the default random scenario
        #scenario = RandomScenario(start_time=start_time, seed=seed3)

        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(self.patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render()

    @property
    def action_space(self):
        # pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1000, shape=(self.number_of_last_obs,))

# Discrete state space
class T1DDiscreteSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            # patient_name = 'adolescent#001'
            patient_name = random.choice(patient_names)
            print(patient_name)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.env, _, _, _ = self._create_env_from_random_state()

    def step(self, action):
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        observation, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)

        return [np.int(observation[0])], reward, done, info

    def reset(self):
        self.env, _, _, _ = self._create_env_from_random_state()
        obs, _, _, _ = self.env.reset()
        return [np.int(obs[0])]

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = hash(self.np_random.integers(0, 1000)) % 2 ** 31
        seed3 = hash(seed2 + 1) % 2 ** 31
        seed4 = hash(seed3 + 1) % 2 ** 31

        hour = self.np_random.integers(low=0.0, high=24.0)
        current_date = datetime.today()
        year = current_date.year
        month = current_date.month
        day = current_date.day
        start_time = datetime(year, month, day, hour, 0, 0)
        self.patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(self.patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int)

# Only adult patient
class T1DAdultSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            # patient_name = 'adolescent#001'
            adult_patients = [p for p in patient_names if "adult" in p]
            patient_name = random.choice(adult_patients)
            print(patient_name)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.env, _, _, _ = self._create_env_from_random_state()

    def step(self, action):
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        observation, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
        # if done:
        #     reward = -10  # -10000
        return observation, reward, done, info

    def reset(self):
        self.env, _, _, _ = self._create_env_from_random_state()
        obs, _, _, _ = self.env.reset()
        return obs

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = hash(self.np_random.integers(0, 1000)) % 2 ** 31
        seed3 = hash(seed2 + 1) % 2 ** 31
        seed4 = hash(seed3 + 1) % 2 ** 31

        hour = self.np_random.integers(low=0.0, high=24.0)
        current_date = datetime.today()
        year = current_date.year
        month = current_date.month
        day = current_date.day
        start_time = datetime(year, month, day, hour, 0, 0)
        self.patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(self.patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(1,))

# Adult patient with repeated steps
class T1DAdultSimV2Env(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None, repeat_steps=4):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            # patient_name = 'adolescent#001'
            adult_patients = [p for p in patient_names if "adult" in p]
            patient_name = random.choice(adult_patients)
            print(patient_name)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.repeat_steps = repeat_steps
        self.np_random, _ = seeding.np_random(seed=seed)
        self.env, _, _, _ = self._create_env_from_random_state()

    def step(self, action):
        # This gym only controls basal insulin
        observation = 0
        reward = 0
        done = False
        info = None
        act = Action(basal=action, bolus=0)
        for i in range(self.repeat_steps):
            if self.reward_fun is None:
                return self.env.step(act)
            observation, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
            if done:
                reward = -10
        return observation, reward, done, info

    def reset(self):
        self.env, _, _, _ = self._create_env_from_random_state()
        obs, _, _, _ = self.env.reset()
        return obs

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = hash(self.np_random.integers(0, 1000)) % 2 ** 31
        seed3 = hash(seed2 + 1) % 2 ** 31
        seed4 = hash(seed3 + 1) % 2 ** 31

        hour = self.np_random.integers(low=0.0, high=24.0)
        current_date = datetime.today()
        year = current_date.year
        month = current_date.month
        day = current_date.day
        start_time = datetime(year, month, day, hour, 0, 0)
        self.patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(self.patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(1,))

#Discrete action space
class T1DDiscreteEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            # patient_name = 'adolescent#001'
            adult_patients = [p for p in patient_names if "adult" in p]
            patient_name = random.choice(adult_patients)
            print(patient_name)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.actions = [0, 0.03, 0.06, 0.3, 0.6, 1]
        self.env, _, _, _ = self._create_env_from_random_state()

    def step(self, action):
        # This gym only controls basal insulin
        action = self.actions[action]
        act = Action(basal=action, bolus=0)

        if self.reward_fun is None:
            return self.env.step(act)
        observation, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
        if done:
            reward = -2000
        return observation, reward, done, info

    def reset(self):
        self.env, _, _, _ = self._create_env_from_random_state()
        obs, _, _, _ = self.env.reset()
        return obs

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = hash(self.np_random.integers(0, 1000)) % 2 ** 31
        seed3 = hash(seed2 + 1) % 2 ** 31
        seed4 = hash(seed3 + 1) % 2 ** 31

        hour = self.np_random.integers(low=0.0, high=24.0)
        current_date = datetime.today()
        year = current_date.year
        month = current_date.month
        day = current_date.day
        start_time = datetime(year, month, day, hour, 0, 0)
        self.patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(self.patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):

        return spaces.Discrete(6)

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(1,))

# State space include the current cho intake
class T1DCHOObsSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None, max_episode_steps=480):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            # patient_name = 'adolescent#001'
            adult_patients = [p for p in patient_names if "adult" in p]
            patient_name = random.choice(adult_patients)
            print(patient_name)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.env, _, _, _ = self._create_env_from_random_state()
        self.current_step = 0
        self.max_episode_steps = max_episode_steps
        self.last_meal = self.patient.planned_meal

    def step(self, action):
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        observation, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
        # if done:
        #     reward = -10  # -10000
        observation = self.add_last_meal_val_to_obs(observation[0])
        self.last_meal = self.patient.planned_meal
        self.current_step += 1
        done = done or self.current_step >= self.max_episode_steps
        return observation, reward, done, info

    def reset(self):
        self.current_step = 0
        self.env, _, _, _ = self._create_env_from_random_state()
        obs, _, _, _ = self.env.reset()
        self.last_meal = self.patient.planned_meal
        return self.add_last_meal_val_to_obs(obs[0])

    def add_last_meal_val_to_obs(self, obs):
        return [obs, self.patient.planned_meal]

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = hash(self.np_random.integers(0, 1000)) % 2 ** 31
        seed3 = hash(seed2 + 1) % 2 ** 31
        seed4 = hash(seed3 + 1) % 2 ** 31

        hour = self.np_random.integers(low=0.0, high=24.0)
        current_date = datetime.today()
        year = current_date.year
        month = current_date.month
        day = current_date.day
        start_time = datetime(year, month, day, hour, 0, 0)
        self.patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(self.patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(2,))

# State spce include the CGM values, insulin and cho intake in the past 20 time steps
class T1DSimMergeStateEnv(gym.Env):
    '''
        A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
        '''
    metadata = {'render_modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None, max_episode_steps=480):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            # patient_name = 'adolescent#001'
            patient_name = random.choice(patient_names)
            print(patient_name)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.max_episode_steps = max_episode_steps
        self.env, _, _, _ = self._create_env_from_random_state()
        self.reset()

    def step(self, action: object) -> object:
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            observation_, reward, done, info = self.env.step(act)
        else:
            observation_, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
        cho = self.patient.planned_meal
        observation = self.apstate.merge(observation_[0], action, cho)
        self.current_step += 1
        done = done or self.current_step >= self.max_episode_steps
        return observation, reward, done, info

    def reset(self):
        self.current_step = 0
        self.env, _, _, _ = self._create_env_from_random_state()
        obs, _, _, _ = self.env.reset()
        self.apstate = APState(obs[0])
        self.current_state = self.apstate.currentstate
        return self.current_state

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = hash(self.np_random.integers(0, 1000)) % 2 ** 31
        seed3 = hash(seed2 + 1) % 2 ** 31
        seed4 = hash(seed3 + 1) % 2 ** 31

        hour = self.np_random.integers(low=0.0, high=24.0)
        current_date = datetime.today()
        year = current_date.year
        month = current_date.month
        day = current_date.day
        start_time = datetime(year, month, day, hour, 0, 0)
        self.patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(self.patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render()

    @property
    def action_space(self):
        # pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1, shape=(60,))
