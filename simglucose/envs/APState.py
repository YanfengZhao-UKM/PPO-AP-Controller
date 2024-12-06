import numpy as np
from datetime import datetime

state_lookback = 20
time_array = np.arange(state_lookback)
observation_max = 600
insulin_max = 30
cho_max = 100

def normalization(x,max):
    if max == 0:
        return np.zeros(x.size)
    return x/max


class APState:
    def __init__(self, observation):

        #1. INSTANTIATE OBSERVATION VARS
        #if observation>0:
        self.observation = observation
        # else:
        #     self.observation = 159
        self.observation_hist = np.zeros(state_lookback)
        for i in range(state_lookback):
            self.observation_hist[i] = self.observation
        self.observation_norm = normalization(self.observation_hist,observation_max)

        # #2. Instantiate the first and second derivatives of the observation_hist
        # self.observation_hist_dt = np.zeros(19)
        # self.observation_dt_norm = normalization(self.observation_hist_dt,max(self.observation_hist_dt))
        # self.observation_hist_ddt = np.zeros(18)
        # self.observation_ddt_norm = normalization(self.observation_hist_ddt,max(self.observation_hist_ddt))

        #3. INSTANTIATE ACTION HISTORY VARS
        self.insulin = 0.025
        self.insulin_hist = np.zeros(state_lookback)
        for i in range(state_lookback):
            self.insulin_hist[i] = self.insulin
        self.insulin_hist_norm = normalization(self.insulin_hist,insulin_max)

        #4. INSTANTIATE PLANNED MEAL HISTORY ARRAY
        self.cho = 0
        self.cho_hist = np.zeros(state_lookback)
        self.cho_hist_norm = normalization(self.cho_hist,cho_max)

        #5. POPULATE CURRENT STATE ARRAY AND RETURN IT
        self.currentstate = np.concatenate(
            (self.observation_norm, self.insulin_hist_norm, self.cho_hist_norm),
            axis=None).squeeze().astype(np.float32)


    def merge(self, observation, insulin, cho):

        ##1. POPULATE OBSERVATION HISTORY ARRAY
        self.observation = observation
        self.observation_hist = np.roll(self.observation_hist,1)  # cycle the array by 1 position and move the oldest value from the rightmost position to first position
        self.observation_hist[0] = observation  # replace first value in array (aka - the oldest value) with the most recent glucose change value
        self.observation_norm = normalization(self.observation_hist, observation_max)   # normalize the observation history into a new array for NN processing

        # #2. Caculate the first and second derivatives of the observation_hist
        # self.observation_hist_dt = np.diff(self.observation_hist) / np.diff(time_array)
        # self.observation_dt_norm = normalization(self.observation_hist_dt, max(self.observation_hist_dt))
        # self.observation_hist_ddt = np.diff(self.observation_hist_dt) / np.diff(time_array[:-1])
        # self.observation_ddt_norm = normalization(self.observation_hist_ddt, max(self.observation_hist_ddt))

        #3. POPULATE ACTION HISTORY ARRAY
        self.insulin = insulin
        self.insulin_hist = np.roll(self.insulin_hist,
                               1)  # cycle action array by 1 position move the oldest value from the rightmost position to the first position
        self.insulin_hist[0] = self.insulin  # replace first value in array (aka - the oldest value) with the most recent action change value
        self.insulin_hist_norm = normalization(self.insulin_hist, insulin_max)   # normalize the action history array into a new array for NN processing

        #4. POPULATE PLANNED MEAL HISTORY ARRAY
        self.cho = cho
        self.cho_hist = np.roll(self.cho_hist,
                                    1)  # cycle planed meal array by 1 position move the oldest value from the rightmost position to the first position
        self.cho_hist[
            0] = self.cho  # replace first value in array (aka - the oldest value) with the most recent action

        self.cho_hist_norm = normalization(self.cho_hist, cho_max) # normalize the planned meal history into a new array for NN processing

        #5. POPULATE CURRENT STATE ARRAY AND RETURN IT
        self.currentstate = np.concatenate(
            (self.observation_norm, self.insulin_hist_norm, self.cho_hist_norm),
            axis=None).squeeze().astype(np.float32)
        return self.currentstate






