import numpy as np
from simglucose.analysis.risk import risk_index

REF_BG_POINT = 112.517
LOW_BG_LIMIT = 50
HIGH_BG_LIMIT = 300
HIGH_BG_SAFE_LIMIT = 180
LOW_BG_SAFE_LIMIT = 70
MAX_HIGH_SAFE_INTERVAL = HIGH_BG_SAFE_LIMIT - REF_BG_POINT
MAX_LOW_SAFE_INTERVAL = REF_BG_POINT - LOW_BG_SAFE_LIMIT
MAX_HIGH_TO_REF_DIFF = HIGH_BG_LIMIT - REF_BG_POINT
MAX_LOW_TO_REF_DIFF = REF_BG_POINT - LOW_BG_LIMIT
MAX_LOW_SAFE_TO_LOW = LOW_BG_SAFE_LIMIT - 39

# 1-(risk index/180)**0.5,180 is the maximum value of the risk index when the glucose in range 10-1000
# **0.5 to widen the difference of the reward value near the optimal glucose
# range of a single reward:(0~1)
def myreward(BG_last_hour):
    if len(BG_last_hour) < 1:
        return 1e-8
    else:
        _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
        reward = (1 - risk_current / 180)**2  #Trick 7 : Use reward normalization
        if BG_last_hour[-1]<70:
            reward = reward * 0.5
        if reward<=0 or BG_last_hour[-1] < 50:
            reward = 1e-8
        return reward

# Just according to the glucose rang,severe hyper or hypo return -2, hyper or hypo return -1,and normal return 1
# range of a single reward:[-2,-1,1]
def simple_reward(BG_last_hour):
    if len(BG_last_hour) < 1:
        return 0
    else:
        bg_current=BG_last_hour[-1]
        if bg_current >= 70 and bg_current <= 180:
            reward = 1
        elif bg_current<50 or bg_current >300:
            reward = -2
        else:
            reward = -1
        return reward

# Just according to the glucose rang,severe hyper or hypo return -10, hyper or hypo return -2,normal return -1,optimal return 0
# range of a single reward:[-10,-2,-1,0]
def custom_reward(BG_last_hour):
    if BG_last_hour[-1] > 300:
        return -10
    elif BG_last_hour[-1] > 180:
        return -2
    elif BG_last_hour[-1] > 140:
        return -1
    elif BG_last_hour[-1] < 90:
        return -1
    elif BG_last_hour[-1] < 70:
        return -2
    elif BG_last_hour[-1] < 50:
        return -10
    else:
        return 0

# severe hypo or hyper get a additional -10 penalty
def shaped_reward_around_normal_bg(BG_last_hour):
    # current_bg = BG_last_hour[-1]
    #
    # if current_bg > REF_BG_POINT:
    #     diff = current_bg - REF_BG_POINT
    #     reward = 1 - ((diff / MAX_HIGH_TO_REF_DIFF) ** 0.06)
    # else:
    #     diff = REF_BG_POINT - current_bg
    #     reward = 1 - ((diff / MAX_LOW_TO_REF_DIFF) ** 0.2)
    #
    # if current_bg <= LOW_BG_LIMIT:
    #     reward += -10
    # elif current_bg >= HIGH_BG_LIMIT:
    #     reward += -10

    return -1


def neg_risk(BG_last_hour):
    if len(BG_last_hour) < 1:
        return 0
    else:
        _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
        reward = - risk_current
        return reward


def shaped_negative_reward_around_normal_bg(BG_last_hour):
    current_bg = BG_last_hour[-1]
    if REF_BG_POINT < current_bg <= HIGH_BG_SAFE_LIMIT:
        diff = current_bg - REF_BG_POINT
        reward = 100 * (1 - ((diff / MAX_HIGH_SAFE_INTERVAL) ** 0.2))
    elif REF_BG_POINT >= current_bg > LOW_BG_SAFE_LIMIT:
        diff = REF_BG_POINT - current_bg
        reward = 100 * (1 - ((diff / MAX_LOW_SAFE_INTERVAL) ** 0.2))
    elif HIGH_BG_LIMIT > current_bg > HIGH_BG_SAFE_LIMIT:
        diff = HIGH_BG_LIMIT - current_bg
        reward = -(1 - (diff / (HIGH_BG_LIMIT - HIGH_BG_SAFE_LIMIT)))
    elif LOW_BG_LIMIT < current_bg <= LOW_BG_SAFE_LIMIT:
        diff = current_bg - LOW_BG_LIMIT
        reward = -(1 - (diff / (LOW_BG_SAFE_LIMIT - LOW_BG_LIMIT)))
    else:
        reward = -1  # change to -10000

    return reward


def no_negativity(BG_last_hour):
    current_bg = BG_last_hour[-1]
    if REF_BG_POINT < current_bg <= HIGH_BG_SAFE_LIMIT:
        diff = current_bg - REF_BG_POINT
        reward = (1 - ((diff / MAX_HIGH_SAFE_INTERVAL) ** 0.2))
    elif REF_BG_POINT >= current_bg > LOW_BG_SAFE_LIMIT:
        diff = REF_BG_POINT - current_bg
        reward = (1 - ((diff / MAX_LOW_SAFE_INTERVAL) ** 0.2))
    else:
        reward = 3e-3
    return reward


def no_negativityV2(BG_last_hour):
    current_bg = BG_last_hour[-1]
    if REF_BG_POINT < current_bg:
        diff = current_bg - REF_BG_POINT
        reward = (1 - ((diff / MAX_HIGH_TO_REF_DIFF) ** 0.06))
    elif REF_BG_POINT >= current_bg:
        diff = REF_BG_POINT - current_bg
        reward = (1 - ((diff / MAX_LOW_TO_REF_DIFF) ** 0.1))
    else:
        reward = 3e-4

    return reward


def partial_negativity(BG_last_hour):
    current_bg = BG_last_hour[-1]
    if HIGH_BG_SAFE_LIMIT <= current_bg:
        diff = HIGH_BG_LIMIT - current_bg
        reward = -5 * (1 - (diff / (HIGH_BG_LIMIT - HIGH_BG_SAFE_LIMIT)))
    elif REF_BG_POINT < current_bg:
        diff = current_bg - REF_BG_POINT
        reward = (1 - ((diff / MAX_HIGH_SAFE_INTERVAL) ** 0.3))
    elif REF_BG_POINT >= current_bg:
        diff = REF_BG_POINT - current_bg
        reward = (1 - ((diff / MAX_LOW_TO_REF_DIFF) ** 0.1))
    else:
        reward = -3e-4

    return reward


def partial_negativityV2(BG_last_hour):
    current_bg = BG_last_hour[-1]
    if HIGH_BG_SAFE_LIMIT <= current_bg:
        diff = HIGH_BG_LIMIT - current_bg
        reward = -10 * (1 - (diff / (HIGH_BG_LIMIT - HIGH_BG_SAFE_LIMIT)))
    elif REF_BG_POINT < current_bg:
        diff = current_bg - REF_BG_POINT
        reward = (1 - ((diff / MAX_HIGH_SAFE_INTERVAL) ** 0.3))
    elif REF_BG_POINT >= current_bg > LOW_BG_SAFE_LIMIT:
        diff = REF_BG_POINT - current_bg
        reward = (1 - ((diff / (REF_BG_POINT - LOW_BG_SAFE_LIMIT)) ** 0.1))
    elif LOW_BG_SAFE_LIMIT >= current_bg:
        diff = current_bg - LOW_BG_LIMIT
        reward = -(1.0 - (diff / MAX_LOW_SAFE_TO_LOW))
    else:
        reward = -3e-4

    return reward


def risk_diff(BG_last_hour):
    if len(BG_last_hour) < 2:
        return 0
    else:
        _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
        _, _, risk_prev = risk_index([BG_last_hour[-2]], 1)
        reward = risk_prev - risk_current
        if reward < 0:
            reward = 0
        return reward


def smooth_reward(BG_last_hour):
    slope = .5
    reward = 1 - np.tanh(np.abs((BG_last_hour[-1] - REF_BG_POINT) / slope) * .1) ** 2
    if (BG_last_hour[-1] < LOW_BG_LIMIT) or (BG_last_hour[-1] > HIGH_BG_SAFE_LIMIT):
        reward = -1.
    return reward
