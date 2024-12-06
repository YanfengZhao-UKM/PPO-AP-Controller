import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


def compute_loss(policy: nn.Module, optimizer: torch.optim, discount_factor: float):

    trajectory_length = len(policy.rewards)
    returns = np.empty(trajectory_length, dtype = np.float32)
    future_ret = 0.

    # compute the returns efficiently.
    for t in reversed(range(trajectory_length)):
        future_ret = policy.rewards[t] + discount_factor * future_ret
        returns[t] = future_ret
    pass

    loss = -torch.sum(torch.stack(policy.log_probabilities) * torch.tensor(returns))

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    return loss



def compute_sugar_loss(policy: nn.Module, optimizer2: torch.optim, discount_factor: float):

    trajectory_length = len(policy.rewards)
    returns = np.empty(trajectory_length, dtype = np.float32)
    future_ret = 0.

    # compute the returns efficiently.
    for t in reversed(range(trajectory_length)):
        future_ret = policy.rewards[t] + discount_factor * future_ret
        returns[t] = future_ret
    pass
 
    loss2 = -torch.sum(torch.stack(policy.sugar_probabilities) * torch.tensor(returns)) 

    optimizer2.zero_grad()
    loss2.backward(retain_graph=True)
    optimizer2.step()

    return loss2


def CVGA_score(stat): 
    if (400 <= stat[0]):       
        score=65
        
    elif (375 <= stat[1] < 400):
        if (0 <= stat[0] < 50):
            score = 65        
        else:
            score = 60
        
    elif (350 <= stat[1] < 375):
        if (0 <= stat[0] < 50):
            score = 65
        elif (50 <= stat[0] < 55):
            score = 60
        else:
            score = 55
            
    elif (325 <= stat[1] < 350):
        if (0 <= stat[0] < 50):
            score = 65
        elif (50 <= stat[0] < 55):
            score = 60
        elif (55 <= stat[0] < 60):
            score = 55        
        else:
            score = 50
            
    elif (300 <= stat[1] < 325):
        if (0 <= stat[0] < 50):
            score = 65
        elif (50 <= stat[0] < 55):
            score = 60
        elif (55 <= stat[0] < 60):
            score = 55        
        elif (60 <= stat[0] < 65):
            score = 50
        else:
            score = 45
        
        
    elif (270 <= stat[1] < 300):
        if (0 <= stat[0] < 50):
            score = 65
        elif (50 <= stat[0] < 55):
            score = 60
        elif (55 <= stat[0] < 60):
            score = 55        
        elif (60 <= stat[0] < 65):
            score = 50
        elif (65 <= stat[0] < 70):
            score = 45
        else:
            score = 40       
        
    elif (240 <= stat[1] < 270):
        if (0 <= stat[0] < 50):
            score = 65
        elif (50 <= stat[0] < 55):
            score = 60
        elif (55 <= stat[0] < 60):
            score = 55        
        elif (60 <= stat[0] < 65):
            score = 50
        elif (65 <= stat[0] < 70):
            score = 45
        elif (70 <= stat[0] < 75):
            score = 40
        else:
            score = 35      
            
        
    elif (240 <= stat[1] < 240):
        if (0 <= stat[0] < 50):
            score = 65
        elif (50 <= stat[0] < 55):
            score = 60
        elif (55 <= stat[0] < 60):
            score = 55        
        elif (60 <= stat[0] < 65):
            score = 50
        elif (65 <= stat[0] < 70):
            score = 45
        elif (70 <= stat[0] < 75):
            score = 40
        elif (75 <= stat[0] < 80):    
            score = 35      
        else:
            score=30
            
    elif (180 <= stat[1] < 240):
        if (0 <= stat[0] < 50):
            score = 65
        elif (50 <= stat[0] < 55):
            score = 60
        elif (55 <= stat[0] < 60):
            score = 55        
        elif (60 <= stat[0] < 65):
            score = 50
        elif (65 <= stat[0] < 70):
            score = 45
        elif (70 <= stat[0] < 75):
            score = 40
        elif (75 <= stat[0] < 80):    
            score = 35      
        elif (80 <= stat[0] < 85):
            score = 30
        else: 
            score = 25      

    elif (162.5 <= stat[1] < 180):
        if (0 <= stat[0] < 50):
            score = 65
        elif (50 <= stat[0] < 55):
            score = 60
        elif (55 <= stat[0] < 60):
            score = 55        
        elif (60 <= stat[0] < 65):
            score = 50
        elif (65 <= stat[0] < 70):
            score = 45
        elif (70 <= stat[0] < 75):
            score = 40
        elif (75 <= stat[0] < 80):    
            score = 35      
        elif (80 <= stat[0] < 85):
            score = 30
        elif (85 <= stat[0] < 90):
            score = 25             
        else:
            score = 20
        
    elif (145 <= stat[1] < 162.5):
        if (0 <= stat[0] < 50):
            score = 65
        elif (50 <= stat[0] < 55):
            score = 60
        elif (55 <= stat[0] < 60):
            score = 55        
        elif (60 <= stat[0] < 65):
            score = 50
        elif (65 <= stat[0] < 70):
            score = 45
        elif (70 <= stat[0] < 75):
            score = 40
        elif (75 <= stat[0] < 80):    
            score = 35      
        elif (80 <= stat[0] < 85):
            score = 30
        elif (85 <= stat[0] < 90):
            score = 25             
        elif (90 <= stat[0] < 95):
            score = 20
        else:
            score=15
            
    elif (127.5 <= stat[1] < 145):
        if (0 <= stat[0] < 50):
            score = 65
        elif (50 <= stat[0] < 55):
            score = 60
        elif (55 <= stat[0] < 60):
            score = 55        
        elif (60 <= stat[0] < 65):
            score = 50
        elif (65 <= stat[0] < 70):
            score = 45
        elif (70 <= stat[0] < 75):
            score = 40
        elif (75 <= stat[0] < 80):    
            score = 35      
        elif (80 <= stat[0] < 85):
            score = 30
        elif (85 <= stat[0] < 90):
            score = 25             
        elif (90 <= stat[0] < 95):
            score = 20
        elif (95 <= stat[0] < 100):
            score=15
        else:
            score=10
    elif (110 <= stat[1] < 127.5):
        if (0 <= stat[0] < 50):
            score = 65
        elif (50 <= stat[0] < 55):
            score = 60
        elif (55 <= stat[0] < 60):
            score = 55        
        elif (60 <= stat[0] < 65):
            score = 50
        elif (65 <= stat[0] < 70):
            score = 45
        elif (70 <= stat[0] < 75):
            score = 40
        elif (75 <= stat[0] < 80):    
            score = 35      
        elif (80 <= stat[0] < 85):
            score = 30
        elif (85 <= stat[0] < 90):
            score = 25             
        elif (90 <= stat[0] < 95):
            score = 20
        elif (95 <= stat[0] < 100):
            score=15
        elif (100 <= stat[0] < 105):
            score=10
        else:
            score=5
    else:
        if (0 <= stat[0] < 50):
            score = 65
        elif (50 <= stat[0] < 55):
            score = 60
        elif (55 <= stat[0] < 60):
            score = 55        
        elif (60 <= stat[0] < 65):
            score = 50
        elif (65 <= stat[0] < 70):
            score = 45
        elif (70 <= stat[0] < 75):
            score = 40
        elif (75 <= stat[0] < 80):    
            score = 35      
        elif (80 <= stat[0] < 85):
            score = 30
        elif (85 <= stat[0] < 90):
            score = 25             
        elif (90 <= stat[0] < 95):
            score = 20
        elif (95 <= stat[0] < 100):
            score=15
        elif (100 <= stat[0] < 105):
            score=10
        elif (105 <= stat[0] < 110):
            score=5        
        else:
            score=65
    return score