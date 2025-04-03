# train/reinforcement.py

import torch
import torch.nn.functional as F

def compute_reward(generated_text):
    """
    Computes a simple reward for reinforcement learning.
    Reward: number of words minus penalty for repeated words.
    """
    words = generated_text.split()
    reward = len(words)
    if len(words) != len(set(words)):
        reward -= 2
    return reward

def reinforce_update(model, generated_text, optimizer):
    """
    Performs a simple REINFORCE update using the computed reward.
    """
    reward = compute_reward(generated_text)
    loss = -reward  # Negative loss to maximize reward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()