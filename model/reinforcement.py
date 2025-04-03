# model/reinforcement.py
import torch
import torch.nn.functional as F

def compute_reward(generated_text):
    """
    Computes a simple reward: rewards longer texts and penalizes repetition.
    Replace this with a more advanced reward function as needed.
    """
    words = generated_text.split()
    reward = len(words)
    if len(words) != len(set(words)):
        reward -= 2
    return reward

def reinforce_update(model, generated_text, optimizer):
    """
    Performs a simple REINFORCE update based on computed reward.
    """
    reward = compute_reward(generated_text)
    loss = -reward  # Negate because we maximize reward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
