import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    LEARNING_RATE : float = 1e-4
    BATCH_SIZE : int = 64
    # Do not change without messing with the models
    IMAGE_SIZE : int = 64

    # 1: monochrome
    # 3: RGB
    CHANNELS_IMG : int = 3
    Z_DIM : int = 100

    NUM_EPOCHS : int = 5

    FEATURES_DISC : int = 64
    FEATURES_GEN : int = 64

    CRITIC_ITERATIONS : int = 5
    LAMBDA_GP : int = 10

    NOISE_DIM : int = 100

    # No momentum
    ADAM_BETAS : tuple = (0.0, 0.9)

def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    eps = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_img = real* eps + fake * (1 - eps)

    # calculate critic scores
    scores = critic(interpolated_img)
    gradient = torch.autograd.grad(
        inputs=interpolated_img,
        outputs=scores,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1)**2)

    return gradient_penalty
