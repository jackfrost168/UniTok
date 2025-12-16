import torch
import torch.nn as nn


# class MoEGate(nn.Module):
#     def __init__(self, latent_dim, num_experts, levels):
#         super().__init__()
#         self.num_experts = num_experts
#         self.gate = nn.Linear(latent_dim, self.num_experts * levels)  # Separate gating per level

#     def forward(self, z, level):
#         logits = self.gate(z)[:, level * self.num_experts : (level + 1) * self.num_experts]  # Select correct level
#         gate_probs = torch.softmax(logits, dim=-1)  # Compute expert probabilities
#         return gate_probs


class MoEGate(nn.Module):
    def __init__(self, latent_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(latent_dim, num_experts)  # One gate for all levels

    def forward(self, z):
        logits = self.gate(z)  # Compute logits for expert selection
        gate_probs = torch.softmax(logits, dim=-1)  # Convert to probability distribution
        return gate_probs  # Same gating for all levels
