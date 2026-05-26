"""Variational autoencoder serving as the agent's generative world model.

The world model defines a parameterized generative process p(μ|z) over the
agent's own internal macrostate μ, with a variational posterior q(z|μ) over
latent causes z. The variational free energy F = -ELBO is the core metric
of thermodynamic viability: F quantifies surprise (negative log-evidence)
and its minimization drives active inference.

Equation reference:
  dS_int/dt = -k_B · ε(T) · H_env(t) + S_gen       (Eq 1)
  S_gen ≥ k_B ln(2) · d/dt H(μ)                      (Eq 3, Generalized Landauer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INPUT_DIM, HIDDEN_DIM, LATENT_DIM, BETA_KL, DEVICE


class Encoder(nn.Module):
    """Maps internal macrostate μ → variational posterior q(z|μ)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.LeakyReLU(0.2),
        )
        self.mu_out = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.logvar_out = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, x):
        h = self.net(x)
        return self.mu_out(h), self.logvar_out(h)


class Decoder(nn.Module):
    """Maps latent cause z → reconstructed macrostate μ̂."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, INPUT_DIM),
        )

    def forward(self, z):
        return self.net(z)


class WorldModel(nn.Module):
    """Generative model of the agent's internal embedding space.

    The world model implements the core computation underlying the Szilard
    negentropy engine: it extracts structured information (reconstruction)
    from the ambient flux (the macrostate μ) by inferring latent causes.
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.to(DEVICE)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = μ + σ · ε."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Full forward pass: encode, reparameterize, decode.

        Returns:
            x_hat: reconstructed macrostate
            z: latent sample
            mu_z: latent mean
            logvar_z: latent log-variance
        """
        mu_z, logvar_z = self.encoder(x)
        z = self.reparameterize(mu_z, logvar_z)
        x_hat = self.decoder(z)
        return x_hat, z, mu_z, logvar_z

    def compute_elbo(self, x, x_hat, mu_z, logvar_z):
        """Evidence Lower Bound: ELBO = -KL(q||p) + log p(x|z).

        Variational free energy F = -ELBO, so minimizing F = maximizing ELBO.
        """
        kl_div = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=-1)
        recon_loss = F.mse_loss(x_hat, x, reduction='none').sum(dim=-1)
        elbo = -BETA_KL * kl_div - recon_loss
        return elbo, kl_div, recon_loss

    def compute_free_energy(self, x):
        """Compute variational free energy F = -ELBO for a given macrostate."""
        x_hat, z, mu_z, logvar_z = self.forward(x)
        elbo, kl, recon = self.compute_elbo(x, x_hat, mu_z, logvar_z)
        return -elbo, kl, recon, x_hat, z

    def step_optimize(self, x, optimizer):
        """Single optimization step to minimize F for a single macrostate."""
        x = x.to(DEVICE)
        optimizer.zero_grad()
        F, kl, recon, x_hat, z = self.compute_free_energy(x)
        loss = F  # F is already -ELBO, minimizing F = maximizing ELBO
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        return F.item(), kl.mean().item(), recon.mean().item()
