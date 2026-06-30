"""Reconstruction + GAN + LPIPS + (VQ) quantizer loss for src Stage 1.

Adapted from REPA-E `loss/losses.py` (`ReconstructionLoss_Single_Stage`),
keeping the ``quantize_mode == "vq"`` branch and dropping the ``vae`` branch
since this codebase is VQ-based.

Loss components for the generator step (Stage 1, joint VQ + AR):

  total = w_rec    * L1(input, recon)
        + w_perc   * LPIPS(input, recon)
        + w_quant  * (vq_loss + commit_loss + w_entropy * entropy_loss)
        + w_disc   * disc_factor * (-mean(D(recon)))
        + proj_coef * REPA_align_loss      # if running the alignment-only call

Where ``vq_loss``, ``commit_loss``, ``entropy_loss`` come straight out of
the existing ``VectorQuantizer.forward`` in
``UniWorld-V1-Backup/models/vq_model.py``.
"""

from typing import Mapping, Text, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .perceptual_loss import PerceptualLoss
from .discriminator import NLayerDiscriminator, weights_init


def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def compute_lecam_loss(
    logits_real_mean: torch.Tensor,
    logits_fake_mean: torch.Tensor,
    ema_logits_real_mean: torch.Tensor,
    ema_logits_fake_mean: torch.Tensor,
) -> torch.Tensor:
    lecam_loss = torch.mean(torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2))
    lecam_loss += torch.mean(torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2))
    return lecam_loss


class ReconstructionLossVQ(nn.Module):
    """Loss head for Stage 1 of src.

    Different from the REPA-E variant:
      * Only the VQ branch is implemented.
      * ``forward(... mode="generator", quantizer_losses=(vq, commit, entropy))``
        receives the three VQ-internal losses as a tuple instead of a single
        scalar (so we can log them separately and weight ``entropy`` ourselves).
    """

    def __init__(self, config):
        super().__init__()
        loss_config = config.losses

        # Discriminator ---------------------------------------------------
        self.discriminator = NLayerDiscriminator(
            input_nc=3, n_layers=3, use_actnorm=False
        ).apply(weights_init)

        # Reconstruction --------------------------------------------------
        self.reconstruction_loss = loss_config.reconstruction_loss
        self.reconstruction_weight = loss_config.reconstruction_weight

        # Perceptual ------------------------------------------------------
        self.perceptual_loss = PerceptualLoss(loss_config.perceptual_loss).eval()
        self.perceptual_weight = loss_config.perceptual_weight

        # Quantizer (VQ) --------------------------------------------------
        self.quantizer_weight = loss_config.quantizer_weight
        # Multiplier on top of `entropy_loss` returned by VectorQuantizer; the
        # underlying VectorQuantizer already scales by `entropy_loss_ratio`.
        self.entropy_weight = float(loss_config.get("entropy_weight", 1.0))

        # GAN -------------------------------------------------------------
        self.discriminator_factor = loss_config.discriminator_factor
        self.discriminator_weight = loss_config.discriminator_weight
        self.discriminator_iter_start = loss_config.discriminator_start
        self.lecam_regularization_weight = loss_config.lecam_regularization_weight
        self.lecam_ema_decay = loss_config.get("lecam_ema_decay", 0.999)
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        # REPA align ------------------------------------------------------
        # Coefficient for the projection alignment loss when running the
        # alignment-only generator pass (Stage 1).
        self.proj_coef = float(loss_config.get("proj_coef", 0.0))

        self.config = config

    @autocast(enabled=False)
    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        quantizer_losses,
        global_step: int,
        mode: str = "generator",
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        inputs = inputs.float()
        reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, quantizer_losses, global_step)
        if mode == "discriminator":
            return self._forward_discriminator(inputs, reconstructions, global_step)
        raise ValueError(f"Unsupported mode {mode}")

    def should_discriminator_be_trained(self, global_step: int) -> bool:
        return global_step >= self.discriminator_iter_start

    def _forward_generator(self, inputs, reconstructions, quantizer_losses, global_step):
        """Generator step. ``quantizer_losses`` = (vq_loss, commit_loss, entropy_loss).

        Each entry is allowed to be ``None``/``0`` if the underlying VQ disabled it.
        """
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()

        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        else:
            raise ValueError(f"Unsupported reconstruction_loss {self.reconstruction_loss}")
        reconstruction_loss = reconstruction_loss * self.reconstruction_weight

        perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()

        device = inputs.device
        generator_loss = torch.zeros((), device=device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0.0
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)
        d_weight *= self.discriminator_weight

        # Quantizer pieces ------------------------------------------------
        vq_loss, commit_loss, entropy_loss, sample_entropy, avg_entropy = (
            self._unpack_q(quantizer_losses, device)
        )
        quantizer_loss = vq_loss + commit_loss + self.entropy_weight * entropy_loss

        total_loss = (
            reconstruction_loss
            + self.perceptual_weight * perceptual_loss
            + self.quantizer_weight * quantizer_loss
            + d_weight * discriminator_factor * generator_loss
        )

        loss_dict = dict(
            total_loss=total_loss.detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
            quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
            vq_loss=vq_loss.detach(),
            commit_loss=commit_loss.detach(),
            entropy_loss=entropy_loss.detach(),
            sample_entropy=sample_entropy.detach(),
            avg_entropy=avg_entropy.detach(),
            weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
            discriminator_factor=torch.tensor(discriminator_factor, device=device),
            d_weight=torch.tensor(d_weight, device=device),
            gan_loss=generator_loss.detach(),
        )
        return total_loss, loss_dict

    def _forward_discriminator(self, inputs, reconstructions, global_step):
        device = inputs.device
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0.0
        for param in self.discriminator.parameters():
            param.requires_grad = True

        real_images = inputs.detach().requires_grad_(True)
        logits_real = self.discriminator(real_images)
        logits_fake = self.discriminator(reconstructions.detach())

        discriminator_loss = discriminator_factor * hinge_d_loss(
            logits_real=logits_real, logits_fake=logits_fake
        )

        lecam_loss = torch.zeros((), device=device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = compute_lecam_loss(
                torch.mean(logits_real),
                torch.mean(logits_fake),
                self.ema_real_logits_mean,
                self.ema_fake_logits_mean,
            ) * self.lecam_regularization_weight
            self.ema_real_logits_mean = (
                self.ema_real_logits_mean * self.lecam_ema_decay
                + torch.mean(logits_real).detach() * (1 - self.lecam_ema_decay)
            )
            self.ema_fake_logits_mean = (
                self.ema_fake_logits_mean * self.lecam_ema_decay
                + torch.mean(logits_fake).detach() * (1 - self.lecam_ema_decay)
            )
        discriminator_loss = discriminator_loss + lecam_loss

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
            lecam_loss=lecam_loss.detach(),
        )
        return discriminator_loss, loss_dict

    @staticmethod
    def _unpack_q(quantizer_losses, device):
        """Coerce ``(vq_loss, commit_loss, entropy_loss[, codebook_usage])``.

        ``entropy_loss`` may itself be a 3-tuple ``(total, sample_entropy,
        avg_entropy)`` (current ``VectorQuantizer.forward`` packs the two
        components alongside the scaled total so they can be logged).
        Returns ``(vq, commit, entropy_total, sample_entropy, avg_entropy)``
        with all entries coerced to scalar tensors on ``device``.
        """
        if isinstance(quantizer_losses, dict):
            vq = quantizer_losses.get("vq_loss")
            commit = quantizer_losses.get("commit_loss")
            entropy = quantizer_losses.get("entropy_loss")
        else:
            # Accept both 3-tuple and 4-tuple (the latter includes codebook_usage).
            vq, commit, entropy = quantizer_losses[0], quantizer_losses[1], quantizer_losses[2]
        zero = torch.zeros((), device=device)

        def _coerce(x):
            if x is None:
                return zero
            if isinstance(x, torch.Tensor):
                return x.to(device=device, dtype=torch.float32)
            return torch.tensor(float(x), device=device)

        if isinstance(entropy, (tuple, list)):
            entropy_total = _coerce(entropy[0])
            sample_entropy = _coerce(entropy[1]) if len(entropy) > 1 else zero
            avg_entropy = _coerce(entropy[2]) if len(entropy) > 2 else zero
        else:
            entropy_total = _coerce(entropy)
            sample_entropy = zero
            avg_entropy = zero

        return _coerce(vq), _coerce(commit), entropy_total, sample_entropy, avg_entropy
