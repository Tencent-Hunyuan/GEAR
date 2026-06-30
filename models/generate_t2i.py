"""Text-to-image autoregressive sampling for the dual-stream LlamaGen-t2i.

Mirrors the c2i sampler in :mod:`models.generate` (batch-doubling CFG +
KV-cache decode) but adapts the conditioning to a fixed-length text prefix:

* CFG = batch doubling ``[cond ; uncond]``; ``logits = uncond + (cond - uncond)
  * cfg_scale`` (with optional ``cfg_interval`` early-stop). The uncond branch is
  the empty-string Qwen embedding (caller supplies it), not a learnable token.
* The text prefix (length ``model.cls_token_num``) is processed once in the
  prefill, writing the text-stream K/V into cache positions ``0 .. T-1``. Image
  tokens are then decoded one at a time (image-stream K/V written at positions
  ``>= T``); each image query attends over the concatenated [text K/V ; image
  K/V] cache -- the cache is projection-agnostic so the dual-stream split is
  transparent to it.
* The text prefix block of ``model.causal_mask`` follows ``model.config.
  text_attn``: "causal" (default) keeps it lower-triangular (pure causal LM),
  "prefix" makes it bidirectional among valid text tokens (joint self-attn).
  Either way it is masked by the caption padding mask with a diagonal NaN-guard
  -- consistent with the training-time mask from ``build_t2i_prefix_mask``.
"""

import torch

from models.generate import sample


def prefill_t2i(model, cond_emb, input_pos, cfg_scale, **sampling_kwargs):
    logits = model(None, cond_emb, input_pos)
    if cfg_scale > 1.0:
        cond_logits, uncond_logits = torch.split(logits, len(logits) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token_t2i(model, x, input_pos, cfg_scale, cfg_flag, **sampling_kwargs):
    assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits = model(x_combined, None, input_pos)
        cond_logits, uncond_logits = torch.split(logits, len(logits) // 2, dim=0)
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits = model(x, None, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens_t2i(
    model, cur_token, input_pos, num_new_tokens, cfg_scale, cfg_interval,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []
    cfg_flag = True
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            next_token, next_prob = decode_one_token_t2i(
                model, cur_token, input_pos, cfg_scale, cfg_flag, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(-1, 1)
    return new_tokens, new_probs


def _apply_prefix_mask(model, emb_masks, T, device):
    """Make the text prefix caption-mask-aware (and bidirectional in prefix mode).

    ``model.causal_mask`` starts as a plain lower-triangular (B, S, S) bool
    matrix (set in ``setup_caches``). Depending on ``model.config.text_attn``:
      * "causal" (default): keep the text block lower-triangular as-is (text
        token i sees text <= i) -- nothing to do for the [:T, :T] block.
      * "prefix": set the [:T, :T] block True so text tokens attend to each
        other bidirectionally (joint self-attention).
    Then, for BOTH modes, we:
      1. AND every row's text columns with the caption padding mask so padded
         text positions are never attended;
      2. OR the identity so any fully-masked row keeps its diagonal (NaN guard).
    Image columns (>= T) keep the lower-triangular causal structure, and text
    rows never see image columns because col > row there.

    ``emb_masks`` is expected already batch-aligned with ``model.causal_mask``
    (i.e. doubled when CFG is on). This MUST match the training-time mask built
    by ``build_t2i_prefix_mask`` with the same ``text_attn``.
    """
    text_attn = getattr(model.config, "text_attn", "causal")
    if text_attn == "prefix":
        # text<->text bidirectional
        model.causal_mask[:, :T, :T] = True
    # else "causal": leave the [:T, :T] block lower-triangular (causal text)
    if emb_masks is not None:
        emb_masks = emb_masks.bool().to(device)
        # mask invalid (padded) text columns for ALL rows
        model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] & emb_masks[:, None, :]
    # NaN guard: keep the diagonal
    S = model.causal_mask.size(1)
    eye = torch.eye(S, dtype=torch.bool, device=device)
    model.causal_mask[:] = model.causal_mask | eye[None]


@torch.no_grad()
def generate_t2i(
    model,
    cond_emb,
    uncond_emb,
    max_new_tokens,
    cond_mask=None,
    uncond_mask=None,
    cfg_scale=4.0,
    cfg_interval=-1,
    **sampling_kwargs,
):
    """Sample image-token sequences from text conditioning.

    Parameters
    ----------
    cond_emb : (B, T, caption_dim)   conditional text features.
    uncond_emb : (B, T, caption_dim) unconditional (empty-string) text features.
    cond_mask / uncond_mask : (B, T) caption padding masks (1 = valid token).
    """
    T = cond_emb.shape[1]
    assert T == model.cls_token_num, (
        f"cond_emb prefix len {T} != model.cls_token_num {model.cls_token_num}"
    )
    max_batch_size = cond_emb.shape[0]
    device = cond_emb.device

    if cfg_scale > 1.0:
        cond_combined = torch.cat([cond_emb, uncond_emb], dim=0)
        if cond_mask is not None and uncond_mask is not None:
            emb_masks = torch.cat([cond_mask, uncond_mask], dim=0)
        else:
            emb_masks = None
    else:
        cond_combined = cond_emb
        emb_masks = cond_mask

    T_new = T + max_new_tokens
    max_seq_length = T_new

    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(
            max_batch_size=max_batch_size_cfg,
            max_seq_length=max_seq_length,
            dtype=model.tok_embeddings.weight.dtype,
        )

    # build the prefix-LM causal mask on the (already-allocated) buffer.
    # ``emb_masks`` is already CFG-doubled above to match the cache batch.
    _apply_prefix_mask(model, emb_masks, T, device)

    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

    input_pos = torch.arange(0, T, device=device)
    next_token = prefill_t2i(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    seq[:, T:T + 1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens_t2i(
        model, next_token, input_pos, max_new_tokens - 1, cfg_scale, cfg_interval, **sampling_kwargs,
    )
    seq[:, T + 1:] = torch.cat(generated_tokens, dim=1)

    return seq[:, T:]
