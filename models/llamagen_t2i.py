# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
#
# Text-to-image (t2i) LlamaGen with MMDiT-style dual-stream joint self-attention.
#
# Unlike the original LlamaGen t2i (which simply prepends projected caption
# tokens and shares one ``wqkv`` for text + image, attending causally), this
# variant gives the TEXT stream and the IMAGE stream **separate** attention
# projections (``wqkv_text`` / ``wo_text`` vs ``wqkv`` / ``wo``) and a separate
# pre-attention norm (``attention_norm_text`` vs ``attention_norm``). Q/K/V from
# both streams are concatenated along the sequence, a single SDPA computes joint
# attention under a prefix-LM mask (text bidirectional among valid text tokens,
# image causal + attends to all valid text, text never attends to image), then
# the output is split back and each stream applies its own output projection.
# The FeedForward (``feed_forward`` / ``ffn_norm``) and the token / output heads
# are SHARED across streams.
#
# Weight-name compatibility: the IMAGE stream keeps the original c2i names
# (``wqkv``, ``wo``, ``attention_norm``, ``feed_forward``, ``ffn_norm``,
# ``tok_embeddings``, ``norm``, ``output``) so a class-conditional checkpoint
# loads straight into the image path with ``strict=False``; the text-stream
# params (``wqkv_text`` / ``wo_text`` / ``attention_norm_text``) and the
# ``CaptionEmbedder`` are reported as missing keys (newly initialised, with an
# optional warm copy from the image stream).
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02

    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 1024          # Qwen3.5-0.8B text hidden size
    class_dropout_prob: float = 0.1

    vocab_size: int = 16384
    cls_token_num: int = 300         # text prefix length (padded caption tokens)
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048

    use_checkpoint: bool = False

    # FLUX-style dual-stream -> single-stream split. The first
    # ``n_double_stream_layers`` transformer blocks are DOUBLE-stream (separate
    # text qkv/o + text attention_norm, joint attention); the remaining blocks
    # are SINGLE-stream (text + image share one set of projections / norm over
    # the concatenated sequence). -1 (default) => round(n_layer / 3) double
    # blocks, the rest single -- so most capacity goes to the image-centric
    # unified processing. 0 = all single-stream; n_layer = all double-stream.
    n_double_stream_layers: int = -1

    # Attention topology over the text prefix.
    #   * "causal" (default): the whole [text; image] sequence is a single
    #     lower-triangular causal LM -- text token i sees text <= i, image sees
    #     all valid text + causal image. Matches the original LlamaGen-t2i and
    #     is the natural choice given the text features come from a causal Qwen.
    #   * "prefix": prefix-LM -- text attends BIDIRECTIONALLY among valid text
    #     tokens (joint self-attention), image stays causal. Slightly richer
    #     text conditioning at the cost of a non-causal prefill mask.
    # Padding columns are always masked and a diagonal NaN-guard is always kept
    # in BOTH modes (see ``build_t2i_prefix_mask`` / ``_apply_prefix_mask``).
    text_attn: str = "causal"


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """Project frozen text-encoder features (caption_dim) into the model dim.

    Unconditional (CFG) handling is done OUTSIDE the model: the trainer feeds
    the empty-string Qwen embedding for the uncond branch, so there is no
    learnable ``uncond_embedding`` here. Caption dropout during training is
    likewise applied by replacing the whole text embedding with the
    empty-string embedding before this projection.
    """
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)

    def forward(self, caption):
        return self.cap_proj(caption)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  LlamaGen Model                               #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val.to(k_out.dtype)
        v_out[:, :, input_pos] = v_val.to(v_out.dtype)

        return k_out, v_out


class Attention(nn.Module):
    """Dual-stream (MMDiT-style) joint self-attention.

    ``text_len`` controls how the incoming sequence is split:
      * ``text_len >= seqlen`` -> the whole input is the text stream
        (inference prefill).
      * ``text_len <= 0``      -> the whole input is the image stream
        (inference kv-cache decode).
      * ``0 < text_len < seqlen`` -> training: leading ``text_len`` tokens are
        text, the rest are image; each stream uses its own projections, Q/K/V
        are concatenated along the sequence and a single SDPA computes joint
        attention.
    """
    def __init__(self, config: ModelArgs, dual_stream: bool = True):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim
        self.dual_stream = dual_stream

        # image / shared stream (names match c2i checkpoint -> loads directly).
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        # text stream (DOUBLE-stream blocks only). Single-stream blocks route
        # text through the shared wqkv/wo above, so they carry no extra params.
        if dual_stream:
            self.wqkv_text = nn.Linear(config.dim, total_kv_dim, bias=False)
            self.wo_text = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def _proj_qkv(self, x, proj):
        kv_size = self.n_kv_head * self.head_dim
        return proj(x).split([self.dim, kv_size, kv_size], dim=-1)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        text_len: int = 0,
    ):
        bsz, seqlen, _ = x.shape

        if not self.dual_stream:
            # SINGLE-stream: text + image share the same projections.
            xq, xk, xv = self._proj_qkv(x, self.wqkv)
        elif text_len >= seqlen:         # all-text (prefill)
            xq, xk, xv = self._proj_qkv(x, self.wqkv_text)
        elif text_len <= 0:              # all-image (decode)
            xq, xk, xv = self._proj_qkv(x, self.wqkv)
        else:                            # joint text + image (training)
            qt, kt, vt = self._proj_qkv(x[:, :text_len], self.wqkv_text)
            qi, ki, vi = self._proj_qkv(x[:, text_len:], self.wqkv)
            xq = torch.cat([qt, qi], dim=1)
            xk = torch.cat([kt, ki], dim=1)
            xv = torch.cat([vt, vi], dim=1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        # RoPE: text prefix positions carry zero-frequencies (precompute pads
        # ``cls_token_num`` zero rows), so applying rope to the concatenated
        # q/k leaves text unrotated and rotates image tokens by their 2D pos.
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda t: t.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values,
            attn_mask=mask,
            is_causal=True if mask is None else False,  # t2i always passes a mask
            dropout_p=self.attn_dropout_p if self.training else 0)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        # per-stream output projection
        if not self.dual_stream:
            output = self.wo(output)
        elif text_len >= seqlen:
            output = self.wo_text(output)
        elif text_len <= 0:
            output = self.wo(output)
        else:
            out_t = self.wo_text(output[:, :text_len])
            out_i = self.wo(output[:, text_len:])
            output = torch.cat([out_t, out_i], dim=1)

        output = self.resid_dropout(output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float, dual_stream: bool = True):
        super().__init__()
        self.dual_stream = dual_stream
        self.attention = Attention(config, dual_stream=dual_stream)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        # text gets its own pre-attention norm only in DOUBLE-stream blocks.
        if dual_stream:
            self.attention_norm_text = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, input_pos: int,
        mask: Optional[torch.Tensor] = None, text_len: int = 0):
        seqlen = x.shape[1]
        # per-stream pre-attention norm (single-stream uses the shared norm
        # for the whole concatenated sequence).
        if not self.dual_stream:
            normed = self.attention_norm(x)
        elif text_len >= seqlen:
            normed = self.attention_norm_text(x)
        elif text_len <= 0:
            normed = self.attention_norm(x)
        else:
            normed = torch.cat(
                [self.attention_norm_text(x[:, :text_len]),
                 self.attention_norm(x[:, text_len:])], dim=1)
        h = x + self.drop_path(self.attention(normed, freqs_cis, input_pos, mask, text_len))
        # shared FeedForward over the full sequence
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


def build_t2i_prefix_mask(
    text_mask: torch.Tensor, seq_len: int, text_len: int, text_attn: str = "causal",
) -> torch.Tensor:
    """Build the (B, 1, S, S) boolean t2i attention mask used in training.

    True = attend, False = masked. Layout for S = ``text_len`` + image_len.
    The image half is ALWAYS causal and ALWAYS sees every valid text column;
    only the text<->text block differs between the two modes:

      * ``text_attn="causal"`` (default): text token i attends to valid text
        tokens <= i. Equivalent to a plain lower-triangular causal LM over the
        whole [text; image] sequence, with padded text columns masked out.
      * ``text_attn="prefix"``: text attends BIDIRECTIONALLY among valid text
        tokens (joint self-attention).

    In both modes text rows never see image columns, and a diagonal (eye) is
    always kept so a fully-masked (uncond / empty) text row never produces an
    all -inf softmax row (NaN guard).
    """
    B = text_mask.shape[0]
    T = text_len
    device = text_mask.device
    tm = text_mask.bool()                                  # (B, T)

    mask = torch.zeros(B, seq_len, seq_len, dtype=torch.bool, device=device)
    if text_attn == "prefix":
        # any row -> valid text columns (text<->text bidirectional)
        mask[:, :, :T] = tm[:, None, :]
        # image rows -> causal among image columns
        L = seq_len - T
        if L > 0:
            causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
            mask[:, T:, T:] = causal[None]
        # text rows must NOT see image columns (already False for cols >= T)
        mask[:, :T, T:] = False
    else:
        # Pure causal over the concatenated sequence:
        #   * text row i sees text col j<=i (causal text);
        #   * image row sees all text cols (j < T <= row) + causal image cols;
        #   * text never sees image cols (col >= T > row).
        full_causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )
        mask = full_causal[None].expand(B, -1, -1).clone()
        # mask padded text columns for ALL rows
        mask[:, :, :T] = mask[:, :, :T] & tm[:, None, :]
    # NaN guard: keep the diagonal everywhere
    eye = torch.eye(seq_len, dtype=torch.bool, device=device)
    mask = mask | eye[None]
    return mask[:, None]                                   # (B, 1, S, S)


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.cls_token_num = config.cls_token_num
        # text caption embedder (replaces the c2i LabelEmbedder)
        self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # FLUX-style split: first ``n_double_stream`` blocks are double-stream
        # (separate text qkv/o + text norm); the rest are single-stream
        # (text + image share one projection set over the concatenated seq).
        if config.n_double_stream_layers >= 0:
            self.n_double_stream = min(config.n_double_stream_layers, config.n_layer)
        else:
            self.n_double_stream = max(1, round(config.n_layer / 3))

        # transformer blocks. Linear drop-path schedule 0 -> drop_path_rate,
        # computed in pure Python (avoids a torch tensor .item(), which lets the
        # model be instantiated on the ``meta`` device for free param counting).
        if config.n_layer <= 1:
            dpr = [0.0] * config.n_layer
        else:
            _step = config.drop_path_rate / (config.n_layer - 1)
            dpr = [_step * i for i in range(config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            dual = layer_id < self.n_double_stream
            self.layers.append(TransformerBlock(config, dpr[layer_id], dual_stream=dual))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 2d rotary pos embedding (text prefix gets cls_token_num zero rows)
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)

        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.use_checkpoint = self.config.use_checkpoint

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    @torch.no_grad()
    def init_text_stream_from_image(self):
        """Warm-start the text stream by copying the image-stream params.

        Called after loading a c2i checkpoint so the (newly created) text
        projections start identical to the image projections rather than from
        scratch -- usually converges faster than a cold random init. Only the
        DOUBLE-stream blocks carry text projections; single-stream blocks are
        skipped (they already share the loaded image/shared weights).
        """
        for block in self.layers:
            if not getattr(block, "dual_stream", False):
                continue
            block.attention.wqkv_text.weight.data.copy_(block.attention.wqkv.weight.data)
            block.attention.wo_text.weight.data.copy_(block.attention.wo.weight.data)
            block.attention_norm_text.weight.data.copy_(block.attention_norm.weight.data)

    def train(self, mode=True):
        self.max_batch_size = -1
        self.max_seq_length = -1
        for b in self.layers:
            b.attention.kv_cache = None
        return super().train(mode)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        # Default lower-triangular causal mask. The t2i sampler applies the
        # caption padding mask (+ NaN-guard diagonal) on top, and in "prefix"
        # mode also makes the text<->text block bidirectional. In "causal" mode
        # the text block is left lower-triangular (see _apply_prefix_mask).
        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)

    def embed_input(self, idx: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings, supporting hard indices and soft labels.

        * ``idx`` (B, L) long  -> ``tok_embeddings(idx)``.
        * ``idx`` (B, L, K) float -> ``idx @ tok_embeddings.weight`` (differentiable
          mixture over the codebook; used by REPA-E-style joint training).
        """
        if idx.dim() == 2:
            return self.tok_embeddings(idx)
        if idx.dim() == 3:
            return idx @ self.tok_embeddings.weight
        raise ValueError(
            f"`idx` must be (B, L) hard indices or (B, L, K) soft labels, "
            f"got shape {tuple(idx.shape)}"
        )

    def forward(
        self,
        idx: torch.Tensor,
        cond_emb: torch.Tensor,            # (B, cls_token_num, caption_dim) text features, or None during decode
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,   # (B, cls_token_num) caption padding mask (train)
        return_hidden_at: Optional[int] = None,
        early_exit: bool = False,
    ):
        """Run the dual-stream LlamaGen-t2i transformer.

        Modes (matching the c2i convention):
          * train / naive inference: ``idx`` and ``cond_emb`` both set. Sequence
            = [cls_token_num text tokens] + [image tokens]. ``text_len`` = cls_token_num.
          * prefill (kv cache): ``cond_emb`` set, ``idx`` None -> all text.
          * decode (kv cache): ``idx`` set, ``cond_emb`` None -> single image token.
        """
        if idx is not None and cond_emb is not None:  # training / naive inference
            cond_embeddings = self.cls_embedding(cond_emb)[:, :self.cls_token_num]
            token_embeddings = self.embed_input(idx)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis.to(h.device)
            text_len = self.cls_token_num
            seq_len = h.shape[1]
            if mask is None and text_mask is not None:
                mask = build_t2i_prefix_mask(
                    text_mask, seq_len, text_len, self.config.text_attn,
                ).to(h.device)
        else:
            if cond_emb is not None:  # prefill
                token_embeddings = self.cls_embedding(cond_emb)[:, :self.cls_token_num]
                text_len = token_embeddings.shape[1]
            else:                     # decode (single image token)
                token_embeddings = self.embed_input(idx)
                text_len = 0
            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)

        if input_pos is None:
            freqs_cis = self.freqs_cis[:h.shape[1]]
        else:
            freqs_cis = self.freqs_cis[input_pos]

        hidden_at_tap: Optional[torch.Tensor] = None
        for i, layer in enumerate(self.layers):
            if self.use_checkpoint:
                h = checkpoint(layer, h, freqs_cis, input_pos, mask, text_len, use_reentrant=True)
            else:
                h = layer(h, freqs_cis, input_pos, mask, text_len)
            if return_hidden_at is not None and (i + 1) == return_hidden_at:
                hidden_at_tap = h
                if early_exit:
                    return None, hidden_at_tap

        h = self.norm(h)
        logits = self.output(h).float()

        if self.training:
            logits = logits[:, self.cls_token_num - 1:].contiguous()

        if return_hidden_at is not None:
            return logits, hidden_at_tap
        return logits

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)


#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)  # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache])  # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1)  # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache])  # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)  # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)  # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


#################################################################################
#                                LlamaGen Configs                               #
#################################################################################
def LlamaGen_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs))  # 3.9B

def LlamaGen_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs))  # 1.4B

def LlamaGen_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1440, **kwargs))  # 775M

def LlamaGen_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs))  # 775M

def LlamaGen_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs))  # 343M

def LlamaGen_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs))  # 111M

def LlamaGen_S(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=384, **kwargs))  # 111M

LlamaGen_models = {
    'LlamaGen-S': LlamaGen_S, 'LlamaGen-B': LlamaGen_B, 'LlamaGen-L': LlamaGen_L,
    'LlamaGen-XL': LlamaGen_XL, 'LlamaGen-XXL': LlamaGen_XXL, 'LlamaGen-XXXL': LlamaGen_XXXL, 'LlamaGen-1B': LlamaGen_1B,
}


def _is_text_cond_param(n: str) -> bool:
    # CaptionEmbedder (projects the frozen text-encoder features).
    return n.startswith("cls_embedding")


def _is_text_stream_param(n: str) -> bool:
    # FLUX double-stream text branch (text-only attention projections + norm).
    return ("wqkv_text" in n) or ("wo_text" in n) or ("attention_norm_text" in n)


def _is_img_codebook_param(n: str) -> bool:
    # Image-token embedding + the codebook logits head.
    return n.startswith("tok_embeddings") or n == "output.weight"


def param_breakdown(model: "Transformer") -> dict:
    """Split a t2i Transformer's params into text- vs image-related buckets.

    * text_cond    : CaptionEmbedder (cap_proj over the text-encoder features).
    * text_stream  : double-stream text attention branch (wqkv_text/wo_text/
                     attention_norm_text). Single-stream blocks carry none.
    * img_codebook : image-token embedding (tok_embeddings) + logits head.
    * shared_trunk : everything else -- the joint transformer that ultimately
                     produces image tokens (shared/image attn, FFN, norms).

    "text-related" = text_cond + text_stream;
    "image-related" = img_codebook + shared_trunk.
    """
    text_cond = text_stream = img_codebook = shared = 0
    for n, p in model.named_parameters():
        k = p.numel()
        if _is_text_cond_param(n):
            text_cond += k
        elif _is_text_stream_param(n):
            text_stream += k
        elif _is_img_codebook_param(n):
            img_codebook += k
        else:
            shared += k
    total = text_cond + text_stream + img_codebook + shared
    n_double = sum(1 for b in model.layers if b.dual_stream)
    return {
        "total": total,
        "text_cond": text_cond,
        "text_stream": text_stream,
        "img_codebook": img_codebook,
        "shared_trunk": shared,
        "text_related": text_cond + text_stream,
        "image_related": img_codebook + shared,
        "n_layer": model.n_layer,
        "n_double": n_double,
        "n_single": model.n_layer - n_double,
        "dim": model.config.dim,
        "n_head": model.config.n_head,
    }


if __name__ == '__main__':
    # Build a t2i model, report the dual-stream / single-stream split + a
    # text-vs-image parameter breakdown, run a random forward (train + REPA
    # tap) and a decode step. With --sweep, instead iterate ALL model sizes
    # (on the meta device, no real allocation) and print a param table.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ar-model", type=str, default="LlamaGen-XL",
                        choices=list(LlamaGen_models.keys()))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--downsample-ratio", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--caption-dim", type=int, default=2048)   # Qwen3-1.7B hidden
    parser.add_argument("--cls-token-num", type=int, default=300)  # text prefix length
    parser.add_argument("--n-double-stream-layers", type=int, default=-1,
                        help="-1 => round(n_layer/3) double-stream blocks.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sweep", action="store_true",
                        help="Iterate all model sizes (meta device) and print a "
                             "text-vs-image parameter table; skip the forward test.")
    args = parser.parse_args()

    latent_size = args.image_size // args.downsample_ratio
    block_size = latent_size ** 2
    _common = dict(
        block_size=block_size,
        vocab_size=args.vocab_size,
        caption_dim=args.caption_dim,
        cls_token_num=args.cls_token_num,
        n_double_stream_layers=args.n_double_stream_layers,
        token_dropout_p=0.0, resid_dropout_p=0.0, ffn_dropout_p=0.0,
    )

    if args.sweep:
        # Meta device => params are shape-only (no memory), so even XXXL is
        # free to "build" just for counting.
        print("=" * 104)
        print(f"t2i param sweep | image_size={args.image_size} (latent {latent_size}, "
              f"{block_size} img tokens) | caption_dim={args.caption_dim} | "
              f"cls_token_num={args.cls_token_num} | double-stream=round(n_layer/3)")
        print("-" * 104)
        hdr = (f"{'model':<14}{'dim':>5}{'L':>4}{'dbl':>4}{'sgl':>4}"
               f"{'total':>11}{'text-rel':>11}{'(cond':>9}{'stream)':>9}"
               f"{'image-rel':>11}{'text%':>7}")
        print(hdr)
        for name, fn in LlamaGen_models.items():
            with torch.device("meta"):
                m = fn(**_common)
            b = param_breakdown(m)
            print(f"{name:<14}{b['dim']:>5}{b['n_layer']:>4}{b['n_double']:>4}{b['n_single']:>4}"
                  f"{b['total']/1e6:>10.1f}M{b['text_related']/1e6:>10.1f}M"
                  f"{b['text_cond']/1e6:>8.1f}M{b['text_stream']/1e6:>8.1f}M"
                  f"{b['image_related']/1e6:>10.1f}M{100*b['text_related']/b['total']:>6.1f}%")
            del m
        print("=" * 104)
        print("text-rel = CaptionEmbedder(cond) + double-stream text branch(stream); "
              "image-rel = tok_embeddings + logits head + shared trunk.")
        raise SystemExit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlamaGen_models[args.ar_model](**_common).to(device)
    b = param_breakdown(model)

    print("=" * 72)
    print(f"{args.ar_model} (t2i, FLUX-style double->single stream)")
    print(f"  dim={b['dim']}  n_layer={b['n_layer']}  n_head={b['n_head']}")
    print(f"  double-stream blocks={b['n_double']}  single-stream blocks={b['n_single']}")
    print(f"  cls_token_num(text prefix)={model.cls_token_num}  caption_dim={model.config.caption_dim}")
    print(f"  block_size(image tokens)={model.block_size}  vocab={model.vocab_size}")
    print("-" * 72)
    print(f"  total params           : {b['total']/1e6:9.2f}M")
    print(f"  text-related           : {b['text_related']/1e6:9.2f}M  ({100*b['text_related']/b['total']:.1f}%)")
    print(f"      - caption embedder : {b['text_cond']/1e6:9.2f}M")
    print(f"      - text stream      : {b['text_stream']/1e6:9.2f}M")
    print(f"  image-related          : {b['image_related']/1e6:9.2f}M  ({100*b['image_related']/b['total']:.1f}%)")
    print(f"      - tok emb + head   : {b['img_codebook']/1e6:9.2f}M")
    print(f"      - shared trunk     : {b['shared_trunk']/1e6:9.2f}M")
    print("=" * 72)

    # ---- random training forward ----
    B = args.batch_size
    T = model.cls_token_num
    L = model.block_size
    idx = torch.randint(0, args.vocab_size, (B, L - 1), device=device)
    cond_emb = torch.randn(B, T, args.caption_dim, device=device)
    text_mask = torch.ones(B, T, dtype=torch.long, device=device)
    text_mask[0, T // 2:] = 0  # sample 0 has half the prefix padded

    model.train()
    logits = model(idx=idx, cond_emb=cond_emb, text_mask=text_mask)
    print(f"[train] logits: {tuple(logits.shape)}  (expect (B={B}, L={L}, vocab={args.vocab_size}))")

    tap = max(1, model.n_layer // 2)
    logits2, hidden = model(idx=idx, cond_emb=cond_emb, text_mask=text_mask, return_hidden_at=tap)
    print(f"[train] REPA tap@{tap} hidden: {tuple(hidden.shape)}  (seq = T + L - 1 = {T + L - 1})")

    # ---- decode-style single-token step (single-stream blocks use shared proj) ----
    # ``setup_caches`` builds causal_mask / freqs_cis with the ambient default
    # device, so do it under ``torch.device(device)`` to keep them on-device.
    model.eval()
    with torch.no_grad(), torch.device(device):
        model.setup_caches(max_batch_size=B, max_seq_length=T + L, dtype=model.tok_embeddings.weight.dtype)
        model.causal_mask[:] = model.causal_mask | torch.eye(
            model.causal_mask.size(1), dtype=torch.bool, device=device)[None]
        pos = torch.arange(0, T, device=device)
        _ = model(idx=None, cond_emb=cond_emb, input_pos=pos)  # prefill text prefix
        one = torch.randint(0, args.vocab_size, (B, 1), device=device)
        step_pos = torch.tensor([T], device=device, dtype=torch.int)
        dec = model(idx=one, cond_emb=None, input_pos=step_pos)  # one image-token step
        print(f"[decode] one-step logits: {tuple(dec.shape)}")
    print("OK")
