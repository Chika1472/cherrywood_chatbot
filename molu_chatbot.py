# -*- coding: utf-8 -*-
"""Utilities for training and running the Molu chatbot model."""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import random
import re
import time
import zipfile
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional dependency for seeding
    np = None
import sentencepiece as spm
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader, Dataset, random_split
except (ImportError, OSError, ValueError) as exc:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    LambdaLR = None  # type: ignore[assignment]
    DataLoader = Dataset = random_split = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: Optional[Exception] = exc
else:
    _TORCH_IMPORT_ERROR = None
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    zip_path: str
    pairing_mode: str = "turn_change"
    text_field: str = "form"
    min_len: int = 1
    max_len: int = 160
    shuffle: bool = True
    seed: int = 42

    def validate(self) -> None:
        if self.pairing_mode not in {"next_sentence", "turn_change"}:
            raise ValueError(
                "pairing_mode must be one of {'next_sentence', 'turn_change'}"
            )
        if self.text_field not in {"form", "original_form"}:
            raise ValueError("text_field must be either 'form' or 'original_form'")
        if not Path(self.zip_path).exists():
            raise FileNotFoundError(f"Zip file not found: {self.zip_path}")


@dataclass
class TokenizerConfig:
    model_prefix: str = "spm16k"
    vocab_size: int = 16000
    model_type: str = "unigram"
    character_coverage: float = 0.9998
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 3
    unk_id: int = 4
    control_symbols: Sequence[str] = ("<sep>",)
    byte_fallback: bool = True


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    max_len: int = 160
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 9
    batch_size: int = 128
    grad_accum_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_ratio: float = 0.05
    warmup_steps: Optional[int] = None
    max_grad_norm: float = 1.0
    train_ratio: float = 0.95
    num_workers: int = 2
    seed: int = 42
    log_interval: int = 50
    output_dir: str = "molu_chatbot"
    label_smoothing: float = 0.0

    def warmup_step_count(self, total_updates: int) -> int:
        if self.warmup_steps is not None:
            return self.warmup_steps
        return max(0, int(total_updates * self.warmup_ratio))


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    require_torch()
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(text: str) -> str:
    text = text.replace("\u200b", "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_korean(text: str) -> bool:
    kor = len(re.findall(r"[가-힣]", text))
    return kor >= max(2, int(0.2 * len(text)))


def require_torch() -> None:
    if torch is None:
        message = "PyTorch is required for this operation but could not be imported."
        if _TORCH_IMPORT_ERROR is not None:
            raise RuntimeError(message) from _TORCH_IMPORT_ERROR
        raise RuntimeError(message)


# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------


def load_dialogue_pairs(cfg: DataConfig) -> List[Tuple[str, str]]:
    cfg.validate()
    pairs: List[Tuple[str, str]] = []
    rng = random.Random(cfg.seed)
    zip_path = Path(cfg.zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".json"):
                continue
            with zf.open(name) as fh:
                data = json.load(io.TextIOWrapper(fh, encoding="utf-8"))
            docs = data.get("document", [])
            for doc in docs:
                utts = doc.get("utterance", [])
                if any(
                    utt.get("speaker_id") in {None, ""} or "speaker_id" not in utt
                    for utt in utts
                ):
                    logger.warning(
                        "Skipping document %s due to missing speaker_id metadata",
                        doc.get("id", name),
                    )
                    continue
                utts = sorted(
                    utts,
                    key=lambda u: (
                        u.get("start", float("inf")),
                        u.get("id", ""),
                    ),
                )

                sequence: List[Tuple[str, Optional[str]]] = []
                current_speaker: Optional[str] = None
                buffer: List[str] = []

                def flush_buffer() -> None:
                    nonlocal buffer, current_speaker
                    if not buffer:
                        return
                    combined = clean_text(" ".join(buffer))
                    buffer = []
                    if not combined:
                        return
                    if len(combined) < cfg.min_len or len(combined) > cfg.max_len:
                        return
                    if not is_korean(combined):
                        return
                    sequence.append((combined, current_speaker))

                for utt in utts:
                    raw = (
                        utt.get(cfg.text_field)
                        or utt.get("form")
                        or utt.get("original_form")
                        or ""
                    )
                    text = clean_text(raw)
                    if not text:
                        continue
                    if not is_korean(text):
                        continue
                    if len(text) > cfg.max_len:
                        continue

                    speaker = utt.get("speaker_id")
                    if current_speaker is None:
                        current_speaker = speaker
                        buffer.append(text)
                        continue

                    if speaker == current_speaker:
                        buffer.append(text)
                    else:
                        flush_buffer()
                        current_speaker = speaker
                        buffer.append(text)

                flush_buffer()

                if len(sequence) < 2:
                    continue

                speakers = {speaker for _, speaker in sequence if speaker is not None}
                if len(speakers) < 2:
                    continue

                if cfg.pairing_mode == "turn_change":
                    for (a, spk_a), (b, spk_b) in zip(sequence, sequence[1:]):
                        if spk_a != spk_b and a and b:
                            pairs.append((a, b))
                else:
                    for (a, _), (b, _) in zip(sequence, sequence[1:]):
                        if a and b:
                            pairs.append((a, b))

    if cfg.shuffle:
        rng.shuffle(pairs)

    if not pairs:
        raise RuntimeError(
            "No dialogue pairs were created. Please check the dataset and filters."
        )

    return pairs


class SentencePieceTokenizer:
    """Wrapper around SentencePiece to manage training and loading."""

    def __init__(self, cfg: TokenizerConfig, work_dir: Path):
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.work_dir / f"{self.cfg.model_prefix}.model"
        self.vocab_path = self.work_dir / f"{self.cfg.model_prefix}.vocab"
        self.processor: Optional[spm.SentencePieceProcessor] = None

    # ------------------------------------------------------------------
    # Loading & training
    # ------------------------------------------------------------------
    def train_or_load(self, pairs: Sequence[Tuple[str, str]]) -> "SentencePieceTokenizer":
        if not self.model_path.exists():
            corpus_path = self.work_dir / "corpus.txt"
            with open(corpus_path, "w", encoding="utf-8") as fh:
                for question, answer in pairs:
                    q = question.replace("\n", " ")
                    a = answer.replace("\n", " ")
                    fh.write(f"{q} <sep> {a}\n")
            spm.SentencePieceTrainer.train(
                input=str(corpus_path),
                model_prefix=str(self.work_dir / self.cfg.model_prefix),
                vocab_size=self.cfg.vocab_size,
                model_type=self.cfg.model_type,
                character_coverage=self.cfg.character_coverage,
                pad_id=self.cfg.pad_id,
                bos_id=self.cfg.bos_id,
                eos_id=self.cfg.eos_id,
                unk_id=self.cfg.unk_id,
                control_symbols=list(self.cfg.control_symbols),
                byte_fallback=self.cfg.byte_fallback,
            )
        return self.load()

    def load(self) -> "SentencePieceTokenizer":
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"SentencePiece model not found at {self.model_path}. Train it first."
            )
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))
        if self.processor.piece_to_id("<sep>") < 0:
            raise ValueError("SentencePiece model is missing the <sep> token. Retrain the tokenizer.")
        return self

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    def _ensure_processor(self) -> spm.SentencePieceProcessor:
        if self.processor is None:
            raise RuntimeError("SentencePieceProcessor is not loaded.")
        return self.processor

    @property
    def pad_id(self) -> int:
        return self._ensure_processor().pad_id()

    @property
    def bos_id(self) -> int:
        return self._ensure_processor().bos_id()

    @property
    def eos_id(self) -> int:
        return self._ensure_processor().eos_id()

    @property
    def unk_id(self) -> int:
        return self._ensure_processor().unk_id()

    @property
    def sep_id(self) -> int:
        processor = self._ensure_processor()
        return processor.piece_to_id("<sep>")

    @property
    def vocab_size(self) -> int:
        return self._ensure_processor().get_piece_size()

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        return self._ensure_processor().encode(text, out_type=int)

    def decode(self, token_ids: Sequence[int]) -> str:
        drop = {self.pad_id, self.bos_id, self.eos_id, self.unk_id, self.sep_id}
        filtered = [idx for idx in token_ids if idx not in drop]
        return self._ensure_processor().decode(filtered)


def create_labeled_sequences(
    pairs: Sequence[Tuple[str, str]], tokenizer: SentencePieceTokenizer
) -> List[Tuple[List[int], List[int]]]:
    labeled: List[Tuple[List[int], List[int]]] = []
    for ctx, rsp in pairs:
        ctx_ids = [tokenizer.bos_id] + tokenizer.encode(ctx) + [tokenizer.sep_id]
        rsp_ids = tokenizer.encode(rsp) + [tokenizer.eos_id]
        input_ids = ctx_ids + rsp_ids
        labels = [-100] * len(ctx_ids) + rsp_ids
        labeled.append((input_ids, labels))
    return labeled


def pack_token_sequences(
    sequences: Sequence[Tuple[List[int], List[int]]],
    max_length: int,
    pad_id: int,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]], int]:
    inputs: List[List[int]] = []
    labels: List[List[int]] = []
    attention_masks: List[List[int]] = []
    tokens_per_epoch = 0

    token_stream: List[int] = []
    label_stream: List[int] = []
    for tokens, lbls in sequences:
        token_stream.extend(tokens)
        label_stream.extend(lbls)

    for start in range(0, len(token_stream), max_length):
        end = min(start + max_length, len(token_stream))
        chunk_tokens = token_stream[start:end]
        chunk_labels = label_stream[start:end]
        attn = [1] * len(chunk_tokens)

        if len(chunk_tokens) < max_length:
            pad_len = max_length - len(chunk_tokens)
            chunk_tokens = chunk_tokens + [pad_id] * pad_len
            chunk_labels = chunk_labels + [-100] * pad_len
            attn = attn + [0] * pad_len

        tokens_per_epoch += sum(attn)
        inputs.append(chunk_tokens)
        labels.append(chunk_labels)
        attention_masks.append(attn)

    return inputs, labels, attention_masks, tokens_per_epoch


if torch is not None:

    class PackedConversationDataset(Dataset):
        """Dataset that packs token sequences into fixed-length windows."""

        def __init__(
            self,
            sequences: Sequence[Tuple[List[int], List[int]]],
            max_length: int,
            pad_id: int,
        ) -> None:
            inputs, labels, attention_masks, tokens_per_epoch = pack_token_sequences(
                sequences, max_length, pad_id
            )

            if not inputs:
                raise RuntimeError(
                    "Packed dataset is empty after processing sequences."
                )

            self.input_ids = torch.tensor(inputs, dtype=torch.long)
            self.labels = torch.tensor(labels, dtype=torch.long)
            self.attention_mask = torch.tensor(attention_masks, dtype=torch.long)
            self.tokens_per_epoch = tokens_per_epoch
            self.max_length = max_length

        def __len__(self) -> int:
            return self.input_ids.size(0)

        def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
            return {
                "input_ids": self.input_ids[index],
                "labels": self.labels[index],
                "attention_mask": self.attention_mask[index],
            }


else:

    class PackedConversationDataset:  # pragma: no cover - torch unavailable
        def __init__(self, *args, **kwargs) -> None:
            require_torch()


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


if torch is not None:

    class CausalSelfAttention(nn.Module):
        def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
            super().__init__()
            if d_model % n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads")
            self.nh = n_heads
            self.dk = d_model // n_heads
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.proj = nn.Linear(d_model, d_model, bias=False)
            self.attn_drop = nn.Dropout(dropout)
            self.resid_drop = nn.Dropout(dropout)

        def forward(
            self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            bsz, seq_len, hidden = x.shape
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.view(bsz, seq_len, self.nh, self.dk).transpose(1, 2)
            k = k.view(bsz, seq_len, self.nh, self.dk).transpose(1, 2)
            v = v.view(bsz, seq_len, self.nh, self.dk).transpose(1, 2)

            attn_bias: Optional[torch.Tensor] = None
            if attn_mask is not None:
                if attn_mask.dim() != 2 or attn_mask.shape != (bsz, seq_len):
                    raise ValueError(
                        "attn_mask must be of shape (batch_size, sequence_length)"
                    )
                pad_mask = (attn_mask == 0).unsqueeze(1).unsqueeze(2)
                neg_inf = torch.finfo(q.dtype).min
                attn_bias = pad_mask.to(dtype=q.dtype) * neg_inf

            dropout_p = self.attn_drop.p if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=dropout_p,
                is_causal=True,
            )
            y = y.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
            return self.resid_drop(self.proj(y))


    class TransformerBlock(nn.Module):
        def __init__(
            self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.1
        ) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = CausalSelfAttention(d_model, n_heads, dropout)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_ratio * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_ratio * d_model, d_model),
                nn.Dropout(dropout),
            )

        def forward(
            self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            x = x + self.attn(self.ln1(x), attn_mask)
            x = x + self.mlp(self.ln2(x))
            return x


    class GPTScratch(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            d_model: int = 512,
            n_layers: int = 8,
            n_heads: int = 8,
            max_len: int = 160,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.max_len = max_len
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_len, d_model)
            self.emb_norm = nn.LayerNorm(d_model)
            self.drop = nn.Dropout(dropout)
            self.blocks = nn.ModuleList(
                [TransformerBlock(d_model, n_heads, 4, dropout) for _ in range(n_layers)]
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.apply(self._init_weights)
            # Weight tying tends to improve perplexity and reduces parameters.
            self.head.weight = self.tok_emb.weight

        def _init_weights(self, module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(
            self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            bsz, seq_len = x.shape
            if seq_len > self.max_len:
                x = x[:, -self.max_len :]
                seq_len = x.shape[1]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, -self.max_len :]
            pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
            hidden = self.tok_emb(x) + self.pos_emb(pos)
            hidden = self.drop(self.emb_norm(hidden))
            for block in self.blocks:
                hidden = block(hidden, attention_mask)
            hidden = self.ln_f(hidden)
            return self.head(hidden)


else:

    class CausalSelfAttention:  # pragma: no cover - torch unavailable
        def __init__(self, *args, **kwargs) -> None:
            require_torch()


    class TransformerBlock:  # pragma: no cover - torch unavailable
        def __init__(self, *args, **kwargs) -> None:
            require_torch()


    class GPTScratch:  # pragma: no cover - torch unavailable
        def __init__(self, *args, **kwargs) -> None:
            require_torch()



# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def lm_loss(
    logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    require_torch()
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in the range [0, 1).")
    vocab = logits.size(-1)
    pred = logits[:, :-1].contiguous().view(-1, vocab)
    gold = targets[:, 1:].contiguous().view(-1)
    kwargs = {"ignore_index": -100}
    if label_smoothing > 0.0:
        kwargs["label_smoothing"] = label_smoothing
    return F.cross_entropy(pred, gold, **kwargs)


def count_active_tokens(labels: torch.Tensor) -> int:
    require_torch()
    return int((labels[:, 1:] != -100).sum().item())


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    autocast_ctx,
    label_smoothing: float = 0.0,
) -> Tuple[float, int]:
    require_torch()
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            attn = batch["attention_mask"].to(device)
            tokens = max(1, count_active_tokens(y))
            with autocast_ctx():
                logits = model(x, attention_mask=attn)
                loss_full = lm_loss(
                    logits, y, label_smoothing=label_smoothing
                )
            total_loss += loss_full.item() * tokens
            total_tokens += tokens
    return total_loss / max(1, total_tokens), total_tokens


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: TrainingConfig,
    total_updates: int,
) -> Optional[LambdaLR]:
    require_torch()
    if total_updates <= 0:
        return None

    warmup_steps = train_cfg.warmup_step_count(total_updates)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_updates - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_model(
    pairs: Sequence[Tuple[str, str]],
    tokenizer: SentencePieceTokenizer,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    data_cfg: DataConfig,
) -> Dict[str, List[Dict[str, float]]]:
    require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(train_cfg.seed)

    sequences = create_labeled_sequences(pairs, tokenizer)
    dataset = PackedConversationDataset(sequences, model_cfg.max_len, tokenizer.pad_id)

    total_samples = len(dataset)
    train_len = int(total_samples * train_cfg.train_ratio)
    train_len = min(max(1, train_len), total_samples - 1)
    val_len = total_samples - train_len
    if val_len <= 0:
        raise RuntimeError("Validation split would be empty; adjust train_ratio or dataset size.")

    generator = torch.Generator().manual_seed(train_cfg.seed)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = GPTScratch(
        vocab_size=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads,
        max_len=model_cfg.max_len,
        dropout=model_cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=train_cfg.betas,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    autocast_ctx = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext

    total_updates = math.ceil(len(train_loader) / train_cfg.grad_accum_steps) * train_cfg.epochs
    scheduler = build_scheduler(optimizer, train_cfg, total_updates)

    best_val = float("inf")
    history: List[Dict[str, float]] = []
    save_dir = Path(train_cfg.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_tokens = 0

        epoch_iter = tqdm(
            enumerate(train_loader, 1),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{train_cfg.epochs}",
            ncols=0,
        )

        optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in epoch_iter:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            attn = batch["attention_mask"].to(device)
            tokens = max(1, count_active_tokens(y))

            with autocast_ctx():
                logits = model(x, attention_mask=attn)
                loss_full = lm_loss(
                    logits, y, label_smoothing=train_cfg.label_smoothing
                )
                loss = loss_full / train_cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if step_idx % train_cfg.grad_accum_steps == 0 or step_idx == len(train_loader):
                scaler.unscale_(optimizer)
                if train_cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if scheduler is not None:
                    scheduler.step()

            running_loss += loss_full.item() * tokens
            running_tokens += tokens

            if step_idx % train_cfg.log_interval == 0:
                current_loss = running_loss / max(1, running_tokens)
                epoch_iter.set_postfix(loss=f"{current_loss:.4f}", lr=optimizer.param_groups[0]["lr"])

        train_loss = running_loss / max(1, running_tokens)
        val_loss, val_tokens = evaluate(
            model,
            val_loader,
            device,
            autocast_ctx,
            label_smoothing=train_cfg.label_smoothing,
        )
        epoch_time = time.time() - epoch_start

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_perplexity": math.exp(min(20.0, train_loss)),
            "val_loss": val_loss,
            "val_perplexity": math.exp(min(20.0, val_loss)),
            "train_tokens": running_tokens,
            "val_tokens": val_tokens,
            "epoch_time_sec": epoch_time,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(metrics)

        tqdm.write(
            f"[epoch {epoch}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={epoch_time/60:.1f} min"
        )

        if val_loss < best_val:
            best_val = val_loss
            config_payload = {
                "model": asdict(model_cfg),
                "tokenizer": asdict(tokenizer.cfg),
                "training": asdict(train_cfg),
                "data": asdict(data_cfg),
                "special_tokens": {
                    "PAD": tokenizer.pad_id,
                    "BOS": tokenizer.bos_id,
                    "SEP": tokenizer.sep_id,
                    "EOS": tokenizer.eos_id,
                    "UNK": tokenizer.unk_id,
                },
                "tokens_per_epoch": dataset.tokens_per_epoch,
                "best_val_loss": best_val,
            }
            torch.save(model.state_dict(), save_dir / "model.pt")
            with open(save_dir / "config.json", "w", encoding="utf-8") as fh:
                json.dump(config_payload, fh, indent=2, ensure_ascii=False)
            tqdm.write(f"Saved new best model to {save_dir}")

    with open(save_dir / "training_log.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, ensure_ascii=False)

    return {"history": history, "best_val_loss": best_val}


# ---------------------------------------------------------------------------
# Inference utilities
# ---------------------------------------------------------------------------


def sample_top_p(
    logits_row: torch.Tensor, top_p: float = 0.9, temperature: float = 0.7
) -> int:
    require_torch()
    logits_row = logits_row / max(1e-6, temperature)
    probs = torch.softmax(logits_row, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative > top_p
    if mask.shape[-1] > 0:
        mask[..., 0] = False
    filtered = sorted_probs.masked_fill(mask, 0.0)
    total = filtered.sum()
    if total <= 0:
        return int(torch.argmax(probs).item())
    filtered = filtered / total
    choice = torch.multinomial(filtered, 1)
    return int(sorted_idx[choice])


if torch is not None:

    @torch.no_grad()
    def generate_reply(
        model: GPTScratch,
        tokenizer: SentencePieceTokenizer,
        prompt: str,
        max_ctx: Optional[int] = None,
        max_new_tokens: int = 96,
        top_p: float = 0.9,
        temperature: float = 0.7,
        device: Optional[torch.device] = None,
    ) -> str:
        device = device or next(model.parameters()).device
        context_limit = (
            model.max_len if max_ctx is None else min(max_ctx, model.max_len)
        )
        context_limit = max(1, context_limit)
        history = [tokenizer.bos_id] + tokenizer.encode(prompt) + [tokenizer.sep_id]
        history = history[-context_limit:]

        for _ in range(max_new_tokens):
            x = torch.tensor(history, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x)
            next_token = sample_top_p(
                logits[0, -1], top_p=top_p, temperature=temperature
            )
            history.append(next_token)
            if len(history) > context_limit:
                history = history[-context_limit:]
            if next_token == tokenizer.eos_id:
                break

        try:
            last_sep = len(history) - 1 - history[::-1].index(tokenizer.sep_id)
        except ValueError:
            last_sep = 0
        response_tokens = history[last_sep:]
        return tokenizer.decode(response_tokens)

else:

    def generate_reply(*args, **kwargs):  # pragma: no cover - torch unavailable
        require_torch()


def load_checkpoint(model_dir: Path, device: torch.device) -> Tuple[GPTScratch, SentencePieceTokenizer, Dict]:
    require_torch()
    model_dir = Path(model_dir)
    with open(model_dir / "config.json", "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    tokenizer_cfg = TokenizerConfig(**cfg["tokenizer"])
    tokenizer = SentencePieceTokenizer(tokenizer_cfg, model_dir).load()

    model_cfg = ModelConfig(**cfg["model"])
    model = GPTScratch(
        vocab_size=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads,
        max_len=model_cfg.max_len,
        dropout=model_cfg.dropout,
    ).to(device)
    state_dict = torch.load(model_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer, cfg


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def run_train_from_args(args: argparse.Namespace) -> None:
    data_cfg = DataConfig(
        zip_path=args.zip_path,
        pairing_mode=args.pairing_mode,
        text_field=args.text_field,
        min_len=args.min_len,
        max_len=args.max_len,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )
    tokenizer_cfg = TokenizerConfig(
        model_prefix=args.sp_model_prefix,
        vocab_size=args.vocab_size,
    )
    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        log_interval=args.log_interval,
        output_dir=args.output_dir,
    )

    pairs = load_dialogue_pairs(data_cfg)
    tokenizer = SentencePieceTokenizer(tokenizer_cfg, Path(args.output_dir)).train_or_load(pairs)
    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_len=args.max_seq_len,
        dropout=args.dropout,
    )

    result = train_model(pairs, tokenizer, model_cfg, train_cfg, data_cfg)
    print("Training finished. Best val loss:", result["best_val_loss"])


def run_chat_from_args(args: argparse.Namespace) -> None:
    require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, cfg = load_checkpoint(Path(args.model_dir), device)
    reply = generate_reply(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_ctx=args.max_ctx,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        device=device,
    )
    print("User:", args.prompt)
    print("Molu:", reply)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Molu chatbot trainer and runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the chatbot model")
    train_parser.add_argument("--zip_path", required=True, help="Path to the NIKL dialogue zip file")
    train_parser.add_argument("--output_dir", default="molu_chatbot", help="Directory to store checkpoints")
    train_parser.add_argument(
        "--pairing_mode",
        choices=["next_sentence", "turn_change"],
        default="turn_change",
        help="Pair utterances across speaker changes by default; use 'next_sentence' to restore simple adjacency.",
    )
    train_parser.add_argument("--text_field", choices=["form", "original_form"], default="form")
    train_parser.add_argument("--min_len", type=int, default=1)
    train_parser.add_argument("--max_len", type=int, default=160)
    train_parser.add_argument("--no_shuffle", action="store_true", help="Disable pair shuffling")
    train_parser.add_argument("--seed", type=int, default=42)

    train_parser.add_argument("--sp_model_prefix", default="spm16k")
    train_parser.add_argument("--vocab_size", type=int, default=16000)

    train_parser.add_argument("--d_model", type=int, default=512)
    train_parser.add_argument("--n_layers", type=int, default=8)
    train_parser.add_argument("--n_heads", type=int, default=8)
    train_parser.add_argument("--max_seq_len", type=int, default=160)
    train_parser.add_argument("--dropout", type=float, default=0.1)

    train_parser.add_argument("--epochs", type=int, default=9)
    train_parser.add_argument("--batch_size", type=int, default=128)
    train_parser.add_argument("--grad_accum", type=int, default=4)
    train_parser.add_argument("--learning_rate", type=float, default=3e-4)
    train_parser.add_argument("--weight_decay", type=float, default=0.01)
    train_parser.add_argument("--beta1", type=float, default=0.9)
    train_parser.add_argument("--beta2", type=float, default=0.95)
    train_parser.add_argument("--warmup_ratio", type=float, default=0.05)
    train_parser.add_argument("--warmup_steps", type=int, default=None)
    train_parser.add_argument("--max_grad_norm", type=float, default=1.0)
    train_parser.add_argument("--train_ratio", type=float, default=0.95)
    train_parser.add_argument("--num_workers", type=int, default=2)
    train_parser.add_argument("--log_interval", type=int, default=50)

    train_parser.set_defaults(func=run_train_from_args)

    chat_parser = subparsers.add_parser("chat", help="Generate a reply using a trained model")
    chat_parser.add_argument("--model_dir", default="molu_chatbot", help="Directory containing model.pt and config.json")
    chat_parser.add_argument("--prompt", required=True, help="Prompt text to chat with the model")
    chat_parser.add_argument("--max_ctx", type=int, default=None)
    chat_parser.add_argument("--max_new_tokens", type=int, default=96)
    chat_parser.add_argument("--top_p", type=float, default=0.9)
    chat_parser.add_argument("--temperature", type=float, default=0.7)

    chat_parser.set_defaults(func=run_chat_from_args)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
