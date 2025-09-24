"""Lightweight data utilities used by the tests.

This module intentionally mirrors the behaviour expected by the unit tests in
``tests/test_data_processing.py`` so it can operate without the large Colab
notebook that originally produced the training pipeline.
"""
from __future__ import annotations

import io
import json
import logging
import random
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

LOGGER = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Normalise whitespace and strip zero-width characters."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


@dataclass
class DataConfig:
    """Configuration for loading dialogue data from a zipped JSON archive."""

    zip_path: str
    pairing_mode: str = "turn_change"
    text_field: str = "form"
    min_text_chars: int = 1
    max_text_chars: int = 4000
    shuffle: bool = True
    seed: int = 42
    fallback_fields: Sequence[str] = field(
        default_factory=lambda: ("text", "content", "body", "article", "paragraphs", "summary")
    )

    def resolve_path(self) -> Path:
        return Path(self.zip_path).expanduser().resolve()


def _iter_conversation_docs(payload: object) -> Iterator[dict]:
    if isinstance(payload, dict):
        docs = payload.get("document")
        if isinstance(docs, list):
            for doc in docs:
                if isinstance(doc, dict) and isinstance(doc.get("utterance"), list):
                    yield doc
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_conversation_docs(item)


def _collect_strings(value: object) -> List[str]:
    strings: List[str] = []
    if isinstance(value, str):
        cleaned = clean_text(value)
        if cleaned:
            strings.append(cleaned)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                cleaned = clean_text(item)
                if cleaned:
                    strings.append(cleaned)
    return strings


def _extract_top_level_texts(payload: object, fallback_fields: Sequence[str]) -> List[str]:
    texts: List[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in fallback_fields:
                texts.extend(_collect_strings(value))
            elif isinstance(value, str):
                texts.extend(_collect_strings(value))
            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                texts.extend(_collect_strings(value))
    elif isinstance(payload, list):
        for item in payload:
            texts.extend(_extract_top_level_texts(item, fallback_fields))
    elif isinstance(payload, str):
        texts.extend(_collect_strings(payload))
    return texts


def _select_text(utterance: dict, cfg: DataConfig) -> str:
    candidates = []
    preferred = utterance.get(cfg.text_field)
    if preferred:
        candidates.append(preferred)
    if cfg.text_field != "form":
        fallback = utterance.get("form")
        if fallback:
            candidates.append(fallback)
    if cfg.text_field != "original_form":
        fallback = utterance.get("original_form")
        if fallback:
            candidates.append(fallback)
    for text in candidates:
        cleaned = clean_text(text)
        if cleaned:
            return cleaned
    return ""


def load_dialogue_pairs(cfg: DataConfig) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Load dialogue pairs and monologue documents from a zipped dataset."""

    path = cfg.resolve_path()
    pairs: List[Tuple[str, str]] = []
    mono_docs: List[str] = []
    missing_speaker_docs = 0

    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".json"):
                continue
            with zf.open(name) as handle:
                data = json.load(io.TextIOWrapper(handle, encoding="utf-8"))

            docs = list(_iter_conversation_docs(data))
            if docs:
                for doc in docs:
                    utterances = doc.get("utterance") or []
                    cleaned_turns: List[Tuple[str, str | None]] = []
                    for utt in utterances:
                        if not isinstance(utt, dict):
                            continue
                        text = _select_text(utt, cfg)
                        if not text:
                            continue
                        if len(text) > cfg.max_text_chars:
                            continue
                        speaker = utt.get("speaker_id")
                        if speaker is not None:
                            speaker = str(speaker).strip() or None
                        cleaned_turns.append((text, speaker))

                    if not cleaned_turns:
                        continue

                    has_speaker_annotations = any(speaker is not None for _, speaker in cleaned_turns)

                    if not has_speaker_annotations:
                        doc_text = clean_text(" ".join(text for text, _ in cleaned_turns))
                        if cfg.min_text_chars <= len(doc_text) <= cfg.max_text_chars:
                            mono_docs.append(doc_text)
                        continue

                    if any(speaker is None for _, speaker in cleaned_turns):
                        missing_speaker_docs += 1
                        LOGGER.warning("Skipping document %s due to missing speaker_id", doc.get("id"))
                        continue

                    speakers = {speaker for _, speaker in cleaned_turns if speaker is not None}
                    if len(speakers) < 2:
                        # Treat pure monologues as unusable.
                        continue

                    doc_text = clean_text(" ".join(text for text, _ in cleaned_turns))
                    if cfg.min_text_chars <= len(doc_text) <= cfg.max_text_chars:
                        mono_docs.append(doc_text)

                    merged_turns: List[Tuple[str, str]] = []
                    current_speaker: str | None = None
                    buffer: List[str] = []

                    def flush_buffer(speaker: str | None) -> None:
                        if not buffer or speaker is None:
                            return
                        combined = clean_text(" ".join(buffer))
                        buffer.clear()
                        if not combined:
                            return
                        if cfg.min_text_chars <= len(combined) <= cfg.max_text_chars:
                            merged_turns.append((combined, speaker))

                    for text, speaker in cleaned_turns:
                        if current_speaker is None:
                            current_speaker = speaker
                            buffer.append(text)
                            continue
                        if speaker == current_speaker:
                            buffer.append(text)
                        else:
                            flush_buffer(current_speaker)
                            current_speaker = speaker
                            buffer.append(text)

                    flush_buffer(current_speaker)

                    if len(merged_turns) < 2:
                        continue

                    if cfg.pairing_mode == "turn_change":
                        for (a, speaker_a), (b, speaker_b) in zip(merged_turns, merged_turns[1:]):
                            if speaker_a != speaker_b:
                                pairs.append((a, b))
                    elif cfg.pairing_mode == "next_sentence":
                        pairs.extend((a, b) for (a, _), (b, _) in zip(merged_turns, merged_turns[1:]))
                    else:
                        raise ValueError(f"Unsupported pairing_mode: {cfg.pairing_mode}")
            else:
                prose_texts = _extract_top_level_texts(data, cfg.fallback_fields)
                for raw in prose_texts:
                    if not raw:
                        continue
                    if not (cfg.min_text_chars <= len(raw) <= cfg.max_text_chars):
                        continue
                    mono_docs.append(raw)

    if cfg.shuffle:
        rng = random.Random(cfg.seed)
        rng.shuffle(pairs)
        rng.shuffle(mono_docs)

    if not pairs and not mono_docs:
        raise RuntimeError("No usable text sequences were extracted from the dataset.")

    if missing_speaker_docs:
        LOGGER.debug("Documents skipped due to missing speaker_id: %d", missing_speaker_docs)

    return pairs, mono_docs


def create_labeled_sequences(
    pairs: Sequence[Tuple[str, str]], tokenizer
) -> List[Tuple[List[int], List[int]]]:
    """Create token/label sequences from dialogue pairs."""

    sequences: List[Tuple[List[int], List[int]]] = []
    bos = getattr(tokenizer, "bos_id")
    sep = getattr(tokenizer, "sep_id")
    eos = getattr(tokenizer, "eos_id")

    for context, reply in pairs:
        ctx_tokens = list(tokenizer.encode(context))
        rsp_tokens = list(tokenizer.encode(reply))

        input_ids = [bos] + ctx_tokens + [sep] + rsp_tokens + [eos]
        label_ids = [-100] * (len(ctx_tokens) + 2) + rsp_tokens + [eos]
        sequences.append((input_ids, label_ids))

    return sequences


def pack_token_sequences(
    sequences: Sequence[Tuple[Sequence[int], Sequence[int]]],
    max_length: int,
    pad_id: int,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]], int]:
    """Pack sequences into fixed-length chunks suitable for batching."""

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
        chunk_tokens = list(token_stream[start:end])
        chunk_labels = list(label_stream[start:end])
        attn = [1] * len(chunk_tokens)

        if len(chunk_tokens) < max_length:
            pad_len = max_length - len(chunk_tokens)
            chunk_tokens.extend([pad_id] * pad_len)
            chunk_labels.extend([-100] * pad_len)
            attn.extend([0] * pad_len)

        tokens_per_epoch += sum(attn)
        inputs.append(chunk_tokens)
        labels.append(chunk_labels)
        attention_masks.append(attn)

    return inputs, labels, attention_masks, tokens_per_epoch

