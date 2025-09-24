import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from molu_chatbot import (
    DataConfig,
    create_labeled_sequences,
    load_dialogue_pairs,
    pack_token_sequences,
)


class DummyTokenizer:
    bos_id = 101
    sep_id = 102
    eos_id = 103
    pad_id = 0

    _mapping = {
        "hello": [11, 12],
        "world": [21, 22],
        "context": [31],
        "answer": [41, 42, 43],
    }

    def encode(self, text: str):
        try:
            return self._mapping[text]
        except KeyError:  # pragma: no cover - defensive
            base = 50 + len(text)
            return [base, base + 1]


def make_zip(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("doc.json", json.dumps(payload, ensure_ascii=False))
    return path


def unpack_pairs(result):
    if isinstance(result, tuple):
        if len(result) == 0:
            return [], []
        if len(result) == 1:
            return result[0], []
        return result[0], result[1]
    if hasattr(result, "pairs") and hasattr(result, "mono_docs"):
        return result.pairs, result.mono_docs
    if hasattr(result, "pairs"):
        return result.pairs, getattr(result, "mono_docs", [])
    return result, []


def test_pack_sequences_preserves_eos():
    tokenizer = DummyTokenizer()
    pairs = [("hello", "world")]
    sequences = create_labeled_sequences(pairs, tokenizer)
    assert sequences[0][1][-1] == tokenizer.eos_id

    inputs, labels, attention, tokens = pack_token_sequences(
        sequences, max_length=16, pad_id=tokenizer.pad_id
    )

    assert tokens > 0
    flat_labels = [
        label
        for chunk_labels, mask in zip(labels, attention)
        for label, flag in zip(chunk_labels, mask)
        if flag == 1
    ]
    assert flat_labels[-1] == tokenizer.eos_id


def test_load_dialogue_pairs_skips_monologue(tmp_path: Path):
    zip_path = tmp_path / "monologue.zip"
    with open(Path(__file__).resolve().parent.parent / "example.json", "r", encoding="utf-8") as src:
        data = json.load(src)
    make_zip(zip_path, data)

    cfg = DataConfig(
        zip_path=str(zip_path),
        pairing_mode="next_sentence",
        text_field="form",
        shuffle=False,
    )
    with pytest.raises(RuntimeError):
        load_dialogue_pairs(cfg)


def test_load_dialogue_pairs_conversation(tmp_path: Path):
    payload = {
        "id": "conv",
        "document": [
            {
                "id": "conv.1",
                "utterance": [
                    {
                        "id": "u1",
                        "form": "안녕하세요",
                        "original_form": "안녕하세요",
                        "speaker_id": "A",
                        "start": 0.0,
                        "end": 1.0,
                        "note": "",
                    },
                    {
                        "id": "u2",
                        "form": "반가워요",
                        "original_form": "반가워요",
                        "speaker_id": "B",
                        "start": 1.1,
                        "end": 2.0,
                        "note": "",
                    },
                    {
                        "id": "u3",
                        "form": "오늘 기분이 어때요?",
                        "original_form": "오늘 기분이 어때요?",
                        "speaker_id": "A",
                        "start": 2.1,
                        "end": 3.0,
                        "note": "",
                    },
                ],
            }
        ],
    }

    zip_path = make_zip(tmp_path / "conversation.zip", payload)

    cfg = DataConfig(
        zip_path=str(zip_path),
        pairing_mode="next_sentence",
        text_field="form",
        shuffle=False,
    )
    pairs, mono_docs = unpack_pairs(load_dialogue_pairs(cfg))
    assert pairs == [
        ("안녕하세요", "반가워요"),
        ("반가워요", "오늘 기분이 어때요?"),
    ]
    assert any("안녕하세요" in doc for doc in mono_docs)

    cfg_turn = DataConfig(
        zip_path=str(zip_path),
        pairing_mode="turn_change",
        text_field="form",
        shuffle=False,
    )
    pairs_turn, _ = unpack_pairs(load_dialogue_pairs(cfg_turn))
    assert pairs_turn == [
        ("안녕하세요", "반가워요"),
        ("반가워요", "오늘 기분이 어때요?"),
    ]


def test_load_dialogue_pairs_merges_consecutive_turns(tmp_path: Path):
    payload = {
        "id": "conv",
        "document": [
            {
                "id": "conv.1",
                "utterance": [
                    {
                        "id": "u1",
                        "form": "안녕",
                        "speaker_id": "A",
                        "start": 0.0,
                        "end": 1.0,
                    },
                    {
                        "id": "u2",
                        "form": "반가워",
                        "speaker_id": "A",
                        "start": 1.1,
                        "end": 2.0,
                    },
                    {
                        "id": "u3",
                        "form": "잘 지냈어",
                        "speaker_id": "B",
                        "start": 2.1,
                        "end": 3.0,
                    },
                    {
                        "id": "u4",
                        "form": "오늘 뭐해?",
                        "speaker_id": "B",
                        "start": 3.1,
                        "end": 4.0,
                    },
                ],
            }
        ],
    }

    zip_path = make_zip(tmp_path / "merge.zip", payload)
    cfg = DataConfig(
        zip_path=str(zip_path),
        pairing_mode="turn_change",
        text_field="form",
        shuffle=False,
    )
    pairs, _ = unpack_pairs(load_dialogue_pairs(cfg))
    assert pairs == [("안녕 반가워", "잘 지냈어 오늘 뭐해?")]


def test_load_dialogue_pairs_skips_missing_speaker(tmp_path: Path, caplog):
    caplog.set_level("WARNING")
    payload = {
        "document": [
            {
                "id": "missing",
                "utterance": [
                    {
                        "id": "u1",
                        "form": "첫 발화",
                        "start": 0.0,
                        "end": 1.0,
                    },
                    {
                        "id": "u2",
                        "form": "두번째 발화",
                        "speaker_id": "A",
                        "start": 1.1,
                        "end": 2.0,
                    },
                ],
            },
            {
                "id": "valid",
                "utterance": [
                    {
                        "id": "v1",
                        "form": "안녕",
                        "speaker_id": "A",
                        "start": 0.0,
                        "end": 1.0,
                    },
                    {
                        "id": "v2",
                        "form": "반가워",
                        "speaker_id": "B",
                        "start": 1.1,
                        "end": 2.0,
                    },
                ],
            },
        ]
    }

    zip_path = make_zip(tmp_path / "missing_speaker.zip", payload)
    cfg = DataConfig(
        zip_path=str(zip_path),
        text_field="form",
        shuffle=False,
    )

    pairs, _ = unpack_pairs(load_dialogue_pairs(cfg))
    assert pairs == [("안녕", "반가워")]
    assert "missing speaker_id" in caplog.text


def test_load_dialogue_pairs_handles_prose_documents(tmp_path: Path):
    payload = {
        "id": "news-1",
        "text": "오늘은 맑은 하늘과 따뜻한 바람이 불어오는 봄날입니다.",
        "summary": "맑은 봄날 기사",
    }

    zip_path = make_zip(tmp_path / "news.zip", payload)
    cfg = DataConfig(
        zip_path=str(zip_path),
        text_field="form",
        shuffle=False,
    )

    pairs, mono_docs = unpack_pairs(load_dialogue_pairs(cfg))
    assert isinstance(pairs, list)
    assert pairs == []
    assert mono_docs, "Expected prose documents to populate mono_docs"
    assert any("맑은 하늘" in doc for doc in mono_docs)
