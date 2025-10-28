# Molu_chatbot

Korean Molu Chatbot

This repository now contains a configurable training pipeline that can be executed
directly from Google Colab. The `molu_chatbot.py` script handles dataset
preparation, SentencePiece tokenisation, model training, checkpoint saving and
inference utilities. Dialogue corpora remain first-class citizens, but the data
loader now also understands prose-style JSON exports such as news articles with
top-level `text` or `content` fields.

## Quick start on Google Colab

1. Upload the NIKL dialogue zip file (e.g. `NIKL_DIALOGUE_2024_v1.0.zip`) to
   your Colab runtime or Google Drive.
2. Run the following commands in a Colab notebook cell:

```python
!pip install -q sentencepiece tqdm
!git clone https://github.com/<your-account>/Molu_chatbot.git
%cd Molu_chatbot
!python molu_chatbot.py train \
    --zip_path /content/NIKL_DIALOGUE_2024_v1.0.zip \
    --output_dir /content/molu_ckpt \
    --epochs 3 \
    --batch_size 64 \
    --max_seq_len 160
```

> Replace `<your-account>` with the GitHub owner of the repository you cloned
> from.

By default the trainer now builds examples at each speaker change so prompts and
replies always come from different participants. Pass
`--pairing_mode next_sentence` if you prefer the legacy adjacent-utterance
pairing.

During training you will see tqdm progress bars with token-level losses. The
best checkpoint (based on validation loss) and all configuration metadata are
saved under `--output_dir`. Training history is written to
`training_log.json`.

## Important CLI options

```
python molu_chatbot.py train --zip_path <path/to/data.zip> [options]

--pairing_mode      How to pair utterances (`turn_change` by default).
--text_field        Which field to read from each utterance (`form` or `original_form`).
--vocab_size        SentencePiece vocabulary size (default 16000).
--d_model / --n_layers / --n_heads  Transformer dimensions.
--grad_accum        Gradient accumulation steps for large effective batches.
--warmup_ratio      Fraction of total updates used for LR warm-up.
```

All options have sensible defaults so you can start with only `--zip_path`
and `--output_dir`.

## Chatting with a trained model

Once training finishes (or when you download a pre-trained checkpoint), you
can generate replies directly from the command line:

```python
!python molu_chatbot.py chat \
    --model_dir /content/molu_ckpt \
    --prompt "바다로 여행가면 좋은점이 뭘까?"
```

This command loads `model.pt` and `config.json` from the output directory,
performs top-p sampling and prints the chatbot's response.

## Training features

- SentencePiece tokeniser training with automatic reuse when a model already
  exists.
- Packed token dataset generation that respects maximum sequence lengths.
- Mixed-precision training (when CUDA is available) with gradient accumulation
  and gradient clipping.
- Cosine learning rate schedule with configurable warm-up.
- Automatic validation, checkpointing and logging of metrics per epoch.

## 플릿 일정 웹 캘린더

`calendar_app/` 디렉터리에는 FullCalendar 기반의 정적 웹 페이지가 포함되어 있어
날짜와 시간을 드래그해 선택하고 "플릿 컴포", "독트린", "핸드아웃 여부" 정보를
입력해 일정을 기록할 수 있습니다. 정적 파일(`calendar_app/static/`)을 임의의 웹
서버 또는 호스팅 서비스에 올려 간단히 사용할 수 있으며, 자세한 내용은
`calendar_app/README.md`에서 확인하세요.
