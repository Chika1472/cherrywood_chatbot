# Repository Guidelines

## Project Structure & Module Organization
- `Molu_chatbot_0919_fix_code.py` contains the end-to-end data prep, SentencePiece vocab training, and GPT-in-miniature training loop replicated from the notebook; treat it as the authoritative source for automation.
- `molu_chat_0920.ipynb` is the exploratory notebook executed by `smoke_test.py`; keep heavy cells guardable by the `SKIP_TRAINING` flag so automated runs stay fast.
- `smoke_test.py` replays all executable notebook cells with `SKIP_TRAINING=1` to validate imports and data staging; extend it when adding new critical cells.
- `update_train_dir.py` patches the notebook's checkpoint directory, ensuring artifacts land under `molu_pretrain/`; run it when the training output path changes.
- Large corpora such as `NIKL_DIALOGUE_2024_v1.0.zip` live at repo root; transient outputs (e.g., `molu_pretrain/model.pt`) stay git-ignored.

## Build, Test, and Development Commands
- `python Molu_chatbot_0919_fix_code.py`: end-to-end pipeline (zip -> pairs -> tokenizer -> training). Expect multi-hour GPU runs.
- `python smoke_test.py`: fast sanity check that notebook code executes with training skipped.
- `python update_train_dir.py`: regenerate notebook cell 11 when you retarget checkpoint directories.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, snake_case functions, and caps for module-level constants (`PAD`, `VOCAB_SIZE`).
- Prefer double quotes to match existing modules and keep literal Korean strings UTF-8 encoded.
- Use deterministic randomness (`random.Random(SEED)`) and derive device settings via `torch.cuda.is_available()` as already shown.

## Testing Guidelines
- New notebook cells that should run in CI must respect `SKIP_TRAINING` to avoid long GPU jobs.
- Add lightweight assertions or logging in `smoke_test.py` for new data transforms; keep runtime under 2 minutes.
- Manual training changes should include before/after validation loss snippets in PRs.

## Commit & Pull Request Guidelines
- Use imperative, present-tense commit subjects (`Add inference decoder`) and keep descriptions under 72 chars.
- Scope one logical change per commit; include rationale and links to experiment notes in the body when touching training loops.
- PRs should summarize dataset variants, mention required GPU specs, and attach sample chat outputs when model behavior changes.

## Data & Checkpoint Hygiene
- Do not commit artifacts under `molu_chatbot/` or `molu_pretrain/`; share checkpoints via object storage instead.
- Document tokenizer or config changes in PRs and bump `config.json` metadata accordingly.
