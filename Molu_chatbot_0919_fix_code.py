# Cell 1
# === pairs 생성 ===
ZIP_PATH = "/content/NIKL_DIALOGUE_2024_v1.0.zip"   # 경로
import zipfile, json, io, re, random

# 옵션
PAIRING_MODE = "turn_change"
TEXT_FIELD = "form"             # or "original_form"
MIN_LEN, MAX_LEN = 1, 160       # 길이 필터
SHUFFLE = True
SEED = 42

def clean_text(text: str) -> str:
    text = text.replace("\u200b", "").strip()
    text = re.sub(r"\s+", " ", text)
    return text

def is_korean(text: str) -> bool:
    kor = len(re.findall(r"[가-힣]", text))
    return kor >= max(2, int(0.2 * len(text)))

pairs = []
n_files = 0
rng = random.Random(SEED)
with zipfile.ZipFile(ZIP_PATH) as zf:
    for name in zf.namelist():
        if not name.lower().endswith(".json"):
            continue
        n_files += 1
        with zf.open(name) as f:
            data = json.load(io.TextIOWrapper(f, encoding="utf-8"))
        docs = data.get("document", [])
        for doc in docs:
            utts = doc.get("utterance", [])
            if any(("speaker_id" not in utt) or not utt.get("speaker_id") for utt in utts):
                print(f"[skip] {doc.get('id', name)} missing speaker_id metadata")
                continue
            utts = sorted(
                utts, key=lambda u: (u.get("start", float("inf")), u.get("id", ""))
            )

            sequence = []
            current_speaker = None
            buffer = []

            def flush_buffer():
                if not buffer:
                    return
                combined = clean_text(" ".join(buffer))
                buffer.clear()  # <- clear()
                if not combined:
                    return
                if len(combined) < MIN_LEN or len(combined) > MAX_LEN:
                    return
                if not is_korean(combined):
                    return
                sequence.append((combined, current_speaker))

            for utt in utts:
                if TEXT_FIELD == "form":
                    preferred = utt.get("form")
                elif TEXT_FIELD == "original_form":
                    preferred = utt.get("original_form")
                else:
                    raise ValueError("TEXT_FIELD must be 'form' or 'original_form'")

                raw = preferred or utt.get("form") or utt.get("original_form") or ""
                text = clean_text(raw)
                if not text:
                    continue
                if not is_korean(text):
                    continue
                if len(text) > MAX_LEN:
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

            if PAIRING_MODE == "turn_change":
                for (a, spk_a), (b, spk_b) in zip(sequence, sequence[1:]):
                    if spk_a != spk_b and a and b:
                        pairs.append((a, b))
            else:
                for (a, _), (b, _) in zip(sequence, sequence[1:]):
                    if a and b:
                        pairs.append((a, b))

if SHUFFLE:
    rng.shuffle(pairs)

print(f"JSON files read: {n_files}")
print(f"Total pairs: {len(pairs)}")
print("Sample:", pairs[0] if pairs else ("<empty>", "<empty>"))


# Cell 3
# sentencepiece tokenizatoin

import sentencepiece as spm

with open("corpus.txt", "w", encoding="utf-8") as f:
    for q, a in pairs:
        f.write(q.replace("\n"," ") + " <sep> " + a.replace("\n"," ") + "\n")

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="spm16k",
    vocab_size=16000,
    model_type="unigram",
    character_coverage=0.9998,
    pad_id=0, bos_id=1, eos_id=3, unk_id=4,
    control_symbols=["<sep>"],
    byte_fallback=True
)

sp = spm.SentencePieceProcessor(model_file="spm16k.model")

SPECIALS = ["<pad>","<bos>","<sep>","<eos>","<unk>"]
PAD = sp.pad_id(); BOS = sp.bos_id(); EOS = sp.eos_id(); UNK = sp.unk_id()
SEP = sp.piece_to_id("<sep>")
VOCAB_SIZE = sp.get_piece_size()

def encode_text(s: str):
    return sp.encode(s, out_type=int)

def decode_text(ids):
    drop = {PAD, BOS, SEP, EOS, UNK}
    return sp.decode([i for i in ids if i not in drop])

# Cell 4
sp = spm.SentencePieceProcessor(model_file="spm16k.model")

SPECIALS = ["<pad>","<bos>","<sep>","<eos>","<unk>"]
PAD = sp.pad_id(); BOS = sp.bos_id(); EOS = sp.eos_id(); UNK = sp.unk_id()
SEP = sp.piece_to_id("<sep>")
VOCAB_SIZE = sp.get_piece_size()

def encode_text(s: str):
    return sp.encode(s, out_type=int)

def decode_text(ids):
    drop = {PAD, BOS, SEP, EOS, UNK}
    return sp.decode([i for i in ids if i not in drop])

# Cell 5
labeled = []
for ctx, rsp in pairs:
    ctx_ids = [BOS] + encode_text(ctx) + [SEP]
    rsp_ids = encode_text(rsp) + [EOS]
    input_ids = ctx_ids + rsp_ids
    labels = [-100] * len(ctx_ids) + rsp_ids
    labeled.append((input_ids, labels))

print(f"Labeled pairs: {len(labeled)}")
print("Sample labeled pair:", labeled[0] if labeled else ("<empty>", "<empty>"))

# Cell 6
def pack_token_sequences(sequences, max_length, pad_id):
    inputs, labels, attention_masks = [], [], []
    tokens_per_epoch = 0

    token_stream = []
    label_stream = []
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


packed_inputs, packed_labels, packed_attn, tokens_per_epoch = pack_token_sequences(
    labeled, max_length=MAX_LEN, pad_id=PAD
)

print(f"Packed batches: {len(packed_inputs)}")
print(f"Tokens per epoch: {tokens_per_epoch}")
if packed_inputs:
    print("Sample packed input:", packed_inputs[0][: min(len(packed_inputs[0]), 32)])
    print("Sample packed labels:", packed_labels[0][: min(len(packed_labels[0]), 32)])
    print("Sample attention:", packed_attn[0][: min(len(packed_attn[0]), 32)])

# Cell 8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.nh = n_heads
        self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.nh, self.dk).transpose(1, 2)
        k = k.view(B, T, self.nh, self.dk).transpose(1, 2)
        v = v.view(B, T, self.nh, self.dk).transpose(1, 2)

        attn_bias = None
        if attn_mask is not None:
            if attn_mask.dim() != 2 or attn_mask.shape != (B, T):
                raise ValueError("attn_mask must be of shape (batch_size, sequence_length)")
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
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.1):
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

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPTScratch(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        d_model=512,
        n_layers=8,
        n_heads=8,
        max_len=MAX_LEN,
        dropout=0.1,
    ):
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
        self.apply(self._init)
        self.head.weight = self.tok_emb.weight

    def _init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, attention_mask=None):
        B, T = x.shape
        if T > self.max_len:
            x = x[:, -self.max_len :]
            T = x.shape[1]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.max_len :]
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(self.emb_norm(h))
        for blk in self.blocks:
            h = blk(h, attention_mask)
        h = self.ln_f(h)
        return self.head(h)

# Cell 9
def count_tokens_pairs(pairs, max_len=160):
    tot = 0
    for ctx, rsp in pairs:
        ids = [257] + encode_text(ctx) + [258] + encode_text(rsp) + [259]  # BOS/SEP/EOS
        tot += min(len(ids), max_len)
    return tot

def count_tokens_mono(docs, max_len=160):
    tot = 0
    for s in docs:
        ids = [257] + encode_text(s) + [259]
        tot += min(len(ids), max_len)
    return tot

TOK_PER_EPOCH = 0
if 'pairs' in globals() and len(pairs) > 0:
    TOK_PER_EPOCH += count_tokens_pairs(pairs, max_len=MAX_LEN)
if 'mono_docs' in globals() and len(mono_docs) > 0:
    TOK_PER_EPOCH += count_tokens_mono(mono_docs, max_len=MAX_LEN)

print(f"≈ tokens per epoch: {TOK_PER_EPOCH:,}")


# Cell 11
import torch, torch.nn.functional as F, time, json, os
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPTScratch(
    vocab_size=VOCAB_SIZE,
    d_model=512, n_layers=8, n_heads=8,
    max_len=MAX_LEN, dropout=0.1
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

def lm_loss(logits, targets):
    vocab = logits.size(-1)
    pred = logits[:, :-1].contiguous().view(-1, vocab)
    gold = targets[:, 1:].contiguous().view(-1)
    return F.cross_entropy(pred, gold, ignore_index=-100)

@torch.no_grad()
def evaluate():
    model.eval(); tot, n = 0.0, 0
    for b in val_dl:
        x = b["input_ids"].to(device); y = b["labels"].to(device); attn = b["attention_mask"].to(device)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            loss = lm_loss(model(x, attention_mask=attn), y)
        tot += loss.item(); n += 1
    return tot/max(1,n)

EPOCHS, ACCUM = 20, 4    # EPOCH
best = float("inf"); save_dir = "molu_chatbot"

# Create Datasets and DataLoaders
class PackedDataset(Dataset):
    def __init__(self, inputs, labels, attention_masks):
        self.inputs = inputs
        self.labels = labels
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.inputs[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
        }

# Split data into training and validation sets (80/20 split)
split_idx = int(len(packed_inputs) * 0.8)
train_dataset = PackedDataset(packed_inputs[:split_idx], packed_labels[:split_idx], packed_attn[:split_idx])
val_dataset = PackedDataset(packed_inputs[split_idx:], packed_labels[split_idx:], packed_attn[split_idx:])

BATCH_SIZE = 32 # Define batch size
train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE)


for ep in range(1, EPOCHS+1):
    model.train(); t0=time.time(); optimizer.zero_grad(set_to_none=True); step=0
    for i, b in enumerate(train_dl, 1):
        x = b["input_ids"].to(device); y = b["labels"].to(device); attn = b["attention_mask"].to(device)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            loss = lm_loss(model(x, attention_mask=attn), y) / ACCUM
        scaler.scale(loss).backward()
        if i % ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True); step += 1
            if step % 100 == 0: print(f"step {step} | loss {(loss.item()*ACCUM):.4f}")

    val = evaluate(); print(f"[epoch {ep}] val_loss={val:.4f} | {(time.time()-t0)/60:.1f} min")
    if val < best:
        best = val
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{save_dir}/model.pt")
        with open(f"{save_dir}/config.json","w") as f:
            json.dump({
                "vocab_size": VOCAB_SIZE, "d_model":512,"n_layers":8,"n_heads":8,
                "max_len":MAX_LEN,"dropout":0.1,
                "special_tokens":{"PAD":PAD,"BOS":BOS,"SEP":SEP,"EOS":EOS}
            }, f, indent=2)
        print("SAVED:", save_dir)

# Cell 13
import re, json, torch

SAVE_DIR = "molu_chatbot"  # 저장
device = "cuda" if torch.cuda.is_available() else "cpu"

# state_dict와 config 읽기
sd = torch.load(f"{SAVE_DIR}/model.pt", map_location="cpu")
with open(f"{SAVE_DIR}/config.json") as f:
    cfg = json.load(f)

# state_dict에서 실제 아키텍처 추론
def infer_arch_from_state_dict(state_dict):
    d_model = state_dict["tok_emb.weight"].shape[1]
    vocab_size = state_dict["tok_emb.weight"].shape[0]
    max_len = state_dict["pos_emb.weight"].shape[0]
    layer_idx = set()
    for k in state_dict.keys():
        m = re.match(r"blocks\.(\d+)\.", k)
        if m:
            layer_idx.add(int(m.group(1)))
    n_layers = (max(layer_idx) + 1) if layer_idx else cfg.get("n_layers", 6)
    return vocab_size, d_model, max_len, n_layers

vocab_size, d_model_sd, max_len_sd, n_layers_sd = infer_arch_from_state_dict(sd)

# n_heads 보정
def pick_heads(d_model, preferred=cfg.get("n_heads", 6)):
    divisors = [h for h in range(2, 65) if d_model % h == 0]
    if not divisors:
        raise ValueError(f"d_model={d_model}에 대한 유효한 n_heads가 없습니다.")
    if preferred in divisors:
        return preferred
    return min(divisors, key=lambda h: abs(h - preferred))

n_heads_fixed = pick_heads(d_model_sd, cfg.get("n_heads", 6))

print(f"[info] from state_dict: vocab={vocab_size}, d_model={d_model_sd}, max_len={max_len_sd}, n_layers={n_layers_sd}")
print(f"[info] cfg requested n_heads={cfg.get('n_heads', None)} -> using n_heads={n_heads_fixed}")

# GPTScratch
model = GPTScratch(
    vocab_size=vocab_size,
    d_model=d_model_sd,
    n_layers=n_layers_sd,
    n_heads=n_heads_fixed,   # 헤드 보정
    max_len=max_len_sd,
    dropout=cfg.get("dropout", 0.1),
).to(device).eval()

# 가중치 로드
missing, unexpected = model.load_state_dict(sd, strict=False)
print("loaded. missing:", missing, "unexpected:", unexpected)


# Cell 14
# === char inference ===
import torch

HIST_MAX = MAX_LEN
MAX_NEW  = 96

@torch.no_grad()
def sample_top_p(logits_row, top_p=0.9, temperature=0.7):
    logits_row = logits_row / max(1e-6, temperature)
    probs = torch.softmax(logits_row, dim=-1)
    sp, si = torch.sort(probs, descending=True)
    cum = torch.cumsum(sp, dim=-1)
    mask = cum > top_p
    mask[..., 0] = False
    sp = sp.masked_fill(mask, 0)
    if sp.sum() <= 0:
        return int(torch.argmax(probs).item())
    sp = sp / sp.sum()
    idx = torch.multinomial(sp, 1)
    return int(si[idx])

@torch.no_grad()
def generate_reply(prompt: str,
                   max_ctx: int = HIST_MAX,
                   max_new_tokens: int = MAX_NEW,
                   top_p=0.9, temperature=0.7):
    history = [BOS] + encode_text(prompt) + [SEP]
    history = history[-max_ctx:]
    x = torch.tensor(history, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits = model(x)
        nxt = sample_top_p(logits[0, -1], top_p=top_p, temperature=temperature)
        history.append(nxt)
        x = torch.tensor(history[-max_ctx:], dtype=torch.long, device=device).unsqueeze(0)
        if nxt == EOS: break
    try:
        last_sep = len(history) - 1 - history[::-1].index(SEP)
    except ValueError:
        last_sep = 0
    try:
        last_eos = len(history) - 1 - history[::-1].index(EOS)
    except ValueError:
        last_eos = len(history)
    return decode_text(history[last_sep:last_eos]).strip()


# Cell 16
print(generate_reply("넷플릭스에서 요즘 뭐 봐?"))