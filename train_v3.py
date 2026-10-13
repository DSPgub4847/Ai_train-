import re
from collections import Counter

# =========================
# 1. LOAD DATA
# =========================

with open("data/cyber/train.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# =========================
# 2. TOKENIZER
# =========================

tokens = re.findall(r"\b[a-z]+\b|[.,!?]", text)

print("Total tokens:", len(tokens))

# =========================
# 3. BUILD VOCABULARY
# =========================

counter = Counter(tokens)

special_tokens = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
]

vocab = special_tokens + sorted(counter.keys())

stoi = {
    token: i
    for i, token in enumerate(vocab)
}

itos = {
    i: token
    for token, i in stoi.items()
}

print("Vocabulary size:", len(vocab))

# =========================
# 4. ENCODE
# =========================

encoded = [
    stoi.get(token, stoi["<UNK>"])
    for token in tokens
]

print("Encoded tokens:", len(encoded))

# =========================
# 5. SHOW TOKENIZATION
# =========================

print("\n--- Tokenizer Example ---")

example = tokens[:30]

for token in example:
    print(
        f"{token:20s} -> {stoi[token]}"
    )

# =========================
# 6. SAVE VOCABULARY
# =========================

with open("data/cyber/vocab.txt", "w", encoding="utf-8") as f:

    for i, token in enumerate(vocab):
        f.write(f"{i}\t{token}\n")

print("\nVocabulary saved:")
print("data/cyber/vocab.txt")

# =========================
# 7. SAMPLE
# =========================

print("\n--- First 50 tokens ---")
print(tokens[:50])
