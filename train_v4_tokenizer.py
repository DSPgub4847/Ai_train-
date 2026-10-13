import re
from collections import Counter

DATA_FILE = "data/cyber_v4/train.txt"
VOCAB_FILE = "data/cyber_v4/vocab.txt"

# -----------------------------------------
# Load dataset
# -----------------------------------------

with open(DATA_FILE, "r", encoding="utf-8") as f:
    text = f.read().lower()

# -----------------------------------------
# Tokenizer
# -----------------------------------------

tokens = re.findall(
    r"\b[a-z]+\b|[.,!?]",
    text
)

# -----------------------------------------
# Special tokens
# -----------------------------------------

special_tokens = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
]

# -----------------------------------------
# Build vocabulary
# -----------------------------------------

counter = Counter(tokens)

vocab_tokens = sorted(
    counter.keys()
)

vocab = (
    special_tokens
    + vocab_tokens
)

stoi = {
    token: i
    for i, token in enumerate(vocab)
}

itos = {
    i: token
    for i, token in enumerate(vocab)
}

# -----------------------------------------
# Encode
# -----------------------------------------

encoded = [
    stoi.get(
        token,
        stoi["<UNK>"]
    )
    for token in tokens
]

# -----------------------------------------
# Save vocabulary
# -----------------------------------------

with open(
    VOCAB_FILE,
    "w",
    encoding="utf-8"
) as f:

    for i, token in enumerate(vocab):

        f.write(
            f"{i}\t{token}\n"
        )

# -----------------------------------------
# Statistics
# -----------------------------------------

print("======================================")
print("V4 TOKENIZER")
print("======================================")

print()

print(
    "Total tokens:",
    len(tokens)
)

print(
    "Vocabulary size:",
    len(vocab)
)

print(
    "Encoded tokens:",
    len(encoded)
)

print()

print("Special tokens:")

for token in special_tokens:

    print(
        f"{token:8s} -> {stoi[token]}"
    )

print()

print("First 40 tokens:")

print(tokens[:40])

print()

print("Vocabulary saved:")

print(VOCAB_FILE)
