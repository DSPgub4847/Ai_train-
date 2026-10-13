import re
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/cyber/train.txt"
VOCAB_FILE = "data/cyber/vocab.txt"
MODEL_FILE = "data/cyber/model_v3.npz"

CONTEXT_SIZE = 4
EMBEDDING_SIZE = 16
HIDDEN_SIZE = 32

LEARNING_RATE = 0.5
EPOCHS = 300

SEED = 42


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading dataset...")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    text = f.read().lower()

tokens = re.findall(
    r"\b[a-z]+\b|[.,!?]",
    text
)

print("Tokens:", len(tokens))


# ============================================================
# 2. LOAD VOCABULARY
# ============================================================

vocab = []

with open(VOCAB_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        # vocab.txt:
        #
        # 0 <PAD>
        # 1 <UNK>
        # 2 <BOS>
        # 3 <EOS>
        # 4 .
        # 5 a

        parts = line.split(maxsplit=1)

        if len(parts) == 2:

            index = parts[0]
            token = parts[1]

            vocab.append(token)

        else:

            vocab.append(parts[0])


stoi = {
    token: i
    for i, token in enumerate(vocab)
}

itos = {
    i: token
    for i, token in enumerate(vocab)
}

vocab_size = len(vocab)

print("Vocabulary:", vocab_size)

print(
    "<PAD> ID:",
    stoi["<PAD>"]
)

print(
    "<UNK> ID:",
    stoi["<UNK>"]
)


# ============================================================
# 3. ENCODE TEXT
# ============================================================

encoded = np.array(
    [
        stoi.get(
            token,
            stoi["<UNK>"]
        )
        for token in tokens
    ],
    dtype=np.int64
)

print(
    "Encoded tokens:",
    len(encoded)
)


# ============================================================
# 4. CREATE TRAINING DATA
# ============================================================

contexts = []
targets = []

for i in range(
    len(encoded) - CONTEXT_SIZE
):

    context = encoded[
        i:i + CONTEXT_SIZE
    ]

    target = encoded[
        i + CONTEXT_SIZE
    ]

    contexts.append(context)
    targets.append(target)


contexts = np.array(
    contexts,
    dtype=np.int64
)

targets = np.array(
    targets,
    dtype=np.int64
)

N = len(contexts)

print(
    "Training examples:",
    N
)


# ============================================================
# 5. INITIALIZE MODEL
# ============================================================

rng = np.random.default_rng(SEED)


# Embedding matrix
E = (
    rng.normal(
        0,
        0.05,
        (
            vocab_size,
            EMBEDDING_SIZE
        )
    )
    .astype(np.float32)
)


# Hidden layer
W1 = (
    rng.normal(
        0,
        0.05,
        (
            CONTEXT_SIZE * EMBEDDING_SIZE,
            HIDDEN_SIZE
        )
    )
    .astype(np.float32)
)

b1 = np.zeros(
    HIDDEN_SIZE,
    dtype=np.float32
)


# Output layer
W2 = (
    rng.normal(
        0,
        0.05,
        (
            HIDDEN_SIZE,
            vocab_size
        )
    )
    .astype(np.float32)
)

b2 = np.zeros(
    vocab_size,
    dtype=np.float32
)


# ============================================================
# 6. SOFTMAX
# ============================================================

def softmax(logits):

    logits = (
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True
        )
    )

    exp_logits = np.exp(logits)

    return (
        exp_logits
        / np.sum(
            exp_logits,
            axis=1,
            keepdims=True
        )
    )


# ============================================================
# 7. TRAINING
# ============================================================

print()
print("======================================")
print("Starting training...")
print("======================================")
print()


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    x = E[contexts]

    x_flat = x.reshape(
        N,
        CONTEXT_SIZE * EMBEDDING_SIZE
    )

    h_pre = (
        x_flat @ W1
        + b1
    )

    h = np.tanh(h_pre)

    logits = (
        h @ W2
        + b2
    )

    probs = softmax(logits)


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    correct_probs = probs[
        np.arange(N),
        targets
    ]

    loss = -np.mean(
        np.log(
            correct_probs
            + 1e-10
        )
    )


    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    dlogits = probs.copy()

    dlogits[
        np.arange(N),
        targets
    ] -= 1

    dlogits /= N


    # Output gradients

    dW2 = (
        h.T @ dlogits
    )

    db2_grad = np.sum(
        dlogits,
        axis=0
    )


    # Hidden gradients

    dh = (
        dlogits @ W2.T
    )

    dh_pre = (
        dh
        * (1 - h * h)
    )


    # First layer gradients

    dW1 = (
        x_flat.T @ dh_pre
    )

    db1_grad = np.sum(
        dh_pre,
        axis=0
    )


    # Embedding gradients

    dx_flat = (
        dh_pre @ W1.T
    )

    dx = dx_flat.reshape(
        N,
        CONTEXT_SIZE,
        EMBEDDING_SIZE
    )

    dE = np.zeros_like(E)

    np.add.at(
        dE,
        contexts,
        dx
    )


    # --------------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------------

    np.clip(
        dE,
        -1.0,
        1.0,
        out=dE
    )

    np.clip(
        dW1,
        -1.0,
        1.0,
        out=dW1
    )

    np.clip(
        db1_grad,
        -1.0,
        1.0,
        out=db1_grad
    )

    np.clip(
        dW2,
        -1.0,
        1.0,
        out=dW2
    )

    np.clip(
        db2_grad,
        -1.0,
        1.0,
        out=db2_grad
    )


    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    E -= LEARNING_RATE * dE

    W1 -= LEARNING_RATE * dW1

    b1 -= LEARNING_RATE * db1_grad

    W2 -= LEARNING_RATE * dW2

    b2 -= LEARNING_RATE * db2_grad


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        epoch % 25 == 0
        or epoch == EPOCHS - 1
    ):

        print(
            f"Epoch {epoch:3d} | "
            f"Loss: {loss:.4f}"
        )


# ============================================================
# 8. PREDICTION
# ============================================================

def predict_next(
    sentence,
    top_k=5
):

    words = re.findall(
        r"\b[a-z]+\b|[.,!?]",
        sentence.lower()
    )

    ids = [
        stoi.get(
            word,
            stoi["<UNK>"]
        )
        for word in words
    ]


    # Padding

    if len(ids) < CONTEXT_SIZE:

        ids = (
            [stoi["<PAD>"]]
            * (
                CONTEXT_SIZE
                - len(ids)
            )
            + ids
        )


    # Last context

    ids = ids[
        -CONTEXT_SIZE:
    ]


    # Embeddings

    x = E[
        np.array(
            ids,
            dtype=np.int64
        )
    ]


    x = x.reshape(
        1,
        -1
    )


    # Hidden layer

    h = np.tanh(
        x @ W1
        + b1
    )


    # Output

    logits = (
        h @ W2
        + b2
    )


    # Softmax

    logits -= np.max(logits)

    p = np.exp(logits)

    p /= np.sum(p)


    # Top K

    top = np.argsort(
        p[0]
    )[-top_k:][::-1]


    results = []

    for i in top:

        index = int(i)

        results.append(
            (
                itos[index],
                float(
                    p[0, index]
                )
            )
        )

    return results


# ============================================================
# 9. TEST MODEL
# ============================================================

print()
print("======================================")
print("MODEL PREDICTIONS")
print("======================================")
print()


tests = [

    "sql injection occurs when",

    "authentication verifies the identity",

    "a firewall controls network",

    "strong passwords protect",

    "https uses",

    "cybersecurity is the",

    "encryption protects",

    "incident response helps",

]


for prompt in tests:

    print(
        "Prompt:",
        prompt
    )

    predictions = predict_next(
        prompt,
        top_k=5
    )

    for token, probability in predictions:

        print(
            f"  {token:15s}"
            f" {probability:.4f}"
        )

    print()


# ============================================================
# 10. SAVE MODEL
# ============================================================

np.savez(
    MODEL_FILE,

    E=E,

    W1=W1,

    b1=b1,

    W2=W2,

    b2=b2
)


print("======================================")
print("TRAINING COMPLETE")
print("======================================")
print()

print(
    "Model saved:"
)

print(
    MODEL_FILE
)

print()
