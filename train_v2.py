import math
import random

# =========================
# 1. DATASET
# =========================

text = open("data/train.txt", "r", encoding="utf-8").read()

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

data = [stoi[ch] for ch in text]

print("Vocabulary:", vocab_size)
print("Training characters:", len(data))


# =========================
# 2. TRAINING SETTINGS
# =========================

context_size = 4
hidden_size = 32

learning_rate = 0.05
epochs = 1000


# =========================
# 3. WEIGHT INITIALIZATION
# =========================

def matrix(rows, cols, scale=0.01):
    return [
        [random.uniform(-scale, scale) for _ in range(cols)]
        for _ in range(rows)
    ]


# Input = 4 characters
input_size = context_size * vocab_size

W1 = matrix(input_size, hidden_size)
b1 = [0.0] * hidden_size

W2 = matrix(hidden_size, vocab_size)
b2 = [0.0] * vocab_size


# =========================
# 4. ACTIVATION FUNCTIONS
# =========================

def tanh(x):
    return math.tanh(x)


def tanh_derivative(x):
    t = math.tanh(x)
    return 1.0 - t * t


def softmax(logits):
    maximum = max(logits)

    exp_values = [
        math.exp(x - maximum)
        for x in logits
    ]

    total = sum(exp_values)

    return [
        x / total
        for x in exp_values
    ]


# =========================
# 5. FORWARD PASS
# =========================

def forward(context):

    # One-hot input
    x = [0.0] * input_size

    for position, char_id in enumerate(context):
        index = position * vocab_size + char_id
        x[index] = 1.0

    # Hidden layer
    hidden_raw = []

    for j in range(hidden_size):

        value = b1[j]

        for i in range(input_size):
            value += x[i] * W1[i][j]

        hidden_raw.append(value)

    hidden = [
        tanh(v)
        for v in hidden_raw
    ]

    # Output layer
    logits = []

    for j in range(vocab_size):

        value = b2[j]

        for i in range(hidden_size):
            value += hidden[i] * W2[i][j]

        logits.append(value)

    probabilities = softmax(logits)

    return x, hidden_raw, hidden, probabilities


# =========================
# 6. TRAINING
# =========================

for epoch in range(epochs):

    total_loss = 0.0
    examples = 0

    for i in range(context_size, len(data)):

        context = data[i - context_size:i]
        target = data[i]

        # Forward
        x, hidden_raw, hidden, probabilities = forward(context)

        probability = max(
            probabilities[target],
            1e-12
        )

        loss = -math.log(probability)
        total_loss += loss
        examples += 1

        # =====================
        # BACKPROP OUTPUT
        # =====================

        d_logits = probabilities[:]
        d_logits[target] -= 1.0

        # Gradients W2 / b2
        dW2 = matrix(hidden_size, vocab_size, 0.0)
        db2 = [0.0] * vocab_size

        for j in range(vocab_size):

            db2[j] = d_logits[j]

            for k in range(hidden_size):
                dW2[k][j] = hidden[k] * d_logits[j]

        # =====================
        # BACKPROP HIDDEN
        # =====================

        d_hidden = [0.0] * hidden_size

        for k in range(hidden_size):

            value = 0.0

            for j in range(vocab_size):
                value += W2[k][j] * d_logits[j]

            d_hidden[k] = value

        d_hidden_raw = [0.0] * hidden_size

        for k in range(hidden_size):
            d_hidden_raw[k] = (
                d_hidden[k]
                * tanh_derivative(hidden_raw[k])
            )

        # =====================
        # GRADIENT W1 / b1
        # =====================

        dW1 = matrix(input_size, hidden_size, 0.0)
        db1 = [0.0] * hidden_size

        for k in range(hidden_size):

            db1[k] = d_hidden_raw[k]

            for j in range(input_size):
                dW1[j][k] = x[j] * d_hidden_raw[k]

        # =====================
        # UPDATE WEIGHTS
        # =====================

        for j in range(input_size):
            for k in range(hidden_size):
                W1[j][k] -= learning_rate * dW1[j][k]

        for k in range(hidden_size):
            b1[k] -= learning_rate * db1[k]

        for k in range(hidden_size):
            for j in range(vocab_size):
                W2[k][j] -= learning_rate * dW2[k][j]

        for j in range(vocab_size):
            b2[j] -= learning_rate * db2[j]

    if epoch % 100 == 0:
        average_loss = total_loss / max(examples, 1)

        print(
            f"Epoch {epoch:4d} | "
            f"Loss: {average_loss:.4f}"
        )


# =========================
# 7. TEXT GENERATION
# =========================

def generate(start="Hell", length=150):

    context = list(start)

    # Ensure context is exactly context_size
    if len(context) < context_size:
        context = (
            [" "] * (context_size - len(context))
            + context
        )

    context = context[-context_size:]

    output = "".join(context)

    for _ in range(length):

        context_ids = [
            stoi.get(ch, 0)
            for ch in context
        ]

        _, _, _, probabilities = forward(context_ids)

        next_id = random.choices(
            range(vocab_size),
            weights=probabilities
        )[0]

        next_char = itos[next_id]

        output += next_char

        context.append(next_char)
        context = context[-context_size:]

    return output


print()
print("========== GENERATED TEXT ==========")
print(generate("Hell", 150))
