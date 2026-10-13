import math
import random

# -----------------------------
# 1. Training data
# -----------------------------

text = open("data/train.txt", "r", encoding="utf-8").read()

# Character-level vocabulary
chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

data = [stoi[ch] for ch in text]

print("Characters:", vocab_size)
print("Training characters:", len(data))


# -----------------------------
# 2. Simple model
# -----------------------------

class TinyModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

        # Very small character transition table
        self.weights = [
            [random.uniform(-0.01, 0.01) for _ in range(vocab_size)]
            for _ in range(vocab_size)
        ]

    def predict(self, current_char):
        row = self.weights[current_char]

        # Softmax
        exp_values = [math.exp(x) for x in row]
        total = sum(exp_values)

        return [x / total for x in exp_values]


model = TinyModel(vocab_size)


# -----------------------------
# 3. Training
# -----------------------------

epochs = 1000
learning_rate = 0.05

for epoch in range(epochs):

    loss = 0.0

    for i in range(len(data) - 1):

        current = data[i]
        target = data[i + 1]

        probabilities = model.predict(current)

        # Cross entropy loss
        probability = max(probabilities[target], 1e-12)
        loss -= math.log(probability)

        # Gradient update
        for j in range(vocab_size):
            gradient = probabilities[j]

            if j == target:
                gradient -= 1

            model.weights[current][j] -= learning_rate * gradient

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")


# -----------------------------
# 4. Generate text
# -----------------------------

def generate(start, length=100):

    current = stoi.get(start, data[0])
    output = start

    for _ in range(length):

        probabilities = model.predict(current)

        next_char = random.choices(
            range(vocab_size),
            weights=probabilities
        )[0]

        output += itos[next_char]
        current = next_char

    return output


print("\n--- Generated text ---")
print(generate("H", 100))
