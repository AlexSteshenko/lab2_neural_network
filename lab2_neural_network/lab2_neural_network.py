"""
Lab 2: A neural network from scratch (Ch. 3)
Run: python lab2_neural_network.py
Dependencies: numpy, scikit-learn, matplotlib
"""
from pathlib import Path
from time import perf_counter
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
OUT = Path("results")
OUT.mkdir(exist_ok=True)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def one_hot(y, classes=10):
    return np.eye(classes)[y]


def init_params(input_dim=64, width=16, output_dim=10, seed=42):
    rng = np.random.default_rng(seed)
    # Shapes: W1 (64,W), W2 (W,W), W3 (W,10); biases (1, units)
    return {
        "W1": rng.normal(0, 0.01, (input_dim, width)),
        "b1": np.zeros((1, width)),
        "W2": rng.normal(0, 0.01, (width, width)),
        "b2": np.zeros((1, width)),
        "W3": rng.normal(0, 0.01, (width, output_dim)),
        "b3": np.zeros((1, output_dim)),
    }


def forward(X, p):
    z1 = X @ p["W1"] + p["b1"]
    a1 = np.maximum(0, z1)
    z2 = a1 @ p["W2"] + p["b2"]
    a2 = np.maximum(0, z2)
    z3 = a2 @ p["W3"] + p["b3"]
    probs = softmax(z3)
    cache = (X, z1, a1, z2, a2, probs)
    return probs, cache


def cross_entropy(probs, y_onehot):
    # Clip only to avoid log(0) in numerical edge cases.
    return -np.mean(np.sum(y_onehot * np.log(np.clip(probs, 1e-15, 1.0)), axis=1))


def backward(y_onehot, p, cache):
    X, z1, a1, z2, a2, probs = cache
    n = X.shape[0]
    dz3 = (probs - y_onehot) / n
    grads = {}
    grads["W3"] = a2.T @ dz3
    grads["b3"] = np.sum(dz3, axis=0, keepdims=True)
    da2 = dz3 @ p["W3"].T
    dz2 = da2 * (z2 > 0)
    grads["W2"] = a1.T @ dz2
    grads["b2"] = np.sum(dz2, axis=0, keepdims=True)
    da1 = dz2 @ p["W2"].T
    dz1 = da1 * (z1 > 0)
    grads["W1"] = X.T @ dz1
    grads["b1"] = np.sum(dz1, axis=0, keepdims=True)
    return grads


def predict(X, p):
    return np.argmax(forward(X, p)[0], axis=1)


def gradient_check():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(5, 64))
    y = np.array([0, 2, 4, 6, 8])
    Y = one_hot(y)
    p = init_params(width=8, seed=9)
    probs, cache = forward(X, p)
    grads = backward(Y, p, cache)
    eps = 1e-5
    checks = []
    # Nudge ten randomly selected weight entries, as required.
    keys = ["W1", "W2", "W3"]
    picks = [(k, tuple(rng.integers(0, s) for s in p[k].shape)) for k in keys for _ in range(4)]
    picks = picks[:10]
    for k, idx in picks:
        old = p[k][idx]
        p[k][idx] = old + eps
        lp = cross_entropy(forward(X, p)[0], Y)
        p[k][idx] = old - eps
        lm = cross_entropy(forward(X, p)[0], Y)
        p[k][idx] = old
        numerical = (lp - lm) / (2 * eps)
        analytic = grads[k][idx]
        rel = abs(analytic - numerical) / max(1e-12, abs(analytic) + abs(numerical))
        checks.append((k, idx, analytic, numerical, rel))
    return checks


def train(X_train, Y_train, X_test, y_test, width, lr, epochs=50, batch_size=32, seed=42):
    rng = np.random.default_rng(seed)
    p = init_params(width=width, seed=seed)
    history = []
    n = len(X_train)
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            ids = order[start:start + batch_size]
            probs, cache = forward(X_train[ids], p)
            grads = backward(Y_train[ids], p, cache)
            for key in p:
                p[key] -= lr * grads[key]
        train_probs = forward(X_train, p)[0]
        loss = cross_entropy(train_probs, Y_train)
        acc = np.mean(np.argmax(train_probs, axis=1) == np.argmax(Y_train, axis=1))
        test_acc = np.mean(predict(X_test, p) == y_test)
        history.append((epoch + 1, loss, acc, test_acc))
        print(f"width={width:2d} lr={lr:g} epoch={epoch+1:02d}/50 "
              f"loss={loss:.4f} train_acc={acc:.4f} test_acc={test_acc:.4f}")
    return p, history


def timing(X, Y, p, runs=100):
    # Warm-up, then measure forward and backward independently; report medians.
    for _ in range(5):
        probs, cache = forward(X, p)
        backward(Y, p, cache)
    f_times, b_times = [], []
    for _ in range(runs):
        t0 = perf_counter()
        probs, cache = forward(X, p)
        f_times.append(perf_counter() - t0)
        t0 = perf_counter()
        backward(Y, p, cache)
        b_times.append(perf_counter() - t0)
    return float(np.median(f_times)), float(np.median(b_times))


def main():
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target
    Y = one_hot(y)
    X_train, X_test, y_train, y_test, Y_train, Y_test = train_test_split(
        X, y, Y, test_size=0.30, random_state=SEED, stratify=y
    )
    print("Dataset shapes:", X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    # Shape and probability sanity check; near-uniform initial predictions yield ln(10).
    p0 = init_params(width=16, seed=SEED)
    probs0, cache0 = forward(X_train[:32], p0)
    print("Initial output shape:", probs0.shape)
    print("Max row-sum deviation:", np.max(np.abs(probs0.sum(axis=1) - 1)))
    print("Untrained loss:", cross_entropy(probs0, Y_train[:32]), "(expected about", np.log(10), ")")

    checks = gradient_check()
    print("\\nGradient check (10 random weights):")
    for k, idx, a, n, rel in checks:
        print(f"{k}{idx}: analytic={a:.8e}, numerical={n:.8e}, relative_error={rel:.3e}")
    passed = sum(c[-1] < 1e-6 for c in checks)
    print(f"Gradient check passed: {passed}/10 entries have relative error < 1e-6")
    with open(OUT / "gradient_check.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "index", "analytic", "numerical", "relative_error"])
        w.writerows(checks)

    configs = [(lr, width) for lr in (0.001, 0.01, 0.1) for width in (16, 64)]
    all_histories, summary = {}, []
    for lr, width in configs:
        p, hist = train(X_train, Y_train, X_test, y_test, width, lr, seed=SEED)
        all_histories[(lr, width)] = hist
        summary.append({
            "learning_rate": lr,
            "hidden_width": width,
            "final_train_loss": hist[-1][1],
            "final_train_accuracy": hist[-1][2],
            "final_test_accuracy": hist[-1][3],
        })

    with open(OUT / "training_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for lr, width in configs:
        hist = all_histories[(lr, width)]
        label = f"width={width}, lr={lr:g}"
        axes[0].plot([r[0] for r in hist], [r[1] for r in hist], label=label)
        axes[1].plot([r[0] for r in hist], [r[3] for r in hist], label=label)
    axes[0].set_title("Training cross-entropy")
    axes[1].set_title("Test accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(OUT / "loss_curves.png", dpi=160)
    plt.close(fig)

    # Time the two passes separately on one representative mini-batch.
    p_time = init_params(width=64, seed=SEED)
    Xb, Yb = X_train[:32], Y_train[:32]
    f_med, b_med = timing(Xb, Yb, p_time, runs=100)
    with open(OUT / "timing.txt", "w") as f:
        f.write(f"Median forward time (100 runs): {f_med*1e3:.6f} ms\\n")
        f.write(f"Median backward time (100 runs): {b_med*1e3:.6f} ms\\n")
        f.write("Timing uses a width-64 network and batch size 32.\\n")
    print("\\nFinal results:")
    for row in summary:
        print(row)
    print(f"Timing medians: forward={f_med*1e3:.4f} ms, backward={b_med*1e3:.4f} ms")
    print(f"Files written to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
