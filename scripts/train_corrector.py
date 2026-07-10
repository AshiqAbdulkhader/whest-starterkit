"""Train the learned residual-correction model on extracted features.

Reads the .npz from extract_features.py, trains a small numpy MLP to predict
the residual (truth - K2_prediction) per final-layer neuron, evaluates with
grouped (per-MLP) holdout, and saves the weights as corrector.npz for
shipping inside the submission (see estimator.py setup()).

The inference path must be exactly reproducible with flopscope.numpy at
predict() time: standardize -> Linear -> tanh -> Linear -> tanh -> Linear.

Usage:
    uv run python scripts/train_corrector.py --features features_full.npz --out corrector.npz
"""

from __future__ import annotations

import argparse

import numpy as np


def make_mlp(rng, sizes):
    params = []
    n = len(sizes) - 1
    for li, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        if li == n - 1:
            # zero-init output layer: initial prediction is exactly 0 residual,
            # i.e. the model starts from the K=2 baseline and can only improve.
            W = np.zeros((fan_in, fan_out))
        else:
            W = rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))
        b = np.zeros(fan_out)
        params.append([W, b])
    return params


def forward(params, X):
    h = X
    acts = [h]
    for i, (W, b) in enumerate(params):
        z = h @ W + b
        h = np.tanh(z) if i < len(params) - 1 else z
        acts.append(h)
    return h[:, 0], acts


def train(params, X, y, *, lr, epochs, batch, weight_decay, rng, X_va=None, y_va=None):
    n = len(X)
    m_adam = [[np.zeros_like(W), np.zeros_like(b)] for W, b in params]
    v_adam = [[np.zeros_like(W), np.zeros_like(b)] for W, b in params]
    t = 0
    best_va = np.inf
    best_params = None
    for ep in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            Xb, yb = X[idx], y[idx]
            # forward
            hs = [Xb]
            zs = []
            h = Xb
            for i, (W, b) in enumerate(params):
                z = h @ W + b
                zs.append(z)
                h = np.tanh(z) if i < len(params) - 1 else z
                hs.append(h)
            pred = h[:, 0]
            err = pred - yb  # dL/dpred, L = mean 0.5*err^2
            g = (err / len(idx))[:, None]
            # backward
            grads = [None] * len(params)
            for i in reversed(range(len(params))):
                W, b = params[i]
                gW = hs[i].T @ g + weight_decay * W
                gb = g.sum(axis=0)
                grads[i] = [gW, gb]
                if i > 0:
                    g = (g @ W.T) * (1 - np.tanh(zs[i - 1]) ** 2)
            # adam
            t += 1
            for i in range(len(params)):
                for j in range(2):
                    m_adam[i][j] = 0.9 * m_adam[i][j] + 0.1 * grads[i][j]
                    v_adam[i][j] = 0.999 * v_adam[i][j] + 0.001 * grads[i][j] ** 2
                    mh = m_adam[i][j] / (1 - 0.9**t)
                    vh = v_adam[i][j] / (1 - 0.999**t)
                    params[i][j] -= lr * mh / (np.sqrt(vh) + 1e-8)
        if X_va is not None:
            pv, _ = forward(params, X_va)
            va = float(np.mean((pv - y_va) ** 2))
            if va < best_va:
                best_va = va
                best_params = [[W.copy(), b.copy()] for W, b in params]
            if (ep + 1) % 10 == 0:
                pt, _ = forward(params, X)
                tr = float(np.mean((pt - y) ** 2))
                print(f"epoch {ep+1}: train {tr:.4e}  val {va:.4e}  best {best_va:.4e}")
    return (best_params or params), best_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", choices=["k2", "mc"], default="mc",
                    help="which estimate the model corrects from")
    args = ap.parse_args()

    data = np.load(args.features)
    X = data["feats"].astype(np.float64)
    base = data["mc"] if args.base == "mc" and "mc" in data else data["base"]
    truth = data["truth"]
    mlp_id = data["mlp_id"]
    y = truth - base  # residual target

    # grouped holdout by MLP id
    rng = np.random.default_rng(args.seed)
    ids = np.unique(mlp_id)
    rng.shuffle(ids)
    n_hold = max(1, int(len(ids) * args.holdout_frac))
    hold_ids = set(ids[:n_hold].tolist())
    va_mask = np.isin(mlp_id, list(hold_ids))
    tr_mask = ~va_mask

    mu_f = X[tr_mask].mean(axis=0)
    sd_f = X[tr_mask].std(axis=0) + 1e-12
    Xn = (X - mu_f) / sd_f
    # standardize target so the net optimizes at O(1) scale
    y_sd = float(y[tr_mask].std()) + 1e-18
    yn = y / y_sd

    print(f"train rows {tr_mask.sum()}, val rows {va_mask.sum()}, features {X.shape[1]}")
    base_mse_va = float(np.mean((truth[va_mask] - base[va_mask]) ** 2))
    print(f"holdout baseline (K=2 alone) final-layer MSE: {base_mse_va:.4e}")

    # ridge-regression sanity baseline
    lam = 1e-3
    A = Xn[tr_mask]
    coef = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr_mask])
    ridge_res = Xn[va_mask] @ coef
    mse_ridge = float(np.mean((truth[va_mask] - base[va_mask] - ridge_res) ** 2))
    print(f"holdout ridge-corrected MSE: {mse_ridge:.4e} "
          f"({base_mse_va/mse_ridge:.1f}x vs K=2 alone)")

    sizes = [X.shape[1]] + [args.hidden] * args.layers + [1]
    params = make_mlp(rng, sizes)
    params, _ = train(
        params, Xn[tr_mask], yn[tr_mask],
        lr=args.lr, epochs=args.epochs, batch=args.batch,
        weight_decay=args.weight_decay, rng=rng,
        X_va=Xn[va_mask], y_va=yn[va_mask],
    )

    pred_res, _ = forward(params, Xn[va_mask])
    corrected = base[va_mask] + pred_res * y_sd
    mse_corr = float(np.mean((truth[va_mask] - corrected) ** 2))
    print(f"holdout MLP-corrected final-layer MSE: {mse_corr:.4e} "
          f"({base_mse_va/mse_corr:.1f}x better than K=2 alone)")

    # save flat arrays for flopscope.Module (pickle-free npz)
    save = {
        "feat_mu": mu_f,
        "feat_sd": sd_f,
        "y_sd": np.asarray(y_sd),
        "n_layers": np.asarray(len(params)),
    }
    for i, (W, b) in enumerate(params):
        save[f"W{i}"] = W
        save[f"b{i}"] = b
    np.savez(args.out, **save)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
