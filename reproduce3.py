"""
Corrected paper-scale reproduction of Section 3.4 in:
Yang, Wainwright and Jordan (2016),
"On the Computational Complexity of High-Dimensional Bayesian Variable Selection".

Implemented experiments
-----------------------
1. Figure 1-style MCMC trajectories:
   n=500, p=1000, SNR in {1, 3}, 100 chains, 5p iterations.
2. Tables 1-2-style mixing experiment:
   (n,p) in {(500,1000), (500,5000), (1000,1000), (1000,5000)},
   independent/correlated designs, SNR in {0.5,1,2,3}, 20 datasets per
   setting, six chains per dataset, 20p iterations per chain, s0=100.
3. Figure 2-style Bayesian variable selection versus cross-validated Lasso:
   n=300, p=80, s*=20, 100 replicates.

Important
---------
This is an independent Python implementation, not the authors' original code.
The paper does not report every numerical implementation detail, especially
kappa and the exact MCMC settings for Figure 2, so those values are exposed as
command-line arguments. Full paper-scale runs are extremely expensive.

Dependencies
------------
pip install numpy pandas matplotlib scikit-learn

Examples
--------
python reproduce3.py --mode quick
python reproduce3.py --mode figure1
python reproduce3.py --mode paper-mixing
python reproduce3.py --mode figure2
python reproduce3.py --mode paper-all
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

Model = frozenset[int]


@dataclass
class PosteriorScorer:
    """Unnormalised log-posterior scorer based on supplementary equation (A.2)."""

    X: np.ndarray
    y: np.ndarray
    g: float
    kappa: float
    s0: int
    max_cache_size: int = 50_000

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.X.ndim != 2:
            raise ValueError("X must be two-dimensional.")
        self.n, self.p = self.X.shape
        if self.y.shape != (self.n,):
            raise ValueError("y must have shape (n,).")
        if self.g <= 0 or self.s0 < 0:
            raise ValueError("g must be positive and s0 nonnegative.")

        # Gram-based scoring is substantially faster than repeatedly calling lstsq.
        self.gram = self.X.T @ self.X
        self.xty = self.X.T @ self.y
        self.yty = float(self.y @ self.y)
        self.logp = math.log(self.p)
        self.log1pg = math.log1p(self.g)
        self.cache: OrderedDict[tuple[int, ...], tuple[float, float]] = OrderedDict()

    def evaluate(self, support: Iterable[int]) -> tuple[float, float]:
        key = tuple(sorted(int(j) for j in support))
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
            return cached

        k = len(key)
        if k > self.s0:
            return -math.inf, 0.0

        if k == 0:
            r2 = 0.0
        else:
            idx = np.fromiter(key, dtype=int)
            G = self.gram[np.ix_(idx, idx)]
            c = self.xty[idx]
            try:
                coef = np.linalg.solve(G, c)
            except np.linalg.LinAlgError:
                coef = np.linalg.lstsq(G, c, rcond=1e-10)[0]
            explained = float(c @ coef)
            r2 = min(max(explained / max(self.yty, 1e-300), 0.0), 1.0 - 1e-14)

        log_post = (
            -self.kappa * k * self.logp
            -0.5 * k * self.log1pg
            -0.5 * self.n * math.log1p(self.g * (1.0 - r2))
        )
        result = (float(log_post), float(r2))

        if self.max_cache_size > 0:
            self.cache[key] = result
            self.cache.move_to_end(key)
            if len(self.cache) > self.max_cache_size:
                self.cache.popitem(last=False)
        return result


def normalize_columns(X: np.ndarray) -> np.ndarray:
    """Scale each column to Euclidean norm sqrt(n), as assumed in the theory."""
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=0)
    norms[norms == 0] = 1.0
    return X * (math.sqrt(X.shape[0]) / norms)


def generate_design(
    n: int,
    p: int,
    kind: str,
    rng: np.random.Generator,
    normalize: bool = True,
) -> np.ndarray:
    """Generate independent or Sigma_jk=exp(-|j-k|) correlated Gaussian rows."""
    if n <= 0 or p <= 0:
        raise ValueError("n and p must be positive.")

    if kind == "independent":
        X = rng.standard_normal((n, p))
    elif kind == "correlated":
        # AR(1) recursion with rho=e^{-1} gives Cov(X_j,X_k)=e^{-|j-k|}.
        rho = math.exp(-1.0)
        X = np.empty((n, p), dtype=float)
        X[:, 0] = rng.standard_normal(n)
        innovation_sd = math.sqrt(1.0 - rho * rho)
        for j in range(1, p):
            X[:, j] = rho * X[:, j - 1] + innovation_sd * rng.standard_normal(n)
    else:
        raise ValueError("kind must be 'independent' or 'correlated'.")

    return normalize_columns(X) if normalize else X


def paper_beta(p: int, n: int, snr: float, sigma: float = 1.0) -> np.ndarray:
    """Construct beta* from Section 3.4.1; its true sparsity is ten."""
    if p < 10:
        raise ValueError("p must be at least 10.")
    pattern = np.array([2, -3, 2, 2, -3, 3, -2, 3, -2, 3], dtype=float)
    beta = np.zeros(p, dtype=float)
    beta[:10] = snr * math.sqrt(sigma * sigma * math.log(p) / n) * pattern
    return beta


def generate_linear_data(
    n: int,
    p: int,
    snr: float,
    design: str,
    rng: np.random.Generator,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Model]:
    X = generate_design(n, p, design, rng)
    beta = paper_beta(p, n, snr, sigma)
    y = X @ beta + rng.normal(scale=sigma, size=n)
    return X, y, beta, frozenset(range(10))


def perturb_null(p: int, rng: np.random.Generator, size: int = 5) -> Model:
    return frozenset(rng.choice(p, size=min(size, p), replace=False).tolist())


def mh_chain(
    scorer: PosteriorScorer,
    initial: Model,
    steps: int,
    rng: np.random.Generator,
    record_stride: int = 1,
    collect_models: bool = False,
) -> dict[str, object]:
    """Run the paper's 50%-single-flip / 50%-double-flip MH chain."""
    if steps < 1 or record_stride < 1:
        raise ValueError("steps and record_stride must be positive.")
    if len(initial) > scorer.s0:
        raise ValueError("initial model exceeds s0.")

    current = initial
    current_log, current_r2 = scorer.evaluate(current)
    times: list[int] = []
    logs: list[float] = []
    r2s: list[float] = []
    models: list[tuple[int, ...]] = []
    best_log = current_log
    best_model = current
    accepted = 0

    for t in range(steps + 1):
        if t % record_stride == 0:
            times.append(t)
            logs.append(current_log)
            r2s.append(current_r2)
            if collect_models:
                models.append(tuple(sorted(current)))
        if t == steps:
            break

        if rng.random() < 0.5:
            # Single flip.
            j = int(rng.integers(scorer.p))
            proposal_set = set(current)
            if j in proposal_set:
                proposal_set.remove(j)
            elif len(proposal_set) < scorer.s0:
                proposal_set.add(j)
            proposal = frozenset(proposal_set)
        else:
            # Double flip: remove one active and add one inactive variable.
            if not current or len(current) == scorer.p:
                proposal = current
            else:
                selected = tuple(current)
                remove_j = selected[int(rng.integers(len(selected)))]
                add_j = int(rng.integers(scorer.p))
                while add_j in current:
                    add_j = int(rng.integers(scorer.p))
                proposal_set = set(current)
                proposal_set.remove(remove_j)
                proposal_set.add(add_j)
                proposal = frozenset(proposal_set)

        proposal_log, proposal_r2 = scorer.evaluate(proposal)
        if math.log(rng.random()) < min(0.0, proposal_log - current_log):
            current = proposal
            current_log = proposal_log
            current_r2 = proposal_r2
            accepted += 1
            if current_log > best_log:
                best_log = current_log
                best_model = current

    return {
        "times": np.asarray(times),
        "log_post": np.asarray(logs),
        "r2": np.asarray(r2s),
        "models": models,
        "best_model": tuple(sorted(best_model)),
        "best_log_post": float(best_log),
        "acceptance_rate": accepted / steps,
    }


def gelman_rubin(chains: list[np.ndarray]) -> float:
    """Classical Gelman-Rubin potential scale reduction factor."""
    m = len(chains)
    n = min(len(x) for x in chains)
    if m < 2 or n < 2:
        return float("nan")
    arr = np.vstack([np.asarray(x[-n:], dtype=float) for x in chains])
    means = arr.mean(axis=1)
    W = arr.var(axis=1, ddof=1).mean()
    B = n * means.var(ddof=1)
    if W <= 1e-15:
        return 1.0 if B <= 1e-15 else float("inf")
    var_hat = (n - 1) * W / n + B / n
    return float(math.sqrt(var_hat / W))


def make_figure1(
    output_dir: Path,
    seed: int,
    kappa: float,
    chains: int = 100,
    cache_size: int = 50_000,
) -> dict[str, object]:
    """Paper Figure 1 scale: n=500, p=1000, 100 chains and 5p iterations."""
    n, p, s0 = 500, 1000, 100
    results: dict[str, object] = {}

    for snr in (1.0, 3.0):
        data_rng = np.random.default_rng(seed + int(1000 * snr))
        X, y, _, true_model = generate_linear_data(n, p, snr, "independent", data_rng)
        scorer = PosteriorScorer(X, y, float(p**3), kappa, s0, cache_size)
        true_log, _ = scorer.evaluate(true_model)
        null_log, _ = scorer.evaluate(frozenset())

        trajectories = []
        highest = -math.inf
        acceptance = []
        for chain_id in range(chains):
            chain_rng = np.random.default_rng(seed + int(10_000 * snr) + chain_id)
            if chain_id < chains // 2:
                initial = perturb_null(p, chain_rng, size=50,) ; Pylance: initial = true_model
            out = mh_chain(
                scorer,
                initial,
                steps=5 * p,
                rng=chain_rng,
                record_stride=max(1, p // 100),
            )
            trajectories.append(out)
            highest = max(highest, float(out["best_log_post"]))
            acceptance.append(float(out["acceptance_rate"]))
            if (chain_id + 1) % 10 == 0:
                print(f"Figure 1, SNR={snr:g}: {chain_id + 1}/{chains} chains")

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for out in trajectories:
            ax.plot(out["times"] / p, out["log_post"], alpha=0.25, linewidth=0.8)
        # Reference levels requested for Figure 1.
        ax.axhline(
            y=true_log,
            color="green",
            linestyle="--",
            linewidth=1.5,
            label="true model",
        )
        ax.axhline(
            y=null_log,
            color="darkblue",
            linestyle="--",
            linewidth=1.5,
            label="null model",
        )
        ax.axhline(
            y=highest,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="highest probability model",
        )
        ax.set_xlabel("iterations / p")
        ax.set_ylabel("unnormalised log posterior")
        ax.set_ylim(-6200, -5000)
        ax.set_title(f"Independent design, SNR={snr:g}")
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"figure1_snr{int(snr)}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)

        results[f"snr_{snr:g}"] = {
            "figure": str(path),
            "mean_acceptance_rate": float(np.mean(acceptance)),
            "true_log_post": true_log,
            "null_log_post": null_log,
            "highest_found": highest,
        }
    return results


def paper_mixing_experiment(
    output_dir: Path,
    seed: int,
    kappa: float,
    replicates: int = 20,
    steps_per_p: int = 20,
    record_stride: int = 1,
    cache_size: int = 50_000,
    resume: bool = True,
) -> pd.DataFrame:
    """Paper Tables 1-2 scale; saves after every dataset and supports resuming."""
    settings = [(500, 1000), (500, 5000), (1000, 1000), (1000, 5000)]
    designs = ("independent", "correlated")
    snrs = (0.5, 1.0, 2.0, 3.0)
    raw_path = output_dir / "paper_mixing_raw.csv"
    summary_path = output_dir / "paper_mixing_summary.csv"

    if resume and raw_path.exists():
        old = pd.read_csv(raw_path)
        rows = old.to_dict("records")
        completed = {
            (str(r["design"]), int(r["n"]), int(r["p"]), float(r["SNR"]), int(r["replicate"]))
            for r in rows
        }
        print(f"Resuming from {len(rows)} completed datasets.")
    else:
        rows = []
        completed = set()

    total = len(settings) * len(designs) * len(snrs) * replicates
    started = time.time()

    for design_id, design in enumerate(designs):
        for setting_id, (n, p) in enumerate(settings):
            for snr_id, snr in enumerate(snrs):
                for rep in range(1, replicates + 1):
                    key = (design, n, p, float(snr), rep)
                    if key in completed:
                        continue

                    data_seed = seed + design_id * 1_000_000 + setting_id * 100_000 + snr_id * 10_000 + rep
                    rng = np.random.default_rng(data_seed)
                    print(
                        f"\nDataset {len(rows)+1}/{total}: design={design}, n={n}, "
                        f"p={p}, SNR={snr:g}, replicate={rep}"
                    )

                    X, y, _, true_model = generate_linear_data(n, p, snr, design, rng)
                    scorer = PosteriorScorer(X, y, float(p**3), kappa, 100, cache_size)
                    true_log, _ = scorer.evaluate(true_model)
                    null_log, _ = scorer.evaluate(frozenset())

                    r2_chains: list[np.ndarray] = []
                    highest = -math.inf
                    acceptance = []
                    for chain_id in range(6):
                        chain_rng = np.random.default_rng(data_seed + 50_000 + chain_id)
                        initial = perturb_null(p, chain_rng, size=5) if chain_id < 3 else true_model
                        out = mh_chain(
                            scorer,
                            initial,
                            steps=steps_per_p * p,
                            rng=chain_rng,
                            record_stride=record_stride,
                        )
                        r2_values = np.asarray(out["r2"])
                        r2_chains.append(r2_values[len(r2_values)//2:])
                        highest = max(highest, float(out["best_log_post"]))
                        acceptance.append(float(out["acceptance_rate"]))
                        print(f"  chain {chain_id+1}/6, acceptance={acceptance[-1]:.4f}")

                    rhat_value = gelman_rubin(r2_chains)
                    rows.append({
                        "design": design,
                        "n": n,
                        "p": p,
                        "SNR": snr,
                        "replicate": rep,
                        "Rhat": rhat_value,
                        "success": bool(rhat_value <= 1.5),
                        "H_minus_T": highest - true_log,
                        "N_minus_T": null_log - true_log,
                        "mean_acceptance": float(np.mean(acceptance)),
                    })
                    pd.DataFrame(rows).to_csv(raw_path, index=False)
                    elapsed = (time.time() - started) / 3600
                    print(f"  Rhat={rhat_value:.4f}; elapsed={elapsed:.2f} hours")
                    del scorer, X, y

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["design", "n", "p", "SNR"], as_index=False)
        .agg(
            SP_percent=("success", lambda x: 100.0 * np.mean(x)),
            median_Rhat=("Rhat", "median"),
            H_minus_T=("H_minus_T", "mean"),
            N_minus_T=("N_minus_T", "mean"),
            mean_acceptance=("mean_acceptance", "mean"),
        )
    )
    summary.to_csv(summary_path, index=False)
    return summary


def sigma_bad(p: int) -> np.ndarray:
    """Covariance matrix used in Section 3.4.2."""
    mu = 1.0 / (2.0 * math.sqrt(p))
    S = np.eye(p)
    S[0, 1:] = mu
    S[1:, 0] = mu
    eig_min = float(np.linalg.eigvalsh(S)[0])
    if eig_min <= 0:
        S += (abs(eig_min) + 1e-8) * np.eye(p)
    return S


def bvs_median_probability_model(
    X: np.ndarray,
    y: np.ndarray,
    true_model: Model,
    seed: int,
    kappa: float,
    steps: int,
    cache_size: int,
) -> tuple[Model, PosteriorScorer]:
    """Approximate the median probability model using two dispersed chains."""
    p = X.shape[1]
    scorer = PosteriorScorer(X, y, float(p**3), kappa, p, cache_size)
    pip_counts = np.zeros(p, dtype=float)
    kept = 0

    perturbed = set(true_model)
    rng0 = np.random.default_rng(seed + 17)
    active = list(perturbed)
    inactive = list(set(range(p)) - perturbed)
    swaps = min(8, len(active), len(inactive))
    if swaps > 0:
        removed = rng0.choice(active, size=swaps, replace=False)
        added = rng0.choice(inactive, size=swaps, replace=False)
        perturbed.difference_update(int(x) for x in removed)
        perturbed.update(int(x) for x in added)

    initials = [frozenset(), frozenset(perturbed)]
    for chain_id, initial in enumerate(initials):
        rng = np.random.default_rng(seed + 100 * chain_id)
        out = mh_chain(
            scorer,
            initial,
            steps=steps,
            rng=rng,
            record_stride=5,
            collect_models=True,
        )
        models = out["models"]
        burn = len(models) // 2
        for model in models[burn:]:
            if model:
                pip_counts[list(model)] += 1.0
            kept += 1

    pips = pip_counts / max(kept, 1)
    return frozenset(np.flatnonzero(pips >= 0.5).tolist()), scorer


def make_figure2(
    output_dir: Path,
    seed: int,
    kappa: float,
    replicates: int = 100,
    bvs_steps: int = 7000,
    cache_size: int = 50_000,
) -> dict[str, object]:
    """Paper Figure 2 scale: n=300, p=80, s*=20 and 100 replicates."""
    n, p, s_star = 300, 80, 20
    covariance = sigma_bad(p)
    chol = np.linalg.cholesky(covariance)
    true_model = frozenset(range(1, s_star + 1))
    raw_path = output_dir / "figure2_log_posterior_differences.csv"

    bvs_diffs: list[float] = []
    lasso_diffs: list[float] = []
    bvs_sizes: list[int] = []
    lasso_sizes: list[int] = []

    for rep in range(replicates):
        rng = np.random.default_rng(seed + 30_000 + rep)
        X = rng.standard_normal((n, p)) @ chol.T
        X = normalize_columns(X)
        beta = np.zeros(p)
        beta[list(true_model)] = 1.0
        y = X @ beta + rng.standard_normal(n)

        bvs_model, scorer = bvs_median_probability_model(
            X,
            y,
            true_model,
            seed=seed + 40_000 + rep,
            kappa=kappa,
            steps=bvs_steps,
            cache_size=cache_size,
        )
        true_log, _ = scorer.evaluate(true_model)
        bvs_log, _ = scorer.evaluate(bvs_model)
        bvs_diffs.append(bvs_log - true_log)
        bvs_sizes.append(len(bvs_model))

        Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
        lasso = LassoCV(
            cv=5,
            alphas=np.logspace(-4, 1, 80),
            max_iter=20_000,
            random_state=seed + rep,
        ).fit(Xs, y)
        lasso_model = frozenset(np.flatnonzero(np.abs(lasso.coef_) > 1e-8).tolist())
        lasso_log, _ = scorer.evaluate(lasso_model)
        lasso_diffs.append(lasso_log - true_log)
        lasso_sizes.append(len(lasso_model))

        pd.DataFrame({"BVS": bvs_diffs, "Lasso": lasso_diffs}).to_csv(raw_path, index=False)
        if (rep + 1) % 5 == 0:
            print(f"Figure 2: {rep + 1}/{replicates} replicates")

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.boxplot([bvs_diffs, lasso_diffs], tick_labels=["BVS", "Lasso"], showfliers=True)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_ylabel("log posterior(selected) - log posterior(true)")
    ax.set_title("Bayesian variable selection versus Lasso")
    fig.tight_layout()
    path = output_dir / "figure2_bvs_vs_lasso.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return {
        "figure": str(path),
        "replicates": replicates,
        "bvs_median_difference": float(np.median(bvs_diffs)),
        "lasso_median_difference": float(np.median(lasso_diffs)),
        "bvs_exact_zero_fraction": float(np.mean(np.isclose(bvs_diffs, 0.0, atol=1e-8))),
        "mean_bvs_size": float(np.mean(bvs_sizes)),
        "mean_lasso_size": float(np.mean(lasso_sizes)),
    }


def quick_test(output_dir: Path, seed: int, kappa: float, cache_size: int) -> dict[str, object]:
    """Small installation test; this is not a paper-scale reproduction."""
    rng = np.random.default_rng(seed)
    n, p, snr = 120, 180, 2.0
    X, y, _, _ = generate_linear_data(n, p, snr, "correlated", rng)
    scorer = PosteriorScorer(X, y, float(p**3), kappa, 40, cache_size)
    out = mh_chain(
        scorer,
        frozenset(),
        steps=2 * p,
        rng=np.random.default_rng(seed + 1),
        record_stride=1,
    )
    path = output_dir / "quick_test.csv"
    pd.DataFrame({
        "iteration": out["times"],
        "log_posterior": out["log_post"],
        "r2": out["r2"],
    }).to_csv(path, index=False)
    return {
        "output": str(path),
        "acceptance_rate": float(out["acceptance_rate"]),
        "best_model_size": len(out["best_model"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("quick", "figure1", "paper-mixing", "figure2", "paper-all"),
        default="quick",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reproduce3_output"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--kappa",
        type=float,
        default=1.0,
        help=(
            "Sparsity-prior exponent. Section 3.4 explicitly reports g=p^3 and s0=100, "
            "but not every implementation detail for kappa, so it is configurable."
        ),
    )
    parser.add_argument("--cache-size", type=int, default=50_000)
    parser.add_argument("--record-stride", type=int, default=1)
    parser.add_argument("--mixing-replicates", type=int, default=20)
    parser.add_argument("--steps-per-p", type=int, default=20)
    parser.add_argument("--figure1-chains", type=int, default=100)
    parser.add_argument("--figure2-replicates", type=int, default=100)
    parser.add_argument("--figure2-bvs-steps", type=int, default=7000)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    summary: dict[str, object] = {
        "mode": args.mode,
        "seed": args.seed,
        "kappa": args.kappa,
    }

    if args.mode == "quick":
        summary["quick"] = quick_test(args.output_dir, args.seed, args.kappa, args.cache_size)

    elif args.mode == "figure1":
        summary["figure1"] = make_figure1(
            args.output_dir,
            args.seed,
            args.kappa,
            chains=args.figure1_chains,
            cache_size=args.cache_size,
        )

    elif args.mode == "paper-mixing":
        table = paper_mixing_experiment(
            args.output_dir,
            args.seed,
            args.kappa,
            replicates=args.mixing_replicates,
            steps_per_p=args.steps_per_p,
            record_stride=args.record_stride,
            cache_size=args.cache_size,
            resume=not args.no_resume,
        )
        summary["paper_mixing_rows"] = len(table)

    elif args.mode == "figure2":
        summary["figure2"] = make_figure2(
            args.output_dir,
            args.seed,
            args.kappa,
            replicates=args.figure2_replicates,
            bvs_steps=args.figure2_bvs_steps,
            cache_size=args.cache_size,
        )

    elif args.mode == "paper-all":
        summary["figure1"] = make_figure1(
            args.output_dir,
            args.seed,
            args.kappa,
            chains=args.figure1_chains,
            cache_size=args.cache_size,
        )
        table = paper_mixing_experiment(
            args.output_dir,
            args.seed,
            args.kappa,
            replicates=args.mixing_replicates,
            steps_per_p=args.steps_per_p,
            record_stride=args.record_stride,
            cache_size=args.cache_size,
            resume=not args.no_resume,
        )
        summary["paper_mixing_rows"] = len(table)
        summary["figure2"] = make_figure2(
            args.output_dir,
            args.seed,
            args.kappa,
            replicates=args.figure2_replicates,
            bvs_steps=args.figure2_bvs_steps,
            cache_size=args.cache_size,
        )

    summary["elapsed_seconds"] = time.time() - started
    summary_path = args.output_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nFinished. Summary saved to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
