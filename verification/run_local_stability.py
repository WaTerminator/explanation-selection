"""Finite-radius stability and rate experiments (Appendix C.11)."""
from __future__ import annotations
import csv
import warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
SEED = 20260830
ENDPOINT_LOG_GAP = 10.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -x))


def posterior_batch(
    y: np.ndarray, codes: np.ndarray, priors: np.ndarray, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    logits = np.log(priors)[None, :] / tau + np.sum(
        -np.logaddexp(0.0, -codes[None, :, :] * y[:, None, :]), axis=2
    ) / tau
    logits -= np.max(logits, axis=1, keepdims=True)
    q = np.exp(logits)
    q /= np.sum(q, axis=1, keepdims=True)
    return q, logits


def gap_resolved_endpoint(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the provisional gap label evaluated at one explicit state."""
    order = np.argsort(logits, axis=1)
    rows = np.arange(len(logits))
    gap = logits[rows, order[:, -1]] - logits[rows, order[:, -2]]
    endpoint = np.where(gap > ENDPOINT_LOG_GAP, order[:, -1], -1)
    return endpoint, gap


def potential_batch(
    y: np.ndarray, codes: np.ndarray, priors: np.ndarray, tau: float
) -> np.ndarray:
    terms = np.log(priors)[None, :] / tau + np.sum(
        -np.logaddexp(0.0, -codes[None, :, :] * y[:, None, :]), axis=2
    ) / tau
    return 2.0 * tau * logsumexp(terms, axis=1)


def random_spd(n: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigenvalues = np.linspace(0.35, 1.0, n)
    rng.shuffle(eigenvalues)
    return (q * eigenvalues) @ q.T


def random_candidate_set(
    n: int, code_count: int, rng: np.random.Generator
) -> np.ndarray:
    if code_count > 2**n:
        raise ValueError(f"code_count={code_count} exceeds 2**n={2**n}")
    # The all-positive anchor is canonical: coordinatewise sign flips reduce
    # every lambda=0 anchor to this representation without changing defects.
    sampled = rng.choice(2**n - 1, size=code_count - 1, replace=False)
    bits = ((sampled[:, None] >> np.arange(n)) & 1).astype(float)
    rivals = 2.0 * bits - 1.0
    return np.vstack([np.ones(n), rivals])


def critical_functional(
    codes: np.ndarray,
    priors: np.ndarray,
    anchor: int,
    tau: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    z = codes[anchor]
    terms: list[tuple[np.ndarray, float]] = []
    for j, w in enumerate(codes):
        defects = np.flatnonzero(w != z)
        if j != anchor and len(defects) == tau:
            coefficient = tau * (priors[j] / priors[anchor]) ** (1.0 / tau)
            terms.append((defects, coefficient))
    if not terms:
        raise ValueError("critical_functional requires a rival at distance tau")

    eps = 1e-12

    def value(p: np.ndarray) -> float:
        return float(
            sum(
                coefficient * np.prod(p[defects] ** (1.0 / tau))
                for defects, coefficient in terms
            )
        )

    starts = [np.full(codes.shape[1], 1.0 / codes.shape[1])]
    for defects, _ in terms:
        p = np.full(codes.shape[1], eps)
        p[defects] = 1.0 / len(defects)
        starts.append(p / p.sum())
    starts.extend(rng.dirichlet(np.full(codes.shape[1], 0.3), size=12))

    best_value = -np.inf
    best_p: np.ndarray | None = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Values in x were outside bounds")
        for start in starts:
            result = minimize(
                lambda p: -value(p),
                start,
                method="SLSQP",
                bounds=[(eps, 1.0)] * codes.shape[1],
                constraints={"type": "eq", "fun": lambda p: p.sum() - 1.0},
                options={"ftol": 1e-12, "maxiter": 1500},
            )
            if result.success and -result.fun > best_value:
                best_value = float(-result.fun)
                best_p = result.x / result.x.sum()
    if best_p is None:
        raise RuntimeError("SLSQP failed on every chi_z start")
    return best_value, best_p


def nearest_defect_direction(codes: np.ndarray) -> np.ndarray:
    distances = np.sum(codes[1:] != codes[0], axis=1)
    defects = np.flatnonzero(codes[1 + int(np.argmin(distances))] != codes[0])
    p = np.zeros(codes.shape[1])
    p[defects] = 1.0 / len(defects)
    return p


def initialize_from_compactified(
    anchor: np.ndarray, radius: float, direction: np.ndarray
) -> np.ndarray:
    p = np.maximum(direction, 1e-10)
    p /= p.sum()
    return anchor * (-np.log(radius * p))


def simulate_local_batch(
    initial_y: np.ndarray,
    grams: np.ndarray,
    radii: np.ndarray,
    codes: np.ndarray,
    priors: np.ndarray,
    tau: float,
    eta: float,
    steps: int,
) -> list[dict[str, float | int | str]]:
    y = initial_y.copy()
    anchor = codes[0]
    labels = (codes + 1.0) / 2.0
    maximum_radius = radii.copy()
    midpoint_radius = np.full(len(y), np.nan)
    selected = np.full(len(y), -1, dtype=int)
    active = np.ones(len(y), dtype=bool)
    escape_radius = 4.0 * radii

    q, logits = posterior_batch(y, codes, priors, tau)
    for step in range(1, steps + 1):
        targets = q @ labels
        residual = 2.0 * (targets - sigmoid(y))
        update = eta * np.einsum("bij,bj->bi", grams, residual)
        y[active] += update[active]
        q, logits = posterior_batch(y, codes, priors, tau)

        if step % 10 == 0 or step in (steps // 2, steps):
            current_radius = np.sum(
                np.exp(np.clip(-anchor[None, :] * y, -700.0, 50.0)), axis=1
            )
            maximum_radius = np.maximum(maximum_radius, current_radius)
            if step == steps // 2:
                midpoint_radius = current_radius.copy()

            provisional, _ = gap_resolved_endpoint(logits)
            newly_selected = active & (provisional > 0)
            selected[newly_selected] = provisional[newly_selected]
            escaped = active & (current_radius >= escape_radius)
            active[newly_selected | escaped] = False
            if not np.any(active) and step >= steps // 2:
                break

    q, logits = posterior_batch(y, codes, priors, tau)
    final_radius = np.sum(
        np.exp(np.clip(-anchor[None, :] * y, -700.0, 50.0)), axis=1
    )
    midpoint_radius[np.isnan(midpoint_radius)] = final_radius[np.isnan(midpoint_radius)]
    provisional, _ = gap_resolved_endpoint(logits)
    unresolved = (selected < 0) & (provisional >= 0)
    selected[unresolved] = provisional[unresolved]

    results = []
    for i in range(len(y)):
        escaped = bool(selected[i] > 0 or maximum_radius[i] >= escape_radius[i])
        contracting = bool(
            not escaped
            and q[i, 0] > 0.9
            and final_radius[i] < 0.995 * midpoint_radius[i]
            and maximum_radius[i] <= 1.5 * radii[i]
        )
        outcome = "escaped" if escaped else ("contracting" if contracting else "unclear")
        results.append(
            {
                "outcome": outcome,
                "selected_code": int(selected[i]),
                "final_radius": float(final_radius[i]),
                "midpoint_radius": float(midpoint_radius[i]),
                "maximum_radius": float(maximum_radius[i]),
                "anchor_posterior": float(q[i, 0]),
            }
        )
    return results


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty result file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_local_experiment(output: Path, quick: bool) -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    ns = [6, 8] if quick else [6, 8, 10, 12]
    code_counts = [4, 8] if quick else [4, 8, 16]
    repeats = 1 if quick else 15
    gram_count = 1 if quick else 3
    steps = 1200 if quick else 8000
    critical_targets = [0.8, 1.2] if quick else [0.6, 0.8, 0.95, 1.05, 1.2, 1.4]
    local_rows: list[dict[str, object]] = []
    case_index = 0

    for n in ns:
        for code_count in code_counts:
            for repeat in range(repeats):
                codes = random_candidate_set(n, code_count, rng)
                distances = np.sum(codes[1:] != codes[0], axis=1)
                d_min = int(np.min(distances))
                base_weights = np.exp(rng.normal(0.0, 0.35, code_count))
                base_priors = base_weights / base_weights.sum()
                chi_base, p_star = critical_functional(
                    codes, base_priors, 0, d_min, rng
                )
                critical_target = critical_targets[case_index % len(critical_targets)]
                critical_weights = base_weights.copy()
                critical_weights[0] *= (chi_base / critical_target) ** d_min
                critical_priors = critical_weights / critical_weights.sum()

                directions = {
                    "witness": nearest_defect_direction(codes),
                    "dense": rng.dirichlet(np.ones(n)),
                    "near_face": rng.dirichlet(np.full(n, 0.15)),
                }
                grams = [np.eye(n)] + [
                    random_spd(n, rng) for _ in range(gram_count - 1)
                ]
                regimes = [
                    ("strict_viable", max(0.5, d_min - 0.5), base_priors, np.nan, True),
                    ("critical", float(d_min), critical_priors, critical_target, critical_target < 1.0),
                    ("strict_nonviable", d_min + 0.5, base_priors, np.nan, False),
                ]

                for regime, tau, priors, chi_value, theory_stable in regimes:
                    regime_directions = directions.copy()
                    if regime == "critical":
                        regime_directions["witness"] = p_star
                    eta = 1.6 / (n / tau + 1.0)
                    metadata: list[tuple[int, str, float]] = []
                    initial_y = []
                    batch_grams = []
                    batch_radii = []
                    for gram_id, gram in enumerate(grams):
                        for direction_name, direction in regime_directions.items():
                            for radius in (0.1, 0.05, 0.01):
                                metadata.append((gram_id, direction_name, radius))
                                initial_y.append(
                                    initialize_from_compactified(codes[0], radius, direction)
                                )
                                batch_grams.append(gram)
                                batch_radii.append(radius)
                    initial_y_array = np.asarray(initial_y)
                    batch_radii_array = np.asarray(batch_radii)
                    initial_gap = potential_batch(
                        initial_y_array, codes, priors, tau
                    ) - 2.0 * np.log(priors[0])
                    results = simulate_local_batch(
                        initial_y_array,
                        np.asarray(batch_grams),
                        batch_radii_array,
                        codes,
                        priors,
                        tau,
                        eta,
                        steps,
                    )
                    for meta, gap_value, result in zip(metadata, initial_gap, results):
                        gram_id, direction_name, radius = meta
                        empirical_stable = result["outcome"] == "contracting"
                        # Non-viability asserts existence of an escaping direction,
                        # not escape from every nearby direction.  Only the analytic
                        # witness is therefore a claim-level test in that regime.
                        claim_test = theory_stable or direction_name == "witness"
                        local_rows.append(
                            {
                                "case_id": case_index,
                                "n": n,
                                "code_count": code_count,
                                "repeat": repeat,
                                "d_min": d_min,
                                "tau": tau,
                                "regime": regime,
                                "chi_z": chi_value,
                                "theory_stable": int(theory_stable),
                                "gram_id": gram_id,
                                "direction": direction_name,
                                "r0": radius,
                                "eta": eta,
                                "initial_potential_gap": float(gap_value),
                                **result,
                                "claim_test": int(claim_test),
                                "agreement": (
                                    int(empirical_stable == theory_stable) if claim_test else -1
                                ),
                            }
                        )
                case_index += 1
                print(
                    f"local case {case_index}: N={n}, |C|={code_count}, "
                    f"d_min={d_min}, target chi={critical_target:.2f}"
                )

    write_csv(output / "local_stability.csv", local_rows)
    claim_rows = [row for row in local_rows if row["claim_test"]]
    decisive = [row for row in claim_rows if row["outcome"] != "unclear"]
    agreement = sum(int(row["agreement"]) for row in decisive) / max(1, len(decisive))
    coverage = len(decisive) / len(claim_rows)
    return {
        "candidate_sets": case_index,
        "trajectories": len(local_rows),
        "claim_test_trajectories": len(claim_rows),
        "finite_horizon_steps": steps,
        "decisive_fraction": coverage,
        "agreement_on_decisive": agreement,
    }


def block_codes(m: int) -> np.ndarray:
    codes = np.ones((m + 1, 2 * m))
    for block in range(m):
        codes[block + 1, 2 * block:2 * block + 2] = -1.0
    return codes


def m2_codes() -> np.ndarray:
    return block_codes(2)


def run_rate_check(output: Path, quick: bool) -> dict[str, object]:
    codes = m2_codes()
    cases = [
        ("strict", 1.0, np.full(3, 1.0 / 3.0)),
        ("critical_chi_0.8", 2.0, np.array([1.0, 0.64, 0.64]) / 2.28),
    ]
    steps = 20000 if quick else 200000
    sample_times = np.unique(np.geomspace(1, steps, 320).astype(int))
    rows = []
    slopes = {}
    for name, tau, priors in cases:
        y = initialize_from_compactified(codes[0], 0.05, np.full(4, 0.25))[None, :]
        labels = (codes + 1.0) / 2.0
        samples = []
        sample_index = 0
        for step in range(1, steps + 1):
            q, _ = posterior_batch(y, codes, priors, tau)
            y += 0.05 * (2.0 * (q @ labels - sigmoid(y)))
            if sample_index < len(sample_times) and step == sample_times[sample_index]:
                radius = float(np.sum(np.exp(-codes[0] * y[0])))
                samples.append((step, radius))
                rows.append({"case": name, "step": step, "radius": radius})
                sample_index += 1
        samples_array = np.asarray(samples)
        tail = samples_array[:, 0] >= steps / 2
        slopes[name] = float(
            np.polyfit(
                np.log(samples_array[tail, 0]), np.log(samples_array[tail, 1]), 1
            )[0]
        )
    write_csv(output / "rate_traces.csv", rows)
    return {"steps": steps, "tail_log_log_slopes": slopes}
