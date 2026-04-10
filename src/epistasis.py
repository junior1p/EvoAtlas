"""EvoAtlas Layer 4: Mutual Information + Walsh-Hadamard epistasis decomposition."""
import numpy as np
from collections import Counter


def compute_mutual_information_matrix(msa_matrix: np.ndarray,
                                      max_sites: int = 150,
                                      min_entropy: float = 0.1) -> tuple:
    """All-pairs NMI between alignment positions."""
    n_seqs, align_len = msa_matrix.shape
    entropies = np.zeros(align_len)
    for col in range(align_len):
        col_data = [c for c in msa_matrix[:, col] if c not in ("-", "N", "?")]
        if not col_data:
            continue
        cnt = Counter(col_data)
        freqs = np.array(list(cnt.values()), dtype=float)
        freqs /= freqs.sum()
        entropies[col] = -np.sum(freqs * np.log(freqs + 1e-300))

    variable_sites = np.where(entropies > min_entropy)[0]
    if len(variable_sites) > max_sites:
        top_idx = np.argsort(entropies[variable_sites])[::-1][:max_sites]
        variable_sites = np.sort(variable_sites[top_idx])

    n_sites = len(variable_sites)
    if n_sites < 2:
        return variable_sites, np.zeros((n_sites, n_sites))

    print(f"MI: {n_sites} variable sites, {n_sites*(n_sites-1)//2} pairs...")
    MI_matrix = np.zeros((n_sites, n_sites))

    for ii, i in enumerate(variable_sites):
        col_i = [c for c in msa_matrix[:, i] if c not in ("-", "N", "?")]
        cnt_i = Counter(col_i)
        H_i = -sum((v / len(col_i)) * np.log(v / len(col_i))
                    for v in cnt_i.values())
        for jj, j in enumerate(variable_sites):
            if jj <= ii:
                continue
            col_j = [c for c in msa_matrix[:, j] if c not in ("-", "N", "?")]
            n_pairs = len(col_j)
            cnt_j = Counter(col_j)
            H_j = -sum((v / n_pairs) * np.log(v / n_pairs)
                        for v in cnt_j.values())
            joint = Counter(zip(col_i, col_j))
            MI = 0.0
            for (x, y), c_xy in joint.items():
                p_xy = c_xy / n_pairs
                p_x = cnt_i.get(x, 0) / len(col_i)
                p_y = cnt_j.get(y, 0) / n_pairs
                if p_x > 0 and p_y > 0:
                    MI += p_xy * np.log(p_xy / (p_x * p_y))
            denom = np.sqrt(max(H_i * H_j, 1e-10))
            NMI = MI / denom
            MI_matrix[ii, jj] = MI_matrix[jj, ii] = NMI

    return variable_sites, MI_matrix


def fwht_numpy(a: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform, in-place."""
    a = a.astype(float).copy()
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x, y = a[j], a[j + h]
                a[j], a[j + h] = x + y, x - y
        h *= 2
    return a / n


def walsh_hadamard_epistasis(frequency_profile: np.ndarray) -> dict:
    """WHT decomposition into additive / pairwise / higher-order epistasis."""
    n = len(frequency_profile)
    import math
    L = max(1, math.ceil(math.log2(n + 1)))
    padded = np.zeros(2**L)
    padded[:n] = frequency_profile[:min(n, len(padded))]
    f_hat = fwht_numpy(padded)

    def hamming_weight(x: int) -> int:
        return bin(x).count("1")

    power = {0: 0.0, 1: 0.0, 2: 0.0}
    total = 0.0
    for idx in range(len(f_hat)):
        p = f_hat[idx]**2
        total += p
        w = hamming_weight(idx)
        if w <= 2:
            power[w] += p

    higher = max(0.0, total - sum(power.values()))
    total = max(total, 1e-10)
    return {
        "additive_fraction":  power[1] / total,
        "pairwise_fraction":  power[2] / total,
        "higher_fraction":    higher / total,
        "coefficients":       f_hat,
        "n_sites":            L,
    }
