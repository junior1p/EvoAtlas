"""EvoAtlas Layer 2: Site-wise dN/dS via Felsenstein pruning + Layer 3 population genetics."""
from scipy.linalg import expm as _expm
import numpy as np
from collections import Counter
from .phylogeny import hky85_Q_matrix, TreeNode

GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def felsenstein_pruning(root: TreeNode, site_pattern: dict,
                        Q: np.ndarray, pi: np.ndarray,
                        states: list) -> float:
    """Compute log-likelihood of an alignment column given a tree and substitution model."""
    K = len(states)
    state_idx = {s: i for i, s in enumerate(states)}
    P_cache = {}

    def get_P(t: float):
        t = round(t, 8)
        if t not in P_cache:
            P_cache[t] = _expm(Q * max(t, 1e-10))
        return P_cache[t]

    partials = {}
    for node in root.post_order():
        if node.is_leaf():
            L = np.zeros(K)
            obs = site_pattern.get(node.name, None)
            if obs and obs.upper() in state_idx:
                L[state_idx[obs.upper()]] = 1.0
            else:
                L[:] = 1.0
            partials[id(node)] = L
        else:
            L = np.ones(K)
            for child in node.children:
                L *= get_P(child.branch_length) @ partials[id(child)]
            partials[id(node)] = L

    return np.log(max(np.dot(pi, partials[id(root)]), 1e-300))


def _compute_protein_omega(msa_matrix: np.ndarray, seq_ids: list, _) -> np.ndarray:
    """Entropy-based variability proxy for protein alignments (fast mode)."""
    from scipy.special import comb as scipy_comb
    n_seqs, align_len = msa_matrix.shape
    aa_states = list("ACDEFGHIKLMNPQRSTVWY")
    K = len(aa_states)
    omega = np.ones(align_len)
    for col in range(align_len):
        col_data = [c for c in msa_matrix[:, col] if c not in ("-", "X", "?")]
        if len(col_data) < 2:
            omega[col] = 0.0
            continue
        cnt = Counter(col_data)
        freqs = np.array(list(cnt.values()), dtype=float)
        freqs /= freqs.sum()
        H = -np.sum(freqs * np.log(freqs + 1e-300))
        max_H = np.log(K)
        omega[col] = H / max_H if max_H > 0 else 0.0
    return omega


def compute_site_omega_fast(msa_matrix: np.ndarray, seq_ids: list,
                            _) -> np.ndarray:
    """Fast mode: entropy proxy for ω (no tree required)."""
    return _compute_protein_omega(msa_matrix, seq_ids, None)


def compute_nucleotide_diversity(msa_matrix: np.ndarray) -> np.ndarray:
    """π: mean pairwise differences per site."""
    from scipy.special import comb as scipy_comb
    n_seqs, align_len = msa_matrix.shape
    pi_per_site = np.zeros(align_len)
    for col in range(align_len):
        col_data = [c for c in msa_matrix[:, col] if c not in ("-", "N", "?")]
        if len(col_data) < 2:
            continue
        n = len(col_data)
        n_pairs = int(n * (n - 1) / 2)
        diffs = sum(1 for i in range(n) for j in range(i + 1, n)
                    if col_data[i] != col_data[j])
        pi_per_site[col] = diffs / n_pairs if n_pairs > 0 else 0.0
    return pi_per_site


def compute_tajimas_d(msa_matrix: np.ndarray, window: int = 50,
                      step: int = 10) -> tuple:
    """Tajima's D in sliding windows."""
    from scipy.special import comb as scipy_comb
    n_seqs, align_len = msa_matrix.shape
    n = n_seqs
    if n < 4:
        return np.array([align_len // 2]), np.array([0.0])
    a1 = sum(1.0 / i for i in range(1, n))
    a2 = sum(1.0 / i**2 for i in range(1, n))
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n**2 + n + 3) / (9 * n * (n - 1))
    c1 = b1 - 1.0 / a1
    c2 = b2 - (n + 2) / (a1 * n) + a2 / a1**2
    positions, D_values = [], []

    for start in range(0, align_len - window, step):
        end = start + window
        w_msa = msa_matrix[:, start:end]
        S, pi_sum, n_valid = 0, 0.0, 0
        for col in range(w_msa.shape[1]):
            col_data = [c for c in w_msa[:, col] if c not in ("-", "N", "?")]
            if len(col_data) < 2:
                continue
            n_valid += 1
            cnt = Counter(col_data)
            if len(cnt) > 1:
                S += 1
            n_c = len(col_data)
            diffs = sum(1 for i in range(n_c) for j in range(i + 1, n_c)
                        if col_data[i] != col_data[j])
            pi_sum += diffs / (n_c * (n_c - 1) / 2) if n_c > 1 else 0.0
        if n_valid == 0 or S == 0:
            positions.append(start + window // 2)
            D_values.append(0.0)
            continue
        theta_pi = pi_sum / n_valid
        theta_W = S / a1 / n_valid
        e1 = c1 / a1
        e2 = c2 / (a1**2 + a2)
        var_D = max(e1 * S + e2 * S * (S - 1), 1e-10)
        positions.append(start + window // 2)
        D_values.append((theta_pi - theta_W) / np.sqrt(var_D))
    return np.array(positions), np.array(D_values)


def compute_fu_li_F(msa_matrix: np.ndarray) -> float:
    """Fu & Li's F* statistic."""
    n_seqs, align_len = msa_matrix.shape
    n = n_seqs
    if n < 4:
        return 0.0
    a1 = sum(1.0 / i for i in range(1, n))
    a2 = sum(1.0 / i**2 for i in range(1, n))
    S, eta_s, pi_total = 0, 0, 0.0
    for col in range(align_len):
        col_data = [c for c in msa_matrix[:, col] if c not in ("-", "N", "?")]
        if len(col_data) < 2:
            continue
        cnt = Counter(col_data)
        if len(cnt) <= 1:
            continue
        S += 1
        n_c = len(col_data)
        diffs = sum(1 for i in range(n_c) for j in range(i + 1, n_c)
                    if col_data[i] != col_data[j])
        pi_total += diffs / (n_c * (n_c - 1) / 2) if n_c > 1 else 0.0
        if min(cnt.values()) == 1:
            eta_s += 1
    if S == 0:
        return 0.0
    cn = 2 * n * a1 - 4 * (n - 1)
    vn = ((cn / ((n - 1) * (n - 2))) * S +
          (cn**2 + 4 * a2 - (4 * a2 + 2) * n / (n - 1)) /
          (a1**2 + a2) * S * (S - 1))
    return float((pi_total / align_len - eta_s / a1 / align_len) / np.sqrt(max(vn, 1e-10)))
