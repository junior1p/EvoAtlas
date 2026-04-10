"""EvoAtlas phylogenetic inference: HKY85 distance + Neighbor-Joining."""
from scipy.optimize import minimize_scalar
from scipy.linalg import expm
import numpy as np

NUC_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3, "-": -1, "N": -1}
IDX_NUC = {0: "A", 1: "C", 2: "G", 3: "T"}
AA_IDX  = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}


def hky85_Q_matrix(kappa: float, pi: np.ndarray) -> np.ndarray:
    TRANSITIONS = {(0, 2), (2, 0), (1, 3), (3, 1)}
    Q = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            rate = kappa * pi[j] if (i, j) in TRANSITIONS else pi[j]
            Q[i, j] = rate
        Q[i, i] = -Q[i, :].sum()
    return Q


def hky85_transition_probs(Q: np.ndarray, t: float) -> np.ndarray:
    return expm(Q * max(t, 1e-10))


def hky85_distance(seq1: str, seq2: str, pi: np.ndarray = None, kappa: float = 4.0) -> float:
    if pi is None:
        pi = np.array([0.25] * 4)
    Q = hky85_Q_matrix(kappa, pi)
    pairs = []
    for c1, c2 in zip(seq1.upper(), seq2.upper()):
        i, j = NUC_IDX.get(c1, -1), NUC_IDX.get(c2, -1)
        if i >= 0 and j >= 0:
            pairs.append((i, j))
    if not pairs:
        return 0.0
    p_dist = sum(i != j for i, j in pairs) / len(pairs)
    if p_dist == 0:
        return 0.0
    if p_dist >= 0.75:
        return 5.0

    def neg_log_likelihood(t):
        if t <= 0:
            return 1e10
        P = hky85_transition_probs(Q, t)
        return -sum(np.log(max(pi[i] * P[i, j], 1e-300)) for i, j in pairs)

    result = minimize_scalar(neg_log_likelihood, bounds=(1e-6, 20.0), method="bounded")
    return result.x if result.success else -np.log(1 - 4/3 * p_dist)


def compute_distance_matrix(msa_matrix: np.ndarray, seq_ids: list) -> tuple:
    n = len(msa_matrix)
    flat = "".join("".join(row) for row in msa_matrix)
    counts = {b: flat.count(b) for b in "ACGT"}
    total = max(sum(counts.values()), 1)
    pi = np.array([max(counts.get(b, 0), 1) / total for b in "ACGT"])
    pi /= pi.sum()
    print(f"Base frequencies: A={pi[0]:.3f} C={pi[1]:.3f} G={pi[2]:.3f} T={pi[3]:.3f}")
    print(f"Computing {n*(n-1)//2} HKY85 distances...")
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = hky85_distance("".join(msa_matrix[i]), "".join(msa_matrix[j]), pi)
            D[i, j] = D[j, i] = d
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n} done")
    return D, pi


class TreeNode:
    def __init__(self, name: str = None, branch_length: float = 0.0):
        self.name = name
        self.branch_length = branch_length
        self.children = []
        self.parent = None

    def is_leaf(self):
        return len(self.children) == 0

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def post_order(self):
        for child in self.children:
            yield from child.post_order()
        yield self

    def get_leaves(self):
        if self.is_leaf():
            return [self]
        leaves = []
        for c in self.children:
            leaves.extend(c.get_leaves())
        return leaves


def neighbor_joining(D: np.ndarray, names: list) -> TreeNode:
    D = D.copy().astype(float)
    n = len(names)
    nodes = [TreeNode(name=names[i]) for i in range(n)]
    active = list(range(n))

    while len(active) > 2:
        m = len(active)
        row_sums = {i: sum(D[i, j] for j in active if j != i) for i in active}
        min_q, min_pair = float("inf"), (active[0], active[1])
        for ii in range(len(active)):
            i = active[ii]
            for jj in range(ii + 1, len(active)):
                j = active[jj]
                q = (m - 2) * D[i, j] - row_sums[i] - row_sums[j]
                if q < min_q:
                    min_q, min_pair = q, (i, j)
        i, j = min_pair
        d_ij = D[i, j]
        len_i = max((d_ij + row_sums[i] - row_sums[j]) / (2 * (m - 2)), 1e-8) if m > 2 else d_ij / 2
        len_j = max(d_ij - len_i, 1e-8) if m > 2 else d_ij / 2
        u_idx = max(active) + 1
        u_node = TreeNode(name=f"int_{u_idx}")
        nodes[i].branch_length = len_i
        nodes[j].branch_length = len_j
        u_node.add_child(nodes[i])
        u_node.add_child(nodes[j])
        new_D = {}
        for k in active:
            if k not in (i, j):
                new_D[k] = (D[i, k] + D[j, k] - d_ij) / 2
        new_len = len(D)
        D_new = np.zeros((new_len + 1, new_len + 1))
        D_new[:new_len, :new_len] = D
        for k, dist in new_D.items():
            D_new[new_len, k] = D_new[k, new_len] = dist
        D = D_new
        active = [k for k in active if k not in (i, j)] + [new_len]
        nodes.append(u_node)

    i, j = active[0], active[1]
    root = TreeNode(name="root")
    nodes[i].branch_length = D[i, j] / 2
    nodes[j].branch_length = D[i, j] / 2
    root.add_child(nodes[i])
    root.add_child(nodes[j])
    print(f"NJ tree: {len(root.get_leaves())} leaves")
    return root
