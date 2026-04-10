# EvoAtlas

**Cross-Scale Evolutionary Pressure Landscape Reconstruction Engine**

A fully self-contained, CPU-only computational evolution engine that reconstructs multi-layer evolutionary pressure landscapes from nucleotide or protein sequence alignments — zero external binaries, zero GPU, pure NumPy/SciPy.

## Features

| Layer | Algorithm | Description |
|-------|-----------|-------------|
| Phylogenetic | HKY85 + Neighbor-Joining | ML distance matrix from HKY85 substitution model; O(n³) NJ tree |
| Selection | Entropy proxy (fast) / Felsenstein pruning (rigorous) | Per-site ω = dN/dS proxy via Shannon entropy or ML |
| Population genetics | Tajima's D, Fu & Li's F*, π | Per-window and genome-wide statistics |
| Epistasis | Mutual Information + WHT | Pairwise MI + Walsh-Hadamard decomposition into additive/pairwise/higher-order |

## Installation

```bash
pip install biopython numpy scipy pandas plotly --break-system-packages
```

Requires Python 3.9+, CPU only.

## Quick Start

```bash
python run_evoatlas.py          # demo on SARS-CoV-2 Spike
```

Or from Python:

```python
from src import run_evoatlas
results = run_evoatlas(
    input_source=None,          # auto-downloads SARS-CoV-2 Spike
    n_seqs=80,
    out_dir="evoatlas_output",
    window_size=30,
    mi_max_sites=120,
    fast_mode=True,
)
```

## Output Files

| File | Description |
|------|-------------|
| `evoatlas_landscape.html` | **Main output**: 4-panel interactive Plotly landscape |
| `per_site_stats.csv` | ω, Tajima's D, selection class per position |
| `mutual_information_matrix.csv` | NMI between all variable site pairs |
| `summary.json` | Machine-readable summary statistics |

## Architecture

```
Input: FASTA sequences (NCBI or local)
  │
  ▼
Global alignment (Needleman-Wunsch, Biopython)
  │
  ▼
Layer 1 ─ HKY85 distance matrix (κ, base frequencies)
  │
  ▼
Layer 2 ─ Neighbor-Joining tree (Saitou & Nei 1987)
  │
  ▼
Layer 3 ─ Entropy-based ω proxy per site
  │
  ▼
Layer 4 ─ Population genetics (Tajima's D, Fu & Li F*)
  │
  ▼
Layer 5 ─ Mutual Information + WHT epistasis decomposition
  │
  ▼
Interactive Plotly landscape (4 panels)
```

## Demo Results (SARS-CoV-2 Spike)

- 5 representative variant sequences (Wuhan-Hu-1, Alpha, Delta, Omicron BA.1, XBB.1.5)
- RBD region shows highest variability signal
- WHT decomposition reveals dominant higher-order epistasis (96.1%)
- Tajima's D near zero across alignment (neutral demographic history)

## Algorithm Complexity

| Layer | Algorithm | Complexity |
|-------|-----------|-----------|
| Distance | HKY85 ML | O(L × n²) |
| Tree | Neighbor-Joining | O(n³) |
| Selection | Entropy proxy | O(n × L) |
| Selection | Felsenstein pruning | O(K² × N × L) |
| Population | Tajima's D | O(n² × L) |
| Epistasis | MI + WHT | O(n × S² + 2^L log 2^L) |

## References

- Felsenstein, J. (1981). Evolutionary trees from DNA sequences. *JME*.
- Saitou, N. & Nei, M. (1987). Neighbor-joining method. *MBE*.
- Hasegawa, M. et al. (1985). HKY85 model. *JME*.
- Tajima, F. (1989). Statistical method for testing the neutral hypothesis. *Genetics*.
- Fu, Y.X. & Li, W.H. (1993). Statistical tests of neutrality. *Genetics*.
- Faure, A.J. et al. (2024). WHT epistasis decomposition. *PLoS Comput. Biol.*

## License

Apache 2.0
