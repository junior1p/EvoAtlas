"""EvoAtlas visualization: interactive 4-panel Plotly landscape."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

SPIKE_DOMAINS = [
    {"name": "NTD",       "start": 13,  "end": 305,  "color": "rgba(173,216,230,0.3)"},
    {"name": "RBD",       "start": 319, "end": 541,  "color": "rgba(255,165,0,0.3)"},
    {"name": "RBM",       "start": 437, "end": 508,  "color": "rgba(255,69,0,0.4)"},
    {"name": "SD1",       "start": 542, "end": 590,  "color": "rgba(144,238,144,0.3)"},
    {"name": "SD2",       "start": 591, "end": 686,  "color": "rgba(144,238,144,0.2)"},
    {"name": "Furin site","start": 681, "end": 685,  "color": "rgba(255,0,0,0.4)"},
    {"name": "FP",        "start": 816, "end": 833,  "color": "rgba(148,0,211,0.3)"},
    {"name": "HR1",       "start": 912, "end": 984,  "color": "rgba(70,130,180,0.3)"},
    {"name": "HR2",       "start":1163, "end":1202,  "color": "rgba(70,130,180,0.2)"},
    {"name": "TM",        "start":1214, "end":1237,  "color": "rgba(105,105,105,0.3)"},
]


def create_evoatlas_landscape(
    omega: np.ndarray,
    tajd_positions: np.ndarray,
    tajd_values: np.ndarray,
    mi_sites: np.ndarray,
    MI_matrix: np.ndarray,
    fu_li_F: float,
    wht_result: dict,
    n_seqs: int,
    align_len: int,
    output_path: str = "evoatlas_landscape.html",
) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1,
        row_heights=[0.35, 0.25, 0.25, 0.15],
        subplot_titles=[
            "Evolutionary Pressure: ω Proxy per Site",
            f"Population Genetics: Tajima's D  (Fu & Li F* = {fu_li_F:.3f})",
            "Epistatic Coupling: Normalized Mutual Information",
            "WHT Epistasis Decomposition",
        ],
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    x_pos = np.arange(align_len)
    omega_valid = np.where(np.isfinite(omega), omega, 1.0)
    omega_clipped = np.clip(omega_valid, 0.0, 5.0)
    colors = ["#d73027" if w > 0.6 else "#4575b4" for w in omega_clipped]

    fig.add_trace(go.Bar(
        x=x_pos, y=omega_clipped, marker_color=colors, name="ω proxy",
        hovertemplate="Pos %{x}<br>ω = %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(y=0.6, line_dash="dash", line_color="black",
                  line_width=1.2, row=1, col=1,
                  annotation_text="ω=0.6 (threshold)", annotation_position="right")

    for dom in SPIKE_DOMAINS:
        if dom["start"] < align_len:
            fig.add_vrect(
                x0=dom["start"], x1=min(dom["end"], align_len),
                fillcolor=dom["color"], line_width=0,
                annotation_text=dom["name"],
                annotation_position="top left",
                annotation_font_size=8, row=1, col=1,
            )

    if len(tajd_positions) > 0:
        d_colors = ["#e31a1c" if d > 2 else "#1f78b4" if d < -2 else "#33a02c"
                   for d in tajd_values]
        fig.add_trace(go.Scatter(
            x=tajd_positions, y=tajd_values,
            mode="lines+markers", marker=dict(color=d_colors, size=5),
            line=dict(color="#636363", width=1.2),
            name="Tajima's D", fill="tozeroy",
            fillcolor="rgba(100,100,200,0.15)",
            hovertemplate="Pos %{x}<br>D = %{y:.3f}<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="gray",
                      line_width=1, row=2, col=1)
        fig.add_hline(y=2, line_dash="dot", line_color="red",
                      line_width=1, row=2, col=1)
        fig.add_hline(y=-2, line_dash="dot", line_color="blue",
                      line_width=1, row=2, col=1)

    if len(mi_sites) > 1:
        fig.add_trace(go.Heatmap(
            x=mi_sites, y=mi_sites, z=MI_matrix,
            colorscale="Reds", name="MI",
            colorbar=dict(title="NMI", len=0.22, y=0.28, x=1.02, thickness=12),
            hovertemplate="Site %{x} ↔ %{y}<br>NMI = %{z:.3f}<extra></extra>",
        ), row=3, col=1)

    if wht_result:
        fracs = [
            wht_result.get("additive_fraction", 0),
            wht_result.get("pairwise_fraction", 0),
            wht_result.get("higher_fraction", 0),
        ]
        labels = ["Additive (α)", "Pairwise (β)", "Higher-order (γ)"]
        fig.add_trace(go.Bar(
            x=labels, y=fracs,
            marker_color=["#2ca02c", "#ff7f0e", "#d62728"],
            name="Variance decomposition",
            text=[f"{f*100:.1f}%" for f in fracs],
            textposition="outside",
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        ), row=4, col=1)

    fig.update_layout(
        title=dict(
            text="<b>EvoAtlas: Evolutionary Pressure Landscape</b><br>"
                 f"<sub>{n_seqs} sequences × {align_len} positions | "
                 "CPU-only: HKY85 · Entropy proxy · MI · WHT</sub>",
            x=0.5, font=dict(size=15),
        ),
        height=1000, template="plotly_white",
        hovermode="x unified",
        paper_bgcolor="#fafafa",
        showlegend=True,
        legend=dict(orientation="h", y=-0.05),
    )
    fig.update_yaxes(title_text="ω (variability)", row=1, col=1)
    fig.update_yaxes(title_text="Tajima's D", row=2, col=1)
    fig.update_yaxes(title_text="Position", row=3, col=1)
    fig.update_yaxes(title_text="Variance fraction", row=4, col=1, range=[0, 1])
    fig.update_xaxes(title_text="Alignment position", row=3, col=1)

    fig.write_html(output_path)
    print(f"Landscape saved: {output_path}")
    return fig
