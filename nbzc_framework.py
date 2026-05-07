#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Three-Stage Framework for Urban EV Fast-Charging Planning
=========================================================
Stage 1 : Activity-Based Dwell Potential (ABDP)          [Paper §III]
Stage 2 : Neutralization-Based Zonal Clustering (NBZC)   [Paper §V–VII]
Stage 3 : Community Detection → Planning Districts        [Paper §VIII–X]

Paper: "From Dwell Potential to Planning Districts: A Three-Stage
        Framework for Urban EV Fast-Charging Planning"
Authors: E. Sobhani, H. E. Z. Farag — York University, Toronto, Canada
"""

# =============================================================================
# 0.  Imports
# =============================================================================
import itertools
import warnings
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import networkx.community as nx_comm
import numpy as np
import pandas as pd
from libpysal.weights import Rook

warnings.filterwarnings("ignore")

# =============================================================================
# 1.  Global Configuration
# =============================================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
SHP_PATH = Path("arterial_with_bevD_activityD_for_clustering.shp")
OUT_DIR  = Path(".")
CRS_METRIC = "EPSG:32617"           # UTM zone 17N — distances in metres

# ── Input column names (normalised to [0, 1]) ─────────────────────────────────
BEV_COL = "BEV_density_norm"        # S^EV_j  normalised BEV density
ACT_COL = "S_norm_con"              # D̄_j    normalised ABDP

# ── ABDP kernel parameters (Paper §III-B, Eq. 7) ─────────────────────────────
T_STAR_MIN = 30.0    # target fast-charging dwell duration (min)
SIGMA_MIN  = 10.0    # Gaussian tolerance width (min)

# ── POI category representative dwell times T_k (Paper §III-A) ───────────────
# Adjust values to match your empirical survey or mobility literature.
POI_DWELL_TIMES: dict[str, float] = {
    "food_service" : 30.0,   # restaurants / cafes — ideal compatibility
    "retail"       : 45.0,   # shopping centres
    "services"     : 20.0,   # banks, pharmacies, post offices
    "leisure"      : 60.0,   # gyms, entertainment venues
    "education"    : 90.0,   # schools, universities
    "healthcare"   : 40.0,   # clinics, hospitals
    "transit"      : 10.0,   # transit stops (too brief)
    "office"       : 120.0,  # workplaces (too long)
}

# ── NBZC sensitivity grid (Paper §Results — 1 008 runs) ──────────────────────
GRID_MAX_CLUSTER_SIZE  = [2, 4, 8, 12, 15, 20]
GRID_MAX_ABS_IMBALANCE = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]
GRID_LAMBDA_SHAPE      = [0.0, 0.0002, 0.0005, 0.0010]
GRID_MAX_RADIUS_M      = [None, 1000, 2000, 4000, 6000, 8000]

# ── Community detection sweep parameters (Paper §IX-B) ───────────────────────
K_MAX_COMMUNITIES  = 130
CD_N_SEEDS         = 20
CD_LOUVAIN_RES     = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
CD_GN_LEVELS       = 15

RANDOM_SEED = 42

# =============================================================================
# 2.  Data Loading & Preprocessing
# =============================================================================

def load_data(shp_path: Path = SHP_PATH) -> gpd.GeoDataFrame:
    """
    Load arterial-zone shapefile, reproject to metric CRS, add derived columns.

    Returns
    -------
    gdf : GeoDataFrame with columns BEV_COL, ACT_COL, zone_area_m2, cx, cy
    """
    gdf = gpd.read_file(shp_path)

    if gdf.crs is None or gdf.crs.to_epsg() != 32617:
        gdf = gdf.to_crs(CRS_METRIC)

    # Normalise legacy column name if present
    if "BEV_densit" in gdf.columns and BEV_COL not in gdf.columns:
        gdf = gdf.rename(columns={"BEV_densit": BEV_COL})

    gdf = gdf.reset_index(drop=True)
    gdf["zone_area_m2"] = gdf.geometry.area
    gdf["cx"] = gdf.geometry.centroid.x
    gdf["cy"] = gdf.geometry.centroid.y

    print(f"Loaded {len(gdf):,} arterial zones  |  CRS: {gdf.crs.to_epsg()}")
    _check_columns(gdf, [BEV_COL, ACT_COL])
    return gdf


def _check_columns(gdf: gpd.GeoDataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise KeyError(
            f"Required columns not found: {missing}\n"
            f"Available columns: {list(gdf.columns)}"
        )


# =============================================================================
# 3.  Stage 1 — Activity-Based Dwell Potential (ABDP)    [Paper §III]
# =============================================================================

def gaussian_dwell_weight(T_k: float,
                           T_star: float = T_STAR_MIN,
                           sigma: float  = SIGMA_MIN) -> float:
    """
    Eq. (7): Gaussian dwell-compatibility weight.

        w_k = exp( -(T_k - T*)² / (2σ²) )

    Assigns maximal weight when T_k = T*, tapering for shorter/longer
    activities.  σ controls the tolerance width of the compatibility window.
    """
    return float(np.exp(-((T_k - T_star) ** 2) / (2.0 * sigma ** 2)))


def compute_abdp_from_pois(
    gdf: gpd.GeoDataFrame,
    poi_gdf: gpd.GeoDataFrame,
    category_col: str,
    T_star: float = T_STAR_MIN,
    sigma: float  = SIGMA_MIN,
) -> gpd.GeoDataFrame:
    """
    Full ABDP pipeline from raw POI data (Paper §III-A — §III-C).

    Steps
    -----
    1. Compute w_k for every POI category via Gaussian kernel (Eq. 7).
    2. Assign each POI to its arterial zone (Eq. 5–6).
    3. Eq. (8): S^act_j = Σ_k w_k · N_jk
    4. Eq. (9): D_j = S^act_j / A^Ar_j
    5. Eq. (10): D̄_j = min-max normalise D_j → [0, 1]   (written to ACT_COL)

    Parameters
    ----------
    gdf          : zone GeoDataFrame (must already have zone_area_m2)
    poi_gdf      : POI GeoDataFrame with 'category_col' and geometry
    category_col : column mapping each POI to a key in POI_DWELL_TIMES
    """
    # Step 1: compatibility weights w_k
    w_k = {cat: gaussian_dwell_weight(T_k, T_star, sigma)
           for cat, T_k in POI_DWELL_TIMES.items()}

    # Step 2: spatial join — assign each POI to its zone  (Eq. 5)
    poi_joined = gpd.sjoin(
        poi_gdf[[category_col, "geometry"]],
        gdf[["geometry"]],
        how="left",
        predicate="within",
    ).rename(columns={"index_right": "zone_idx"})

    # Step 3: S^act_j (Eq. 8)
    poi_joined["w"] = poi_joined[category_col].map(w_k).fillna(0.0)
    s_act = (poi_joined.groupby("zone_idx")["w"]
             .sum()
             .reindex(gdf.index, fill_value=0.0))
    gdf["S_act"] = s_act.values

    # Step 4: area-density D_j (Eq. 9)
    gdf["D_raw"] = gdf["S_act"] / gdf["zone_area_m2"].clip(lower=1e-9)

    # Step 5: min-max normalise → D̄_j = ABDP (Eq. 10)
    gdf[ACT_COL] = _minmax(gdf["D_raw"])

    return gdf


def _minmax(series: pd.Series) -> pd.Series:
    """Min-max normalise a Series to [0, 1]."""
    lo, hi = series.min(), series.max()
    return (series - lo) / (hi - lo + 1e-12)


# =============================================================================
# 4.  Adjacency Graph & Zone-Level Imbalance    [Paper §II-D, §V]
# =============================================================================

def build_rook_graph(gdf: gpd.GeoDataFrame) -> nx.Graph:
    """
    Build undirected rook-contiguity graph G = (V, E)  (Paper §II-D).
    Edge (i, k) ∈ E iff zones i and k share a boundary of positive length.
    """
    w = Rook.from_dataframe(gdf, silence_warnings=True)
    G = nx.Graph()
    G.add_nodes_from(range(len(gdf)))
    for i, neighbours in w.neighbors.items():
        for j in neighbours:
            if i < j:
                G.add_edge(i, j)
    return G


def zone_imbalance(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Eq. (13): d_i = D̄_i − S^EV_i  (signed zone-level imbalance).
    Both signals must already be normalised to [0, 1].
    """
    return (gdf[ACT_COL].values - gdf[BEV_COL].values).astype(float)


def zone_neutralisation_index(gdf: gpd.GeoDataFrame,
                               eps: float = 1e-9) -> np.ndarray:
    """
    Eq. (20): η_i = |d_i| / (D̄_i + S^EV_i + ε)   ∈ [0, 1].
    Lower → zone is internally balanced.
    """
    d     = gdf[ACT_COL].values - gdf[BEV_COL].values
    denom = gdf[ACT_COL].values + gdf[BEV_COL].values + eps
    return np.abs(d) / denom


# =============================================================================
# 5.  Stage 2 — NBZC Algorithm    [Paper §VI, Algorithm 1]
# =============================================================================

def nbzc(
    G: nx.Graph,
    diff: np.ndarray,
    coords: np.ndarray,
    max_cluster_size: int    = 8,
    max_abs_imbalance: float = 0.5,
    lambda_shape: float      = 0.0002,
    max_radius_m: float | None = 1000.0,
) -> np.ndarray:
    """
    Algorithm 1 — Greedy Neutralisation + Shape-Constrained Clustering.

    Parameters
    ----------
    G                  : rook-contiguity graph (N nodes)
    diff               : zone-level imbalance d_i  (shape N,)
    coords             : zone centroid coordinates  (shape N×2, metres)
    max_cluster_size   : hard cap |C| ≤ max_cluster_size
    max_abs_imbalance  : neutralisation tolerance  |Σd_i| ≤ threshold
    lambda_shape       : compactness trade-off λ_shape ≥ 0
    max_radius_m       : hard cap on cluster radius (metres); None = no cap

    Returns
    -------
    labels : integer array (N,) — cluster id for each zone (0-indexed)
    """
    N          = len(diff)
    labels     = np.full(N, -1, dtype=int)
    cluster_id = 0
    unassigned = set(range(N))

    # Pre-compute adjacency lists once
    adj = {i: set(G.neighbors(i)) for i in range(N)}

    while unassigned:
        # §VI-D Seed: zone with largest |d_i| among unassigned
        seed = max(unassigned, key=lambda i: abs(diff[i]))
        cluster     = {seed}
        cluster_sum = float(diff[seed])
        unassigned.remove(seed)

        while True:
            # Frontier F: rook-adjacent unassigned zones  (Eq. 14)
            F = set().union(*(adj[i] for i in cluster)) & unassigned
            if not F:
                break

            # §VI-C: evaluate combined score for every frontier candidate
            best_j, best_score = None, float("inf")
            for j in F:
                new_sum   = cluster_sum + diff[j]
                imb_score = abs(new_sum)

                # Radius: max distance from new centroid  (Eq. 11–12)
                pts      = coords[list(cluster | {j})]
                centroid = pts.mean(axis=0)
                radius   = float(np.max(np.linalg.norm(pts - centroid, axis=1)))

                # Hard shape cap
                if max_radius_m is not None and radius > max_radius_m:
                    continue

                # Eq. (15): combined score
                score = imb_score + lambda_shape * radius
                if score < best_score:
                    best_score, best_j = score, j

            if best_j is None:
                break   # no candidate passes shape cap

            new_sum = cluster_sum + diff[best_j]

            # §VI-E Stopping rules (evaluated in priority order)
            if len(cluster) >= max_cluster_size:
                break
            if abs(new_sum) > max_abs_imbalance:
                break
            # Neutralisation safeguard: halt if balance worsens (|C| ≥ 2)
            if len(cluster) >= 2 and abs(new_sum) > abs(cluster_sum):
                break

            cluster.add(best_j)
            cluster_sum = new_sum
            unassigned.discard(best_j)

        for node in cluster:
            labels[node] = cluster_id
        cluster_id += 1

    return labels


# =============================================================================
# 6.  Evaluation — Cluster-Level Neutralisation    [Paper §VII-A]
# =============================================================================

def cluster_neutralisation(
    labels: np.ndarray,
    gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Per-cluster neutralisation index  η_c  (Eq. 19).

        Δ_c = Σ_{i∈c} d_i
        T_c = Σ_{i∈c} (D̄_i + S^EV_i)
        η_c = |Δ_c| / T_c

    Returns DataFrame indexed by cluster id with columns:
    size, delta_c, T_c, eta_c.
    """
    act  = gdf[ACT_COL].values
    bev  = gdf[BEV_COL].values
    d    = act - bev

    rows = []
    for c in np.unique(labels):
        mask   = labels == c
        delta  = float(d[mask].sum())
        T_c    = float((act[mask] + bev[mask]).sum())
        eta    = abs(delta) / T_c if T_c > 0 else 0.0
        rows.append({"cluster": int(c), "size": int(mask.sum()),
                     "delta_c": delta, "T_c": T_c, "eta_c": eta})
    return pd.DataFrame(rows).set_index("cluster")


def _cluster_radii(labels: np.ndarray, coords: np.ndarray) -> list[float]:
    radii = []
    for c in np.unique(labels):
        pts = coords[labels == c]
        if len(pts) <= 1:
            radii.append(0.0)
        else:
            cen = pts.mean(axis=0)
            radii.append(float(np.max(np.linalg.norm(pts - cen, axis=1))))
    return radii


def print_neutralisation_table(
    eta_i: np.ndarray,
    cldf: pd.DataFrame,
) -> None:
    """Print Table I: Zone-Level vs Cluster-Level Neutralisation."""
    c_eta = cldf["eta_c"].values
    rows  = [("Mean η",   eta_i.mean(),        c_eta.mean()),
             ("Median η", np.median(eta_i),    np.median(c_eta)),
             ("Min η",    eta_i.min(),          c_eta.min()),
             ("Max η",    eta_i.max(),          c_eta.max())]
    print("\n── Table I: Zone vs Cluster Neutralisation ──────────────────────")
    print(f"  {'Metric':<12} {'Zone η_i':>12} {'Cluster η_c':>14}")
    print("  " + "-" * 40)
    for label, z, c in rows:
        print(f"  {label:<12} {z:>12.3f} {c:>14.3f}")
    print()


# =============================================================================
# 7.  Graph Metrics    [Paper §VII-B]
# =============================================================================

def graph_metrics(G: nx.Graph) -> dict:
    """Compute all structural metrics reported in Table II."""
    lcc  = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    degs = sorted(d for _, d in G.degree())
    kc   = nx.core_number(G)
    kmax = max(kc.values())

    return {
        "nodes"            : G.number_of_nodes(),
        "edges"            : G.number_of_edges(),
        "density"          : round(nx.density(G), 4),
        "components"       : nx.number_connected_components(G),
        "deg_min"          : degs[0],
        "deg_med"          : int(np.median(degs)),
        "deg_mean"         : round(float(np.mean(degs)), 2),
        "deg_max"          : degs[-1],
        "leaf_fraction"    : round(sum(1 for d in degs if d == 1) / len(degs), 4),
        "transitivity"     : round(nx.transitivity(G), 4),
        "avg_clustering"   : round(nx.average_clustering(G), 4),
        "triangles"        : sum(nx.triangles(G).values()) // 3,
        "bridges"          : len(list(nx.bridges(G))),
        "artic_points"     : len(list(nx.articulation_points(G))),
        "k_core_max"       : kmax,
        "k_core_max_size"  : sum(1 for v in kc.values() if v == kmax),
        "avg_shortest_path": round(nx.average_shortest_path_length(lcc), 2),
        "diameter"         : nx.diameter(lcc),
        "radius"           : nx.radius(lcc),
        "assortativity"    : round(nx.degree_assortativity_coefficient(G), 4),
    }


def print_graph_metrics_table(zm: dict, cm: dict) -> None:
    """Print Table II: Zone Graph vs Cluster Graph."""
    rows = [
        ("nodes",            "Nodes (|V|)"),
        ("edges",            "Edges (|E|)"),
        ("density",          "Density"),
        ("components",       "Components"),
        ("deg_min",          "Deg min"),
        ("deg_med",          "Deg median"),
        ("deg_mean",         "Deg mean"),
        ("deg_max",          "Deg max"),
        ("leaf_fraction",    "Leaf fraction"),
        ("transitivity",     "Transitivity"),
        ("avg_clustering",   "Avg local clustering"),
        ("triangles",        "Triangles (total)"),
        ("bridges",          "Bridges"),
        ("artic_points",     "Articulation points"),
        ("k_core_max",       "Max k-core"),
        ("k_core_max_size",  "k_max-core size"),
        ("avg_shortest_path","Avg shortest path"),
        ("diameter",         "Diameter"),
        ("radius",           "Radius"),
        ("assortativity",    "Degree assortativity"),
    ]
    print("\n── Table II: Zone Graph vs Cluster Graph ────────────────────────")
    print(f"  {'Metric':<24} {'Zone graph':>12} {'Cluster graph':>14}")
    print("  " + "-" * 52)
    for key, label in rows:
        print(f"  {label:<24} {str(zm.get(key,'–')):>12} {str(cm.get(key,'–')):>14}")
    print()


# =============================================================================
# 8.  NBZC Sensitivity Analysis    [Paper §Results-A]
# =============================================================================

def run_sensitivity(
    G: nx.Graph,
    diff: np.ndarray,
    coords: np.ndarray,
    gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Full-factorial sensitivity sweep (1 008 configurations).
    Grid: 6 × 7 × 4 × 6 = 1 008   (rook adjacency only, as in paper).

    Returns DataFrame with one row per parameter combination.
    """
    grid = list(itertools.product(
        GRID_MAX_CLUSTER_SIZE,
        GRID_MAX_ABS_IMBALANCE,
        GRID_LAMBDA_SHAPE,
        GRID_MAX_RADIUS_M,
    ))
    assert len(grid) == 1008, f"Expected 1008 configs, got {len(grid)}"
    print(f"[Stage 2] Sensitivity sweep: {len(grid)} configurations …")

    records = []
    for idx, (mcs, mai, ls, mr) in enumerate(grid):
        labels = nbzc(G, diff, coords,
                      max_cluster_size=mcs, max_abs_imbalance=mai,
                      lambda_shape=ls, max_radius_m=mr)
        cldf   = cluster_neutralisation(labels, gdf)
        radii  = _cluster_radii(labels, coords)

        records.append({
            "max_cluster_size"  : mcs,
            "max_abs_imbalance" : mai,
            "lambda_shape"      : ls,
            "max_radius_m"      : mr if mr is not None else -1,
            "n_clusters"        : len(cldf),
            "sum_abs_imbalance" : float(cldf["eta_c"].abs().sum()),
            "mean_eta_c"        : float(cldf["eta_c"].mean()),
            "max_eta_c"         : float(cldf["eta_c"].max()),
            "mean_radius_m"     : float(np.mean(radii)),
            "max_radius_obs_m"  : float(np.max(radii)),
            "mean_cluster_size" : float(np.mean(cldf["size"])),
            "pct_singletons"    : 100.0 * float((cldf["size"] == 1).mean()),
        })
        if (idx + 1) % 200 == 0:
            print(f"  … {idx + 1}/{len(grid)}")

    return pd.DataFrame(records)


# =============================================================================
# 9.  Pareto Frontier & Knee Selection    [Paper §Results-A]
# =============================================================================

def pareto_2d(
    df: pd.DataFrame,
    x_col: str = "sum_abs_imbalance",
    y_col: str = "mean_radius_m",
) -> pd.DataFrame:
    """
    Non-dominated Pareto frontier under simultaneous minimisation of
    x_col (neutralisation loss) and y_col (compactness loss).
    """
    x    = df[x_col].values
    y    = df[y_col].values
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if not keep[i]:
            continue
        dom = (x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i]))
        dom[i] = False
        if dom.any():
            keep[i] = False
    return df[keep].sort_values([x_col, y_col]).reset_index(drop=True)


def knee_from_pareto(
    pf: pd.DataFrame,
    x_col: str = "sum_abs_imbalance",
    y_col: str = "mean_radius_m",
) -> pd.Series:
    """
    Knee point: normalise both Pareto axes, select the point closest
    to the ideal origin (0, 0) in the normalised plane.
    """
    x  = pf[x_col].values
    y  = pf[y_col].values
    xn = (x - x.min()) / (x.max() - x.min() + 1e-12)
    yn = (y - y.min()) / (y.max() - y.min() + 1e-12)
    return pf.iloc[int(np.argmin(np.hypot(xn, yn)))]


# =============================================================================
# 10. Cluster Graph & Aggregate Signals    [Paper §IX]
# =============================================================================

def build_cluster_graph(
    G_zone: nx.Graph,
    labels: np.ndarray,
) -> tuple[nx.Graph, dict[int, list[int]]]:
    """
    Collapse zone graph into cluster graph G_c = (V_c, E_c).
    Nodes = cluster ids; edges = inter-cluster rook adjacency.
    """
    Gc  = nx.Graph()
    Gc.add_nodes_from(np.unique(labels))
    for u, v in G_zone.edges():
        cu, cv = int(labels[u]), int(labels[v])
        if cu != cv:
            Gc.add_edge(cu, cv)

    c2z: dict[int, list[int]] = defaultdict(list)
    for z, c in enumerate(labels):
        c2z[int(c)].append(z)
    return Gc, dict(c2z)


def cluster_agg_signals(
    labels: np.ndarray,
    gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Per-cluster aggregate signals used in Stage 3 (Eqs. 21–24).

        D̄_c  = mean(D̄_i) for i ∈ c
        S^EV_c = mean(S^EV_i) for i ∈ c
        d_c   = D̄_c − S^EV_c          (Eq. 21)
        T_c   = D̄_c + S^EV_c          (per-cluster signal magnitude)
    """
    act  = gdf[ACT_COL].values
    bev  = gdf[BEV_COL].values
    rows = {}
    for c in np.unique(labels):
        mask = labels == c
        D_c  = float(act[mask].mean())
        S_c  = float(bev[mask].mean())
        rows[int(c)] = {
            "D_bar_c" : D_c,
            "S_ev_c"  : S_c,
            "delta_c" : D_c - S_c,
            "T_c"     : D_c + S_c,
            "n_zones" : int(mask.sum()),
        }
    return pd.DataFrame(rows).T


# =============================================================================
# 11. Community Detection — J-Neutralisation Objective    [Paper §IX-A]
# =============================================================================

def J_neutralisation(
    communities: list[frozenset],
    agg: pd.DataFrame,
) -> float:
    """
    Eqs. (22)–(26): weighted-average community neutralisation objective.

        Δ_g = Σ_{c∈g} d_c                    (Eq. 23)
        T_g = Σ_{c∈g} T_c                    (Eq. 24)
        η_g = |Δ_g| / T_g                    (Eq. 25)
        J   = Σ_g (η_g · T_g) / Σ_g T_g     (Eq. 26)

    Lower J → stronger behavioural self-sufficiency.
    """
    num, den = 0.0, 0.0
    for g in communities:
        valid   = [c for c in g if c in agg.index]
        if not valid:
            continue
        Delta_g = float(agg.loc[valid, "delta_c"].sum())
        T_g     = float(agg.loc[valid, "T_c"].sum())
        if T_g <= 0:
            continue
        num += abs(Delta_g)   # η_g · T_g = |Δ_g|
        den += T_g
    return num / den if den > 0 else float("nan")


def partition_quality(
    G: nx.Graph,
    communities: list[frozenset],
) -> dict[str, float]:
    """
    Structural quality metrics (Eqs. 30–32).

    coverage    : fraction of edges within communities
    modularity  : Q under null model  (Eq. 31)
    performance : correctly classified node pairs  (Eq. 32)
    """
    comm_list = [list(c) for c in communities]
    node_to_comm = {n: i for i, c in enumerate(comm_list) for n in c}

    # Coverage (Eq. 30)
    intra = sum(1 for u, v in G.edges()
                if node_to_comm.get(u) == node_to_comm.get(v))
    M     = G.number_of_edges()
    cov   = intra / M if M > 0 else 0.0

    # Modularity (Eq. 31)
    try:
        mod = nx_comm.modularity(G, [frozenset(c) for c in comm_list])
    except Exception:
        mod = float("nan")

    # Performance (Eq. 32)
    n  = G.number_of_nodes()
    tp = intra
    tn = sum(1 for u, v in itertools.combinations(G.nodes(), 2)
             if not G.has_edge(u, v)
             and node_to_comm.get(u) != node_to_comm.get(v))
    perf = (tp + tn) / (n * (n - 1) / 2) if n > 1 else 0.0

    return {"modularity": mod, "coverage": cov, "performance": perf}


# =============================================================================
# 12. Community Detection Sweep    [Paper §IX-B]
# =============================================================================

def _fluid_safe(G: nx.Graph, k: int, seed: int) -> list[frozenset]:
    """Fluid communities with graceful fallback for disconnected graphs."""
    if not nx.is_connected(G):
        lcc_nodes = max(nx.connected_components(G), key=len)
        lcc       = G.subgraph(lcc_nodes).copy()
        others    = set(G.nodes()) - lcc_nodes
        k_lcc     = min(k, len(lcc_nodes))
        comms     = list(nx_comm.asyn_fluidc(lcc, k=k_lcc, seed=seed))
        return comms + [frozenset({v}) for v in others]
    return list(nx_comm.asyn_fluidc(G, k=k, seed=seed))


def _gn_at_level(G: nx.Graph, level: int) -> list[frozenset]:
    """Advance the Girvan–Newman generator exactly `level` steps."""
    gen   = nx_comm.girvan_newman(G)
    comms = None
    for _ in range(level):
        try:
            comms = next(gen)
        except StopIteration:
            break
    return [frozenset(c) for c in comms] if comms else []


def community_sweep(
    Gc: nx.Graph,
    agg: pd.DataFrame,
    k_max: int = K_MAX_COMMUNITIES,
    n_seeds: int = CD_N_SEEDS,
) -> pd.DataFrame:
    """
    Exhaustive community detection sweep producing ≈ 2 776 candidate
    partitions (Paper §IX-B).

    Algorithms
    ----------
    1. Greedy modularity                    — deterministic
    2. Label propagation                    — 20 seeds
    3. Asynchronous fluid communities       — k ∈ [2, 130] × 20 seeds
    4. Kernighan–Lin bisection              — 20 seeds
    5. Louvain (γ ∈ {0.3,0.5,0.7,1.0,1.5,2.0}) × 20 seeds
    6. Girvan–Newman                        — 15 levels
    """
    rng    = np.random.default_rng(RANDOM_SEED)
    seeds  = rng.integers(0, 2**31, size=n_seeds).tolist()
    n_nodes = Gc.number_of_nodes()
    k_range = range(2, min(k_max, n_nodes) + 1)
    records = []

    def _add(alg: str, partition, **meta):
        comms = [frozenset(c) for c in partition]
        J     = J_neutralisation(comms, agg)
        q     = partition_quality(Gc, comms)
        records.append({
            "algorithm"  : alg,
            "k"          : len(comms),
            "J_neutral"  : J,
            "modularity" : q["modularity"],
            "coverage"   : q["coverage"],
            "performance": q["performance"],
            **meta,
        })

    # 1. Greedy modularity
    _add("greedy_modularity",
         nx_comm.greedy_modularity_communities(Gc))

    # 2. Label propagation
    for s in seeds:
        _add("label_propagation",
             nx_comm.label_propagation_communities(Gc), seed=int(s))

    # 3. Asynchronous fluid communities
    for k in k_range:
        for s in seeds:
            try:
                _add("fluid", _fluid_safe(Gc, k=k, seed=int(s)),
                     k_target=k, seed=int(s))
            except Exception:
                pass

    # 4. Kernighan–Lin bisection
    for s in seeds:
        try:
            _add("kernighan_lin",
                 nx_comm.kernighan_lin_bisection(Gc, seed=int(s)),
                 seed=int(s))
        except Exception:
            pass

    # 5. Louvain
    for res in CD_LOUVAIN_RES:
        for s in seeds:
            _add("louvain",
                 nx_comm.louvain_communities(Gc, resolution=res, seed=int(s)),
                 resolution=res, seed=int(s))

    # 6. Girvan–Newman
    for level in range(1, CD_GN_LEVELS + 1):
        part = _gn_at_level(Gc, level)
        if part:
            _add("girvan_newman", part, level=level)

    df = pd.DataFrame(records)
    print(f"  Community sweep complete: {len(df):,} candidate partitions.")
    return df


# =============================================================================
# 13. Two-Stage District Resolution Selection    [Paper §X]
# =============================================================================

def best_J_per_k(df_cd: pd.DataFrame) -> pd.Series:
    """Return J_best(k) = min J_neutral at each community count k."""
    return df_cd.groupby("k")["J_neutral"].min().sort_index()


def detect_knee(j_series: pd.Series) -> int:
    """
    Curvature-based knee: maximum perpendicular deviation from the chord
    connecting endpoints in the normalised (k, J_best) plane (Paper §X-B).
    """
    k_vals = np.array(j_series.index, dtype=float)
    j_vals = np.array(j_series.values, dtype=float)
    kn     = (k_vals - k_vals.min()) / (k_vals.max() - k_vals.min() + 1e-12)
    jn     = (j_vals - j_vals.min()) / (j_vals.max() - j_vals.min() + 1e-12)

    p1, p2  = np.array([kn[0], jn[0]]), np.array([kn[-1], jn[-1]])
    chord   = p2 - p1
    chord_L = np.linalg.norm(chord) + 1e-12

    deviations = []
    for i in range(len(kn)):
        pt   = np.array([kn[i], jn[i]])
        dev  = abs((pt[0] - p1[0]) * chord[1] - (pt[1] - p1[1]) * chord[0])
        deviations.append(dev / chord_L)

    return int(j_series.index[int(np.argmax(deviations))])


def stability_scores(df_adm: pd.DataFrame) -> pd.Series:
    """
    Eqs. (37)–(38): aggregate stability score S(k).

        S_M(k) = 1 − |norm(M)(k+1) − norm(M)(k)| / max_Δ
        S(k)   = (S_J + S_Q + S_P + S_C) / 4
    """
    metrics = ["J_neutral", "modularity", "coverage", "performance"]
    result  = pd.DataFrame(index=df_adm.index)
    for m in metrics:
        vals    = df_adm[m].values.astype(float)
        lo, hi  = vals.min(), vals.max()
        norm    = (vals - lo) / (hi - lo + 1e-12)
        diffs   = np.abs(np.diff(norm))
        max_d   = diffs.max() + 1e-12
        S_m     = np.ones(len(df_adm))
        S_m[:-1] = 1.0 - diffs / max_d
        result[f"S_{m}"] = S_m
    return result.mean(axis=1).rename("S_agg")


def select_final_k(
    df_cd: pd.DataFrame,
) -> tuple[int, int, pd.DataFrame]:
    """
    Two-stage resolution selection (Paper §X-B, §X-C).

    Stage 1 — Knee detection on J_best(k) to bound the search space.
    Stage 2 — Pareto efficiency + stability maximisation on k ≤ k_knee.

    Returns
    -------
    k_knee      : knee point index
    k_star      : selected final community count
    df_adm      : admissible-set DataFrame with all metrics + stability
    """
    j_best = best_J_per_k(df_cd)
    k_knee = detect_knee(j_best)
    print(f"  Knee detected at k_knee = {k_knee}")

    # Admissible set  K_adm = {2, …, k_knee}  (Eq. 35)
    adm_k = sorted(k for k in j_best.index if 2 <= k <= k_knee)

    # Best partition per k ∈ K_adm
    rows = []
    for k in adm_k:
        sub  = df_cd[df_cd["k"] == k]
        best = sub.loc[sub["J_neutral"].idxmin()]
        rows.append({
            "k"          : k,
            "J_neutral"  : float(best["J_neutral"]),
            "modularity" : float(best["modularity"]),
            "coverage"   : float(best["coverage"]),
            "performance": float(best["performance"]),
            "algorithm"  : best["algorithm"],
        })
    df_adm = pd.DataFrame(rows).set_index("k")
    df_adm["S_agg"] = stability_scores(df_adm)

    # Pareto efficiency (maximise all; negate J so all are max)  (Eq. 36)
    vals  = df_adm[["J_neutral","modularity","coverage","performance"]].values.copy()
    vals[:, 0] *= -1                      # negate J_neutral
    pareto = np.ones(len(df_adm), dtype=bool)
    for i in range(len(df_adm)):
        dom = (np.all(vals >= vals[i], axis=1) &
               np.any(vals >  vals[i], axis=1))
        dom[i] = False
        if dom.any():
            pareto[i] = False

    df_pareto = df_adm[pareto]
    print(f"  Pareto-admissible k values: {list(df_pareto.index)}")

    # Final choice: maximum stability among Pareto-efficient k  (Eq. 39)
    k_star = int(df_pareto["S_agg"].idxmax())
    print(f"  Selected k* = {k_star}  "
          f"(S_agg = {df_pareto.loc[k_star, 'S_agg']:.4f}, "
          f"J = {df_pareto.loc[k_star, 'J_neutral']:.4f})")

    return k_knee, k_star, df_adm


def retrieve_best_partition(
    df_cd: pd.DataFrame,
    k_star: int,
    Gc: nx.Graph,
    agg: pd.DataFrame,
) -> list[frozenset]:
    """Re-run the best-performing algorithm/parameters for k* to recover
    the actual community partition."""
    sub = df_cd[df_cd["k"] == k_star]
    if sub.empty:
        raise ValueError(f"No partition found for k = {k_star}")
    row  = sub.loc[sub["J_neutral"].idxmin()]
    alg  = row["algorithm"]
    _s   = row.get("seed",       RANDOM_SEED)
    _r   = row.get("resolution", 1.0)
    _l   = row.get("level",      1)
    seed = RANDOM_SEED if pd.isna(_s) else int(_s)
    res  = 1.0         if pd.isna(_r) else float(_r)
    lev  = 1           if pd.isna(_l) else int(_l)

    dispatch = {
        "greedy_modularity": lambda: nx_comm.greedy_modularity_communities(Gc),
        "label_propagation": lambda: nx_comm.label_propagation_communities(Gc),
        "fluid"            : lambda: _fluid_safe(Gc, k=k_star, seed=seed),
        "kernighan_lin"    : lambda: nx_comm.kernighan_lin_bisection(Gc, seed=seed),
        "louvain"          : lambda: nx_comm.louvain_communities(Gc, resolution=res, seed=seed),
        "girvan_newman"    : lambda: _gn_at_level(Gc, lev),
    }
    return [frozenset(c) for c in dispatch[alg]()]


# =============================================================================
# 14. Figures
# =============================================================================

def _save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_zone_maps(
    gdf: gpd.GeoDataFrame,
    labels: np.ndarray,
    diff: np.ndarray,
) -> None:
    """Before / after spatial maps (Fig. 2 in paper)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    gdf_plot = gdf.copy()
    gdf_plot["d_i"] = diff
    gdf_plot.plot(ax=axes[0], column="d_i", cmap="RdBu", edgecolor="grey",
                  linewidth=0.3, legend=True,
                  legend_kwds={"label": "$d_i = \\bar{D}_i - S^{EV}_i$",
                               "shrink": 0.6})
    axes[0].set_title("Before Clustering — Zone-Level Imbalance $d_i$",
                      fontsize=11)
    axes[0].axis("off")

    gdf_c = gdf.copy()
    gdf_c["cluster"] = labels
    dissolved = gdf_c.dissolve(by="cluster", as_index=False)
    dissolved.plot(ax=axes[1], column="cluster", cmap="tab20",
                   edgecolor="black", linewidth=0.5)
    axes[1].set_title("After NBZC — Meso-Scale Cluster Regions", fontsize=11)
    axes[1].axis("off")

    fig.suptitle("Neutralisation-Based Zonal Clustering: Before vs After",
                 fontsize=13, fontweight="bold", y=1.01)
    _save(fig, "before_after_clustering.pdf")


def fig_pareto(
    results: pd.DataFrame,
    pf: pd.DataFrame,
    best: pd.Series,
) -> None:
    """Neutralisation–compactness Pareto scatter (Fig. 1 in paper)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(results["sum_abs_imbalance"], results["mean_radius_m"],
               alpha=0.2, s=10, color="grey", label="All 1008 configurations")
    ax.scatter(pf["sum_abs_imbalance"], pf["mean_radius_m"],
               color="steelblue", s=30, zorder=3, label="Pareto frontier")
    ax.scatter(best["sum_abs_imbalance"], best["mean_radius_m"],
               color="red", marker="*", s=280, zorder=5,
               label=f"Knee point (selected)")
    ax.set_xlabel(r"Neutralisation loss  $\sum_c|\Delta_c|$  (↓ better)")
    ax.set_ylabel("Compactness: mean radius (m)  (↓ better)")
    ax.set_title("Neutralisation–Compactness Trade-off (Sensitivity Analysis)")
    ax.legend(fontsize=9)
    _save(fig, "neutralization_vs_compactness_rook.pdf")


def fig_J_vs_k(
    j_best: pd.Series,
    k_knee: int,
    k_star: int,
) -> None:
    """J_best(k) curve with knee and selected k* annotated (Fig. 4)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(j_best.index, j_best.values, color="steelblue", lw=1.5,
            label=r"$J_{\mathrm{best}}(k)$")
    ax.axvline(k_knee, color="darkorange", ls="--", lw=1.3,
               label=f"Knee  $k_{{\\mathrm{{knee}}}}={k_knee}$")
    ax.axvline(k_star, color="crimson", ls="-.", lw=1.3,
               label=f"Selected  $k^*={k_star}$")
    ax.set_xlabel("Number of communities $k$")
    ax.set_ylabel(r"Best achievable $J_{\mathrm{neutral}}(k)$")
    ax.set_title("Community Count vs Neutralisation Objective")
    ax.legend(fontsize=9)
    _save(fig, "communities_and_neutralization_knee_curve.pdf")


def fig_graph_comparison(
    G_zone: nx.Graph,
    G_cluster: nx.Graph,
    gdf: gpd.GeoDataFrame,
    labels: np.ndarray,
) -> None:
    """Side-by-side zone-level and cluster-level adjacency graphs."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    pos_z = {i: (float(gdf.loc[i, "cx"]), float(gdf.loc[i, "cy"]))
             for i in G_zone.nodes()}
    nx.draw_networkx(G_zone, pos=pos_z, ax=axes[0],
                     node_size=8, node_color="steelblue",
                     edge_color="lightgrey", width=0.4, with_labels=False)
    axes[0].set_title(f"Zone Graph  "
                      f"({G_zone.number_of_nodes()} nodes, "
                      f"{G_zone.number_of_edges()} edges)", fontsize=11)
    axes[0].axis("off")

    pos_c = {}
    for c in np.unique(labels):
        mask    = labels == c
        pos_c[c] = (float(gdf.loc[mask, "cx"].mean()),
                    float(gdf.loc[mask, "cy"].mean()))
    nx.draw_networkx(G_cluster, pos=pos_c, ax=axes[1],
                     node_size=35, node_color="tomato",
                     edge_color="lightgrey", width=0.6, with_labels=False)
    axes[1].set_title(f"Cluster Graph  "
                      f"({G_cluster.number_of_nodes()} nodes, "
                      f"{G_cluster.number_of_edges()} edges)", fontsize=11)
    axes[1].axis("off")

    _save(fig, "before_after_graph_comparison.pdf")


# =============================================================================
# 15. Baseline Comparison    [Paper §Experiments]
# =============================================================================

def _contiguity_violation_rate(labels: np.ndarray, G: nx.Graph) -> float:
    """Fraction of clusters that contain a spatially disconnected sub-region."""
    unique = np.unique(labels)
    violating = sum(
        1 for c in unique
        if nx.number_connected_components(G.subgraph(
            np.where(labels == c)[0].tolist()
        )) > 1
    )
    return violating / len(unique)


def evaluate_labeling(
    method_name: str,
    labels: np.ndarray,
    gdf: gpd.GeoDataFrame,
    coords: np.ndarray,
    G_zone: nx.Graph,
) -> dict:
    """
    Compute all evaluation metrics for an arbitrary zone labeling so that
    every baseline is assessed on equal footing.

    Metrics reported
    ----------------
    mean_eta_c        : mean cluster neutralisation index  (↓ better)
    median_eta_c      : median η_c
    max_eta_c         : worst-case η_c
    mean_radius_m     : mean cluster radius in metres       (↓ better)
    pct_singletons    : % of single-zone clusters           (↓ better)
    pct_noncontiguous : % of spatially disconnected clusters (0 for NBZC by design)
    modularity_zone   : Q of partition on the zone graph    (↑ better)
    coverage_zone     : edge coverage fraction on zone graph (↑ better)
    performance_zone  : performance on zone graph           (↑ better)
    """
    cldf  = cluster_neutralisation(labels, gdf)
    radii = _cluster_radii(labels, coords)
    comms = [frozenset(np.where(labels == c)[0].tolist())
             for c in np.unique(labels)]
    pq    = partition_quality(G_zone, comms)

    return {
        "method"            : method_name,
        "n_clusters"        : int(len(cldf)),
        "mean_eta_c"        : round(float(cldf["eta_c"].mean()),   4),
        "median_eta_c"      : round(float(cldf["eta_c"].median()), 4),
        "max_eta_c"         : round(float(cldf["eta_c"].max()),    4),
        "mean_radius_m"     : round(float(np.mean(radii)),         1),
        "pct_singletons"    : round(100.0 * float((cldf["size"] == 1).mean()), 1),
        "pct_noncontiguous" : round(100.0 * _contiguity_violation_rate(labels, G_zone), 1),
        "modularity_zone"   : round(float(pq["modularity"]),  4),
        "coverage_zone"     : round(float(pq["coverage"]),    4),
        "performance_zone"  : round(float(pq["performance"]), 4),
    }


def baseline_spatial_kmeans(
    coords: np.ndarray,
    n_clusters: int,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Spatial KMeans on centroid (x, y) — no contiguity constraint, no
    behavioral objective.  Represents naive geographic grouping.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return km.fit_predict(coords).astype(int)


def baseline_modularity_zone_graph(G_zone: nx.Graph) -> np.ndarray:
    """
    Greedy modularity optimisation directly on the zone graph.
    Maximises structural cohesion; ignores BEV/ABDP entirely.
    Natural k is determined by the algorithm (not forced to k_nbzc).
    """
    comms  = list(nx_comm.greedy_modularity_communities(G_zone))
    labels = np.zeros(G_zone.number_of_nodes(), dtype=int)
    for cid, comm in enumerate(comms):
        for node in comm:
            labels[node] = cid
    return labels


def baseline_random_contiguous(
    G: nx.Graph,
    n_clusters: int,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Random contiguous regionalization via graph-Voronoi BFS from k seeds.
    Contiguity is respected but no objective is optimised — lower-bound
    baseline that isolates the contribution of the NBZC objective.
    """
    rng    = np.random.default_rng(seed)
    N      = G.number_of_nodes()
    labels = np.full(N, -1, dtype=int)

    nodes_shuffled = list(range(N))
    rng.shuffle(nodes_shuffled)
    seeds = nodes_shuffled[:n_clusters]
    for cid, s in enumerate(seeds):
        labels[s] = cid

    queues: list[list[int]] = [[s] for s in seeds]
    adj = {i: list(G.neighbors(i)) for i in range(N)}

    while any(queues):
        order = list(range(n_clusters))
        rng.shuffle(order)
        for cid in order:
            if not queues[cid]:
                continue
            node = queues[cid].pop(0)
            for nb in adj[node]:
                if labels[nb] == -1:
                    labels[nb] = cid
                    queues[cid].append(nb)

    # Assign any isolated unassigned nodes to an adjacent cluster
    for v in np.where(labels == -1)[0]:
        for nb in adj[v]:
            if labels[nb] != -1:
                labels[v] = labels[nb]
                break
    if (labels == -1).any():
        labels[labels == -1] = 0

    return labels


def baseline_skater(
    gdf: gpd.GeoDataFrame,
    n_clusters: int,
    attr_cols: list | None = None,
) -> np.ndarray | None:
    """
    SKATER spatially-constrained clustering (minimum spanning tree bisection).
    Requires: pip install spopt
    """
    try:
        from spopt.region import Skater
    except ImportError:
        warnings.warn("spopt not installed — SKATER baseline skipped.", stacklevel=2)
        return None
    if attr_cols is None:
        attr_cols = [BEV_COL, ACT_COL]
    w = Rook.from_dataframe(gdf, silence_warnings=True)
    model = Skater(gdf, w, attrs_name=attr_cols, n_clusters=n_clusters,
                   allow_partial_match=True)
    model.solve()
    return np.array(model.labels_).astype(int)


def baseline_azp(
    gdf: gpd.GeoDataFrame,
    n_clusters: int,
    attr_cols: list | None = None,
    seed: int = RANDOM_SEED,
) -> np.ndarray | None:
    """
    AZP (Automatic Zoning Procedure) spatially-constrained clustering.
    Requires: pip install spopt
    """
    try:
        from spopt.region import AZP
    except ImportError:
        warnings.warn("spopt not installed — AZP baseline skipped.", stacklevel=2)
        return None
    if attr_cols is None:
        attr_cols = [BEV_COL, ACT_COL]
    w = Rook.from_dataframe(gdf, silence_warnings=True)
    model = AZP(gdf, w, attrs_name=attr_cols, n_clusters=n_clusters,
                random_state=seed)
    model.solve()
    return np.array(model.labels_).astype(int)


def run_baseline_comparison(
    gdf: gpd.GeoDataFrame,
    G_zone: nx.Graph,
    labels_nbzc: np.ndarray,
    coords: np.ndarray,
    attr_cols: list | None = None,
) -> pd.DataFrame:
    """
    Run all baseline methods and return a comparison DataFrame.

    Baselines (all use k = number of NBZC clusters for fair comparison,
    except Greedy Modularity which finds its own natural k)
    ---------------------------------------------------------------
    1. NBZC (proposed)
    2. Spatial KMeans          — no contiguity, no behavioral signal
    3. Greedy Modularity        — contiguity-unaware, structural-only
    4. Random Contiguous        — contiguity, no objective
    5. SKATER                  — contiguity + attribute variance (spopt)
    6. AZP                     — contiguity + attribute variance (spopt)
    """
    if attr_cols is None:
        attr_cols = [BEV_COL, ACT_COL]
    k = int(len(np.unique(labels_nbzc)))
    print(f"\n[Baseline]  Comparing methods at k = {k} clusters …")

    rows = []

    rows.append(evaluate_labeling(
        "NBZC (proposed)", labels_nbzc, gdf, coords, G_zone))

    print("  Running Spatial KMeans …")
    try:
        rows.append(evaluate_labeling(
            "Spatial KMeans",
            baseline_spatial_kmeans(coords, n_clusters=k),
            gdf, coords, G_zone))
    except Exception as e:
        print(f"    KMeans failed: {e}")

    print("  Running Greedy Modularity …")
    rows.append(evaluate_labeling(
        "Greedy Modularity",
        baseline_modularity_zone_graph(G_zone),
        gdf, coords, G_zone))

    print("  Running Random Contiguous …")
    rows.append(evaluate_labeling(
        "Random Contiguous",
        baseline_random_contiguous(G_zone, n_clusters=k),
        gdf, coords, G_zone))

    print("  Running SKATER …")
    lsk = baseline_skater(gdf, n_clusters=k, attr_cols=attr_cols)
    if lsk is not None:
        rows.append(evaluate_labeling("SKATER", lsk, gdf, coords, G_zone))

    print("  Running AZP …")
    lazp = baseline_azp(gdf, n_clusters=k, attr_cols=attr_cols)
    if lazp is not None:
        rows.append(evaluate_labeling("AZP", lazp, gdf, coords, G_zone))

    df = pd.DataFrame(rows).set_index("method")
    _print_baseline_table(df)
    return df


def _print_baseline_table(df: pd.DataFrame) -> None:
    cols = [
        "n_clusters", "mean_eta_c", "median_eta_c", "max_eta_c",
        "mean_radius_m", "pct_singletons", "pct_noncontiguous",
        "modularity_zone", "coverage_zone", "performance_zone",
    ]
    labels = [
        "k", "mean η_c↓", "med η_c↓", "max η_c↓",
        "radius(m)↓", "singletons%↓", "non-contig%↓",
        "Q↑", "coverage↑", "perf↑",
    ]
    width = 13
    print("\n── Table III: Baseline Comparison ─────────────────────────────────────────")
    print(f"  {'Method':<26}" + "".join(f"{h:>{width}}" for h in labels))
    print("  " + "-" * (26 + width * len(cols)))
    for method, row in df.iterrows():
        line = f"  {method:<26}"
        for c in cols:
            line += f"{str(row.get(c, '–')):>{width}}"
        print(line)
    print()


# =============================================================================
# 16. Main Pipeline
# =============================================================================

def main() -> None:
    print("=" * 65)
    print("  EV Fast-Charging Planning — Three-Stage Framework")
    print("=" * 65)

    # ── Data ──────────────────────────────────────────────────────────────────
    gdf    = load_data()
    coords = np.column_stack([gdf["cx"].values, gdf["cy"].values])

    # ── Stage 1a: Rook graph + zone imbalance ─────────────────────────────────
    print("\n[Stage 1]  Building rook-contiguity graph …")
    G_zone = build_rook_graph(gdf)
    diff   = zone_imbalance(gdf)
    eta_i  = zone_neutralisation_index(gdf)
    print(f"  Zones: {G_zone.number_of_nodes():,}, "
          f"Edges: {G_zone.number_of_edges():,}, "
          f"Components: {nx.number_connected_components(G_zone)}")

    # ── Stage 2a: Sensitivity analysis ────────────────────────────────────────
    df_sens = run_sensitivity(G_zone, diff, coords, gdf)
    pf      = pareto_2d(df_sens)
    best    = knee_from_pareto(pf)
    print(f"\n  Selected NBZC configuration (Pareto knee):")
    for col in ["max_cluster_size","max_abs_imbalance","lambda_shape","max_radius_m",
                "n_clusters","sum_abs_imbalance","mean_radius_m"]:
        print(f"    {col:<22} = {best[col]}")
    fig_pareto(df_sens, pf, best)

    # ── Stage 2b: Apply selected configuration ────────────────────────────────
    print("\n[Stage 2]  Applying selected NBZC configuration …")
    mr = None if best["max_radius_m"] < 0 else float(best["max_radius_m"])
    labels = nbzc(
        G_zone, diff, coords,
        max_cluster_size  = int(best["max_cluster_size"]),
        max_abs_imbalance = float(best["max_abs_imbalance"]),
        lambda_shape      = float(best["lambda_shape"]),
        max_radius_m      = mr,
    )
    gdf["cluster_id"] = labels

    # ── Evaluation: neutralisation ────────────────────────────────────────────
    cldf = cluster_neutralisation(labels, gdf)
    print_neutralisation_table(eta_i, cldf)

    # ── Evaluation: graph metrics ─────────────────────────────────────────────
    G_cluster, c2z = build_cluster_graph(G_zone, labels)
    zm = graph_metrics(G_zone)
    cm = graph_metrics(G_cluster)
    print_graph_metrics_table(zm, cm)

    # Spatial and graph figures
    fig_zone_maps(gdf, labels, diff)
    fig_graph_comparison(G_zone, G_cluster, gdf, labels)

    # ── Baseline comparison ───────────────────────────────────────────────────
    df_baselines = run_baseline_comparison(gdf, G_zone, labels, coords)
    df_baselines.to_csv(OUT_DIR / "baseline_comparison.csv")

    # ── Stage 3: Community detection ──────────────────────────────────────────
    print("\n[Stage 3]  Community detection sweep …")
    agg   = cluster_agg_signals(labels, gdf)
    df_cd = community_sweep(G_cluster, agg)

    # Two-stage resolution selection
    print("\n[Stage 3]  Selecting final district resolution …")
    j_best            = best_J_per_k(df_cd)
    k_knee, k_star, df_adm = select_final_k(df_cd)
    fig_J_vs_k(j_best, k_knee, k_star)

    # Final partition and metrics
    best_partition = retrieve_best_partition(df_cd, k_star, G_cluster, agg)
    J_final        = J_neutralisation(best_partition, agg)
    q_final        = partition_quality(G_cluster, best_partition)
    best_cd_row    = df_cd[df_cd["k"] == k_star].loc[
                         df_cd[df_cd["k"] == k_star]["J_neutral"].idxmin()]

    print("\n── Final Planning District Results ──────────────────────────────")
    print(f"  k*                 = {k_star}")
    print(f"  Algorithm          = {best_cd_row['algorithm']}")
    print(f"  J_neutralisation   = {J_final:.4f}")
    print(f"  Modularity (Q)     = {q_final['modularity']:.4f}")
    print(f"  Coverage           = {q_final['coverage']:.4f}")
    print(f"  Performance        = {q_final['performance']:.4f}")
    print(f"  Stability S(k*)    = {df_adm.loc[k_star, 'S_agg']:.4f}")

    # ── Save all outputs ──────────────────────────────────────────────────────
    df_sens.to_csv(OUT_DIR / "sensitivity_results.csv", index=False)
    df_cd.to_csv(OUT_DIR / "community_detection_results.csv", index=False)
    df_adm.to_csv(OUT_DIR / "admissible_k_analysis.csv")
    gdf.drop(columns=["geometry"]).to_csv(
        OUT_DIR / "zone_cluster_assignments.csv", index=False)

    print(f"\nAll outputs saved to: {OUT_DIR.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
