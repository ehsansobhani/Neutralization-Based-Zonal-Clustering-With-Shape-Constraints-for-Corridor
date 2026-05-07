"""Run baseline comparison and print results for paper Table III."""
import sys
sys.path.insert(0, r"e:\clustering_journal")

import warnings, itertools
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import networkx.algorithms.community as nx_comm
from libpysal.weights import Rook
from collections import defaultdict

SHP_PATH = Path(r"e:\clustering_journal\arterial_with_bevD_activityD_for_clustering\arterial_with_bevD_activityD_for_clustering.shp")
BEV_COL  = "BEV_density_norm"
ACT_COL  = "S_norm_con"
CRS_METRIC = "EPSG:32617"
RANDOM_SEED = 42

# ── Load data ──────────────────────────────────────────────────────────────
gdf = gpd.read_file(SHP_PATH)
if gdf.crs is None or gdf.crs.to_epsg() != 32617:
    gdf = gdf.to_crs(CRS_METRIC)
if "BEV_densit" in gdf.columns and BEV_COL not in gdf.columns:
    gdf = gdf.rename(columns={"BEV_densit": BEV_COL})
gdf = gdf.reset_index(drop=True)
gdf["zone_area_m2"] = gdf.geometry.area
gdf["cx"] = gdf.geometry.centroid.x
gdf["cy"] = gdf.geometry.centroid.y
coords = np.column_stack([gdf["cx"].values, gdf["cy"].values])
print(f"Loaded {len(gdf)} zones | cols: {[c for c in gdf.columns if c != 'geometry']}")

# ── Rook graph ─────────────────────────────────────────────────────────────
w = Rook.from_dataframe(gdf, silence_warnings=True)
G_zone = nx.Graph()
G_zone.add_nodes_from(range(len(gdf)))
for i, nbrs in w.neighbors.items():
    for j in nbrs:
        if i < j:
            G_zone.add_edge(i, j)
print(f"Zone graph: {G_zone.number_of_nodes()} nodes, {G_zone.number_of_edges()} edges")

# ── NBZC with known best parameters ───────────────────────────────────────
diff = (gdf[ACT_COL].values - gdf[BEV_COL].values).astype(float)

def nbzc(G, diff, coords, max_cluster_size=8, max_abs_imbalance=0.5,
         lambda_shape=0.001, max_radius_m=2000.0):
    N = len(diff)
    labels = np.full(N, -1, dtype=int)
    cluster_id = 0
    unassigned = set(range(N))
    adj = {i: set(G.neighbors(i)) for i in range(N)}
    while unassigned:
        seed = max(unassigned, key=lambda i: abs(diff[i]))
        cluster = {seed}
        cluster_sum = float(diff[seed])
        unassigned.remove(seed)
        while True:
            F = set().union(*(adj[i] for i in cluster)) & unassigned
            if not F:
                break
            best_j, best_score = None, float("inf")
            for j in F:
                new_sum = cluster_sum + diff[j]
                pts = coords[list(cluster | {j})]
                centroid = pts.mean(axis=0)
                radius = float(np.max(np.linalg.norm(pts - centroid, axis=1)))
                if max_radius_m is not None and radius > max_radius_m:
                    continue
                score = abs(new_sum) + lambda_shape * radius
                if score < best_score:
                    best_score, best_j = score, j
            if best_j is None:
                break
            new_sum = cluster_sum + diff[best_j]
            if len(cluster) >= max_cluster_size:
                break
            if abs(new_sum) > max_abs_imbalance:
                break
            if len(cluster) >= 2 and abs(new_sum) > abs(cluster_sum):
                break
            cluster.add(best_j)
            cluster_sum = new_sum
            unassigned.discard(best_j)
        for node in cluster:
            labels[node] = cluster_id
        cluster_id += 1
    return labels

print("Running NBZC …")
labels = nbzc(G_zone, diff, coords,
              max_cluster_size=8, max_abs_imbalance=0.5,
              lambda_shape=0.001, max_radius_m=2000.0)
print(f"NBZC produced {len(np.unique(labels))} clusters")

# ── Evaluation helpers ─────────────────────────────────────────────────────
def cluster_neutralisation(labels, gdf):
    act = gdf[ACT_COL].values
    bev = gdf[BEV_COL].values
    d = act - bev
    rows = []
    for c in np.unique(labels):
        mask  = labels == c
        delta = float(d[mask].sum())
        T_c   = float((act[mask] + bev[mask]).sum())
        eta   = abs(delta) / T_c if T_c > 0 else 0.0
        rows.append({"cluster": int(c), "size": int(mask.sum()),
                     "delta_c": delta, "T_c": T_c, "eta_c": eta})
    return pd.DataFrame(rows).set_index("cluster")

def _cluster_radii(labels, coords):
    radii = []
    for c in np.unique(labels):
        pts = coords[labels == c]
        if len(pts) <= 1:
            radii.append(0.0)
        else:
            cen = pts.mean(axis=0)
            radii.append(float(np.max(np.linalg.norm(pts - cen, axis=1))))
    return radii

def partition_quality(G, communities):
    comm_list = [list(c) for c in communities]
    node_to_comm = {n: i for i, c in enumerate(comm_list) for n in c}
    intra = sum(1 for u, v in G.edges()
                if node_to_comm.get(u) == node_to_comm.get(v))
    M = G.number_of_edges()
    cov = intra / M if M > 0 else 0.0
    try:
        mod = nx_comm.modularity(G, [frozenset(c) for c in comm_list])
    except Exception:
        mod = float("nan")
    n = G.number_of_nodes()
    tn = sum(1 for u, v in itertools.combinations(G.nodes(), 2)
             if not G.has_edge(u, v)
             and node_to_comm.get(u) != node_to_comm.get(v))
    perf = (intra + tn) / (n * (n - 1) / 2) if n > 1 else 0.0
    return {"modularity": mod, "coverage": cov, "performance": perf}

def _contiguity_violation_rate(labels, G):
    unique = np.unique(labels)
    violating = sum(
        1 for c in unique
        if nx.number_connected_components(
            G.subgraph(np.where(labels == c)[0].tolist())
        ) > 1
    )
    return violating / len(unique)

def evaluate_labeling(method_name, labels, gdf, coords, G_zone):
    cldf  = cluster_neutralisation(labels, gdf)
    radii = _cluster_radii(labels, coords)
    comms = [frozenset(np.where(labels == c)[0].tolist()) for c in np.unique(labels)]
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

# ── Baseline methods ───────────────────────────────────────────────────────
def baseline_spatial_kmeans(coords, n_clusters, seed=RANDOM_SEED):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(coords).astype(int)

def baseline_modularity_zone_graph(G_zone):
    comms  = list(nx_comm.greedy_modularity_communities(G_zone))
    lbl    = np.zeros(G_zone.number_of_nodes(), dtype=int)
    for cid, comm in enumerate(comms):
        for node in comm:
            lbl[node] = cid
    return lbl

def baseline_random_contiguous(G, n_clusters, seed=RANDOM_SEED):
    rng   = np.random.default_rng(seed)
    N     = G.number_of_nodes()
    lbl   = np.full(N, -1, dtype=int)
    nodes = list(range(N)); rng.shuffle(nodes)
    seeds = nodes[:n_clusters]
    for cid, s in enumerate(seeds):
        lbl[s] = cid
    queues = [[s] for s in seeds]
    adj    = {i: list(G.neighbors(i)) for i in range(N)}
    while any(queues):
        order = list(range(n_clusters)); rng.shuffle(order)
        for cid in order:
            if not queues[cid]: continue
            node = queues[cid].pop(0)
            for nb in adj[node]:
                if lbl[nb] == -1:
                    lbl[nb] = cid; queues[cid].append(nb)
    for v in np.where(lbl == -1)[0]:
        for nb in adj[v]:
            if lbl[nb] != -1: lbl[v] = lbl[nb]; break
    if (lbl == -1).any(): lbl[lbl == -1] = 0
    return lbl

def baseline_skater(gdf, n_clusters):
    from spopt.region import Skater
    w = Rook.from_dataframe(gdf, silence_warnings=True)
    m = Skater(gdf, w, attrs_name=[BEV_COL, ACT_COL],
               n_clusters=n_clusters, allow_partial_match=True)
    m.solve()
    return np.array(m.labels_).astype(int)

def baseline_azp(gdf, n_clusters):
    from spopt.region import AZP
    w = Rook.from_dataframe(gdf, silence_warnings=True)
    m = AZP(gdf, w, attrs_name=[BEV_COL, ACT_COL],
            n_clusters=n_clusters, random_state=RANDOM_SEED)
    m.solve()
    return np.array(m.labels_).astype(int)

# ── Run all ────────────────────────────────────────────────────────────────
k = int(len(np.unique(labels)))
print(f"\nRunning all baselines at k = {k} …")
rows = []

print("  [1/6] NBZC (proposed)")
rows.append(evaluate_labeling("NBZC (proposed)",  labels, gdf, coords, G_zone))

print("  [2/6] Spatial KMeans")
rows.append(evaluate_labeling("Spatial KMeans",   baseline_spatial_kmeans(coords, k), gdf, coords, G_zone))

print("  [3/6] Greedy Modularity")
rows.append(evaluate_labeling("Greedy Modularity", baseline_modularity_zone_graph(G_zone), gdf, coords, G_zone))

print("  [4/6] Random Contiguous")
rows.append(evaluate_labeling("Random Contiguous", baseline_random_contiguous(G_zone, k), gdf, coords, G_zone))

print("  [5/6] SKATER (subprocess-isolated)")
import subprocess, json, tempfile, os
skater_script = f"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, geopandas as gpd
from libpysal.weights import Rook
from spopt.region import Skater
SHP = r"{SHP_PATH}"
BEV = "{BEV_COL}"; ACT = "{ACT_COL}"
gdf = gpd.read_file(SHP)
if gdf.crs is None or gdf.crs.to_epsg() != 32617:
    gdf = gdf.to_crs("EPSG:32617")
if "BEV_densit" in gdf.columns and BEV not in gdf.columns:
    gdf = gdf.rename(columns={{"BEV_densit": BEV}})
gdf = gdf.reset_index(drop=True)
gdf["zone_area_m2"] = gdf.geometry.area
gdf["cx"] = gdf.geometry.centroid.x; gdf["cy"] = gdf.geometry.centroid.y
w = Rook.from_dataframe(gdf, silence_warnings=True)
m = Skater(gdf, w, attrs_name=[BEV, ACT], n_clusters={k}, allow_partial_match=True)
m.solve()
print(",".join(str(x) for x in m.labels_))
"""
try:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(skater_script); tmp = f.name
    res = subprocess.run(["python", tmp], capture_output=True, text=True, timeout=120)
    os.unlink(tmp)
    if res.returncode == 0 and res.stdout.strip():
        lsk = np.array([int(x) for x in res.stdout.strip().split(",")], dtype=int)
        rows.append(evaluate_labeling("SKATER", lsk, gdf, coords, G_zone))
        print("    SKATER done")
    else:
        print(f"    SKATER failed (rc={res.returncode}): {res.stderr[:200]}")
except Exception as e:
    print(f"    SKATER error: {e}")

print("  [6/6] AZP (subprocess-isolated)")
azp_script = f"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, geopandas as gpd
from libpysal.weights import Rook
from spopt.region import AZP
SHP = r"{SHP_PATH}"
BEV = "{BEV_COL}"; ACT = "{ACT_COL}"
gdf = gpd.read_file(SHP)
if gdf.crs is None or gdf.crs.to_epsg() != 32617:
    gdf = gdf.to_crs("EPSG:32617")
if "BEV_densit" in gdf.columns and BEV not in gdf.columns:
    gdf = gdf.rename(columns={{"BEV_densit": BEV}})
gdf = gdf.reset_index(drop=True)
gdf["zone_area_m2"] = gdf.geometry.area
gdf["cx"] = gdf.geometry.centroid.x; gdf["cy"] = gdf.geometry.centroid.y
w = Rook.from_dataframe(gdf, silence_warnings=True)
m = AZP(gdf, w, attrs_name=[BEV, ACT], n_clusters={k}, random_state=42)
m.solve()
print(",".join(str(x) for x in m.labels_))
"""
try:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(azp_script); tmp = f.name
    res = subprocess.run(["python", tmp], capture_output=True, text=True, timeout=120)
    os.unlink(tmp)
    if res.returncode == 0 and res.stdout.strip():
        lazp = np.array([int(x) for x in res.stdout.strip().split(",")], dtype=int)
        rows.append(evaluate_labeling("AZP", lazp, gdf, coords, G_zone))
        print("    AZP done")
    else:
        print(f"    AZP failed (rc={res.returncode}): {res.stderr[:200]}")
except Exception as e:
    print(f"    AZP error: {e}")

df = pd.DataFrame(rows).set_index("method")
df.to_csv(r"e:\clustering_journal\baseline_comparison.csv")

print("\n" + "="*100)
print(df.to_string())
print("="*100)
print("\nSaved: baseline_comparison.csv")
