# infer5.py —— Generate a similar DAG using the reference graph; time is quantized into integers by "node" and the total duration is calibrated
from __future__ import annotations
import argparse, time, pickle, json, math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import networkx as nx
import torch
from networkx.readwrite import json_graph

from config5 import DEVICE, CHECKPOINT_DIR, NORM_N, NORM_E, NORM_L, NORM_W, NORM_T
from models5 import StructureToGraphDecoder5
from utils5 import topological_layers


TIME_KEYS = ["critical_time", "time", "weight", "t", "C", "label"]

def _get_edge_time(d: dict, default=1.0) -> float:
    for k in TIME_KEYS:
        if k in d:
            try:
                return float(d[k])
            except Exception:
                pass
    return float(default)

def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.ndarray,)):  return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

# ---------- read/write ----------
def read_any_gpickle(p: Path):
    try:
        from networkx.readwrite.gpickle import read_gpickle as _read
        return _read(p)
    except Exception:
        with open(p, "rb") as f:
            return pickle.load(f)

def write_any_gpickle(G: nx.DiGraph, p: Path):
    try:
        from networkx.readwrite.gpickle import write_gpickle as _write
        _write(G, p)
    except Exception:
        with open(p, "wb") as f:
            pickle.dump(G, f)

def save_graph(G: nx.DiGraph, out_dir: Path, name_prefix: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{name_prefix}_{int(time.time())}"
    p_gpk = out_dir / f"{base}.gpickle"
    p_json = out_dir / f"{base}.json"
    write_any_gpickle(G, p_gpk)
    data = json_graph.node_link_data(G, edges="links")
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=_json_default)
    print("Saved:", p_gpk)
    print("Saved JSON:", p_json)
    return p_gpk

# ---------- Model & Reference Image Statistics ----------
def load_model(ckpt_name: str = "decoder_best1.pt") -> StructureToGraphDecoder5:
    model = StructureToGraphDecoder5().to(DEVICE)
    ckpt = torch.load(CHECKPOINT_DIR / ckpt_name, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

def s_from_graph(G: nx.DiGraph) -> Tuple[torch.Tensor, List[int], int, float]:
    if not nx.is_directed(G):
        G = G.to_directed()
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("not DAG")
    layers = topological_layers(G)
    widths = [len(Li) for Li in layers]
    N = G.number_of_nodes()
    E = G.number_of_edges()
    L = len(widths)
    W = max(widths) if widths else 0
    T_total = sum(_get_edge_time(d, 1.0) for _, _, d in G.edges(data=True))
    s = torch.tensor([N / NORM_N, E / NORM_E, L / NORM_L, W / NORM_W, T_total / NORM_T],
                     dtype=torch.float32, device=DEVICE)
    return s, widths, E, T_total

# ---------- Structural side selection ----------
def _layer_index_from_widths(widths: List[int]) -> np.ndarray:
    idx = []
    for li, w in enumerate(widths):
        idx += [li] * w
    return np.asarray(idx, dtype=np.int32)

def _structured_select_edges(prob: torch.Tensor, widths: List[int],
                             E_target: Optional[int],
                             neighbor_only: bool = True,
                             seed: int = 0) -> Tuple[np.ndarray, Dict[str, int]]:
    rng = np.random.default_rng(seed)
    prob_np = prob.detach().cpu().numpy()
    N = prob_np.shape[0]
    idx = _layer_index_from_widths(widths)
    L = int(idx.max()) + 1 if N > 0 else 0
    layer_nodes = [np.where(idx == li)[0].tolist() for li in range(L)]
    selected = np.zeros((N, N), dtype=bool)

    def prev_nodes(li: int):
        return layer_nodes[li-1] if (li-1) >= 0 else []
    def next_nodes(li: int):
        return layer_nodes[li+1] if (li+1) < L else []

    # 1) Degree coverage: at least 1 entry for non-first layers and at least 1 exit for non-last layers (adjacent layers)
    cov_added = 0
    for li in range(1, L):
        pnodes = prev_nodes(li)
        for j in layer_nodes[li]:
            if pnodes:
                i_best = max(pnodes, key=lambda i: prob_np[i, j])
                if not selected[i_best, j]:
                    selected[i_best, j] = True; cov_added += 1
    for li in range(0, L-1):
        nnodes = next_nodes(li)
        for i in layer_nodes[li]:
            if nnodes:
                j_best = max(nnodes, key=lambda j: prob_np[i, j])
                if not selected[i, j_best]:
                    selected[i, j_best] = True; cov_added += 1

    # 2) Top-K
    already = int(selected.sum())
    legal_mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(N):
            legal_mask[i, j] = (idx[j] == idx[i] + 1) if neighbor_only else (idx[j] > idx[i])
    remain_mask = legal_mask & (~selected)
    topk_added = 0
    if E_target is not None:
        K = max(0, E_target - already)
        if K > 0 and remain_mask.any():
            pos = np.argwhere(remain_mask)
            scores = prob_np[remain_mask]
            K = min(K, len(scores))
            if K > 0:
                topk_idx = np.argpartition(-scores, K - 1)[:K]
                for ii in topk_idx:
                    i, j = pos[ii]
                    if not selected[i, j]:
                        selected[i, j] = True; topk_added += 1
    return selected, {"cov_added": cov_added, "topk_added": topk_added}

# ---------- Composition ----------
def _build_graph_from_selected(selected: np.ndarray, widths: List[int], time_mat=None) -> nx.DiGraph:
    order = []
    for li, w in enumerate(widths):
        for pj in range(w):
            order.append((li, pj))
    G = nx.DiGraph()
    for i, (li, pj) in enumerate(order):
        G.add_node(i, layer=int(li), pos=int(pj))

    T = None
    if time_mat is not None:
        if isinstance(time_mat, np.ndarray):
            T = time_mat
        else:
            try:
                T = time_mat.detach().cpu().numpy()
            except Exception:
                T = np.asarray(time_mat)

    ii, jj = np.where(selected)
    for i, j in zip(ii, jj):
        t = float(T[i, j]) if T is not None else float(np.random.uniform(1.0, 3.0))
        G.add_edge(int(i), int(j), critical_time=t)
    return G

def _enforce_single_source_sink(G: nx.DiGraph) -> tuple[nx.DiGraph, int]:
    G = G.copy(); added = 0
    # Requires layer information; if not, falls back to topo ordering inference
    layer = {n: G.nodes[n].get("layer", 0) for n in G.nodes()}
    Lmax = max(layer.values()) if layer else 0

    # 1) Merge sources: For each additional source s, try to pick one from the previous layer
    sources = [n for n in G if G.in_degree(n)==0]
    if len(sources) > 1:
        main = min(sources)  
        for s in sources:
            if s == main:
                continue
            ls = layer.get(s, 0)
            # If s is not in level 0: choose one from level ls-1 (preferring nodes with more outgoing edges)
            cands = [u for u in G.nodes() if layer.get(u, -1) == ls-1]
            if cands:
                # Select the u with the largest out-degree (can also be replaced by the u with the highest probability)
                u = max(cands, key=lambda x: G.out_degree(x))
                if not G.has_edge(u, s):
                    G.add_edge(u, s, critical_time=float(np.random.uniform(1.0,3.0))); added += 1
            else:
                # Without the previous layer, it can only degenerate to main→s (ΔL>1 may occur, but rarely)
                if not G.has_edge(main, s):
                    G.add_edge(main, s, critical_time=float(np.random.uniform(1.0,3.0))); added += 1

    # 2) Merge sinks: For each additional sink, try to connect it to the next layer v（t→v，ΔL=1）
    sinks = [n for n in G if G.out_degree(n)==0]
    if len(sinks) > 1:
        main = min(sinks)
        for t in sinks:
            if t == main:
                continue
            lt = layer.get(t, 0)
            cands = [v for v in G.nodes() if layer.get(v, -1) == lt+1]
            if cands:
                v = max(cands, key=lambda x: G.in_degree(x))
                if not G.has_edge(t, v):
                    G.add_edge(t, v, critical_time=float(np.random.uniform(1.0,3.0))); added += 1
            else:
                # There is no next level, it can only degenerate to t→main
                if not G.has_edge(t, main):
                    G.add_edge(t, main, critical_time=float(np.random.uniform(1.0,3.0))); added += 1

    assert nx.is_directed_acyclic_graph(G)
    return G, added


# ---------- Print Time Report ----------
def print_time_report(G: nx.DiGraph, ref_total_T: float | None=None, eps: float=1e-6):
    ts = [d["critical_time"] for _,_,d in G.edges(data=True) if "critical_time" in d]
    if not ts:
        print("[time] There is no edge on the generated graph critical_time"); return
    total = float(sum(ts)); mean = total/len(ts)
    print(f"[time] edges_with_time={len(ts)}/{G.number_of_edges()} | total={total:.3f} | "
          f"min={min(ts):.3f} | max={max(ts):.3f} | mean={mean:.3f}")
    if ref_total_T is not None:
        diff = total - ref_total_T
        rel  = diff / ref_total_T * 100.0 if ref_total_T>0 else float("nan")
        print(f"[time] vs REF total_T={ref_total_T:.3f} -> diff={diff:+.3f} ({rel:+.2f}%)")
    non_uniform = []
    for u in G.nodes():
        ts_u = [d["critical_time"] for _,_,d in G.out_edges(u, data=True) if "critical_time" in d]
        if len(ts_u)>=2 and (max(ts_u)-min(ts_u) > eps):
            non_uniform.append((u, min(ts_u), max(ts_u)))
    print(f"[time] nodes with non-uniform outgoing times: {len(non_uniform)}")
    if non_uniform[:5]: print("       examples:", non_uniform[:5])

# ---------- Integerization + Total Calibration ----------
def generate_like(ref_path: Path, ckpt_name: str, neighbor_only: bool, seed: int,
                  integerize_time: bool = True) -> nx.DiGraph:
    refG = read_any_gpickle(ref_path)
    s, widths, E_target, T_target = s_from_graph(refG)

    # The allowed integer range is derived from the reference graph
    ref_ts = [ _get_edge_time(d, 1.0) for _,_,d in refG.edges(data=True) ]
    allowed_min = int(math.floor(min(ref_ts))) if ref_ts else 1
    allowed_max = int(math.ceil(max(ref_ts)))  if ref_ts else 20
    allowed = np.arange(allowed_min, allowed_max+1, dtype=np.int32)

    model = load_model(ckpt_name)
    with torch.no_grad():
        logits, time_mat, widths_used = model(s, widths=widths)
        prob = torch.sigmoid(logits)
        valid = torch.isfinite(logits)
        prob = torch.where(valid, prob, torch.zeros_like(prob))  

   
    selected, stats = _structured_select_edges(prob, widths_used, E_target=E_target,
                                               neighbor_only=neighbor_only, seed=seed)

   
    if integerize_time and (time_mat is not None):
        T = time_mat.detach().cpu().numpy()
        ii, jj = np.where(selected)
        N = T.shape[0]
       ）
        row_sum = np.bincount(ii, weights=T[ii, jj], minlength=N).astype(np.float32)
        row_cnt = np.bincount(ii, minlength=N).astype(np.int32)
        r_pred = np.where(row_cnt > 0, row_sum / np.maximum(row_cnt, 1), 0.0)

        idx_near = np.argmin(np.abs(allowed.reshape(-1,1) - r_pred.reshape(1,-1)), axis=0)
        r_int = allowed[idx_near].astype(np.int32)
        outdeg = selected.sum(axis=1).astype(np.int32)
        cur = int((outdeg * r_int).sum())
        target = int(round(T_target))
        diff = target - cur
        if diff != 0:
            mn, mx = allowed.min(), allowed.max()
            idx_sorted = np.argsort(-outdeg)  
            tries = 0
            while diff != 0:
                changed = False
                if diff > 0:
                    # Need to increase the total duration: find a node that can be +1
                    for i in idx_sorted:
                        if outdeg[i] > 0 and r_int[i] < mx:
                            r_int[i] += 1
                            diff -= int(outdeg[i])
                            changed = True
                            break
                else:  # diff < 0
                    # Need to reduce the total time: find a node that can be -1
                    for i in idx_sorted:
                        if outdeg[i] > 0 and r_int[i] > mn:
                            r_int[i] -= 1
                            diff += int(outdeg[i])
                            changed = True
                            break
                if not changed:
                    # It has reached the border and cannot be adjusted anymore
                    break
                tries += 1
                if tries > 200000:  
                    break
            if diff != 0:
                print(f"[warn] total_T could not be matched exactly (residual {diff}).")

        # Write the integer r_i back to the selected edge
        T_eq = np.zeros_like(T, dtype=np.float32)
        if ii.size > 0:
            T_eq[ii, jj] = r_int[ii].astype(np.float32)
        time_mat = torch.tensor(T_eq, device=time_mat.device, dtype=time_mat.dtype)

    G = _build_graph_from_selected(selected, widths_used, time_mat=time_mat)
    G, post_added = _enforce_single_source_sink(G)

    print(f"[like] L={len(widths)}, widths={widths}")
    print(f"[like] edges={G.number_of_edges()} (target={E_target}) | cov_added={stats['cov_added']} "
          f"| topk_added={stats['topk_added']} | post_added={post_added}")
    print_time_report(G, ref_total_T=T_target)
    return G

def report_graph(G: nx.DiGraph):
    is_dag = nx.is_directed_acyclic_graph(G)
    sources = [n for n in G if G.in_degree(n)==0]
    sinks   = [n for n in G if G.out_degree(n)==0]
    isolated = [n for n in G if G.degree(n)==0]
    print(f"DAG? {is_dag} | N={G.number_of_nodes()}, E={G.number_of_edges()} "
          f"| sources={len(sources)}, sinks={len(sinks)}, isolated={len(isolated)}")


if __name__ == "__main__":
    # Reference Image
    REF_PATH = Path(r"E:/PythonProject5/data/small/Tau_0_Tau_1.gpickle")

   
    CKPT         = "decoder_best1.pt"  # model 
    OUTDIR       = Path("generated5")
    PREFIX       = "gen_like"
    NEIGHBOR_ONLY = True
    SEED         = 0

    G = generate_like(REF_PATH, CKPT, NEIGHBOR_ONLY, SEED, integerize_time=True)
    report_graph(G)
    save_graph(G, OUTDIR, PREFIX)

