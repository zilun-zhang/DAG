# batch_infer_to_csv.py — Batch-generate similar DAGs and export CSV stats
from __future__ import annotations
import argparse, csv, json, gzip, pickle, time, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import networkx as nx
from networkx.readwrite import json_graph

# 项目内模块（保持和你的工程一致）
from config5 import DEVICE, CHECKPOINT_DIR, NORM_N, NORM_E, NORM_L, NORM_W, NORM_T
from models5 import StructureToGraphDecoder5
from utils5 import topological_layers

TIME_KEYS = ["critical_time", "time", "weight", "t", "C", "label"]

# ===== 在文件靠前位置，加入默认配置（可按需改路径）=====
DEFAULTS = {
    # 参考图目录（你之前用的小数据目录）
    "ref_dir": r"E:\DAG\data\gpickle1",
    # 扫描后缀；会额外把 *.json / *.gpickle.gz / *.json.gz 也一起扫
    "pattern": "*.gpickle",
    # 模型 ckpt
    "ckpt": "decoder_best1.pt",
    # 输出目录 & CSV 文件名
    "outdir": r"E:\DAG\src\generated5",
    "csv": "batch_report.csv",
    # 结构/随机性/时间/后处理开关
    "neighbor_only": False,          # True=只允许相邻层（禁长跳）
    "seed": 0,
    "no_equalize_row_time": False,   # True=关闭“行内等化”
    "no_scale_total_time": False,    # True=关闭“总量缩放到参考”
    "no_post_fix": False,            # True=关闭“单源/单汇”后处理
    "save_pre": True,                # 保存后处理前的图
    "limit": 0                       # 0=全量；>0 只处理前K个
}


# ============ IO helpers ============
def edge_time(d: dict, default=0.0) -> float:
    for k in TIME_KEYS:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                pass
    return float(default)

def read_any(path: Path) -> nx.DiGraph:
    # 读取 gpickle / json(.gz)，兼容老版本 networkx
    path = Path(str(path).strip())
    name = path.name.lower()
    try:
        if name.endswith(".gpickle"):
            read = getattr(nx, "read_gpickle", None)
            if read is not None:
                return read(path)
            with open(path, "rb") as f:
                return pickle.load(f)
        if name.endswith(".gpickle.gz"):
            with gzip.open(path, "rb") as f:
                read = getattr(nx, "read_gpickle", None)
                if read is not None:
                    return read(f)
                return pickle.load(f)
        if name.endswith(".json") or name.endswith(".json.gz"):
            if name.endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    data = json.loads(f.read())
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
            return json_graph.node_link_graph(data, directed=True, multigraph=False)
    except Exception as e:
        raise RuntimeError(f"read error for {path}: {e}") from e
    raise ValueError(f"Unsupported file type: {path}")

def write_any_gpickle(G: nx.DiGraph, p: Path):
    try:
        from networkx.readwrite import gpickle as gx
        gx.write_gpickle(G, p)
    except Exception:
        with open(p, "wb") as f:
            pickle.dump(G, f)

def save_graph(G: nx.DiGraph, out_dir: Path, name_prefix: str) -> Tuple[Path, Path]:
    # 保存为 .gpickle 和 .json
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    p_g = out_dir / f"{name_prefix}_{stamp}.gpickle"
    p_j = out_dir / f"{name_prefix}_{stamp}.json"
    write_any_gpickle(G, p_g)
    data = json_graph.node_link_data(G, link="links")
    p_j.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p_g, p_j

# ============ Stats helpers ============
def layers_by_longest(G: nx.DiGraph) -> Dict[int, int]:
    order = list(nx.topological_sort(G))
    L = {u: 0 for u in order}
    for u in order:
        for v in G.successors(u):
            L[v] = max(L[v], L[u] + 1)
    return L

def deltaL_hist(G: nx.DiGraph) -> Dict[int, int]:
    layer = {n: G.nodes[n].get("layer") for n in G.nodes()}
    if any(v is None for v in layer.values()):
        layer = layers_by_longest(G)
    hist: Dict[int, int] = {}
    for u, v in G.edges():
        dv = layer[v] - layer[u]
        hist[dv] = hist.get(dv, 0) + 1
    return dict(sorted(hist.items()))

def degree_coverage_missing(G: nx.DiGraph) -> Tuple[int, int]:
    # (非首层缺入度, 非末层缺出度)
    L = layers_by_longest(G)
    Lmax = max(L.values()) if L else 0
    no_in = 0
    no_out = 0
    for n, lv in L.items():
        if lv > 0 and G.in_degree(n) == 0:
            no_in += 1
        if lv < Lmax and G.out_degree(n) == 0:
            no_out += 1
    return no_in, no_out

def single_src_sink_counts(G: nx.DiGraph) -> Tuple[int, int]:
    srcs = sum(1 for n in G if G.in_degree(n) == 0)
    sinks = sum(1 for n in G if G.out_degree(n) == 0)
    return srcs, sinks

def time_stats(G: nx.DiGraph) -> Tuple[float, float, float, float]:
    ts = [edge_time(d) for _, _, d in G.edges(data=True)]
    if not ts:
        return 0.0, 0.0, 0.0, 0.0
    return float(sum(ts)), float(sum(ts) / len(ts)), float(min(ts)), float(max(ts))

def longest_path_time(G: nx.DiGraph) -> float:
    order = list(nx.topological_sort(G))
    dist = {u: 0.0 for u in order}
    for u in order:
        for v in G.successors(u):
            w = edge_time(G[u][v], 0.0)
            dist[v] = max(dist[v], dist[u] + w)
    return max(dist.values()) if dist else 0.0

# ============ Model & s-vector ============
def load_model(ckpt_name: str) -> StructureToGraphDecoder5:
    model = StructureToGraphDecoder5().to(DEVICE)
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[load] checkpoint: {ckpt_path}")
    return model

def s_from_graph(G: nx.DiGraph) -> Tuple[torch.Tensor, List[int], int, float]:
    if not nx.is_directed(G):
        G = G.to_directed()
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("reference graph is not a DAG")
    layers = topological_layers(G)
    widths = [len(Li) for Li in layers]
    N, E = G.number_of_nodes(), G.number_of_edges()
    L = len(widths)
    W = max(widths) if widths else 0
    T_total = sum(edge_time(d, 1.0) for _, _, d in G.edges(data=True))
    s = torch.tensor([N/NORM_N, E/NORM_E, L/NORM_L, W/NORM_W, T_total/NORM_T],
                     dtype=torch.float32, device=DEVICE)
    return s, widths, E, float(T_total)

# ============ Generation (same logic as infer5.py) ============
def structured_select(prob: torch.Tensor, widths: List[int], E_target: int,
                      neighbor_only: bool, seed: int = 0) -> Tuple[np.ndarray, Dict[str, int]]:
    P = prob.detach().cpu().numpy()
    N = P.shape[0]
    # 层号数组
    layer: List[int] = []
    for li, w in enumerate(widths):
        layer += [li]*w
    layer = np.asarray(layer, dtype=np.int32)
    Lmax = int(layer.max()) if layer.size else 0

    selected = np.zeros((N, N), dtype=bool)
    cov_added = 0

    # 度覆盖：相邻层
    for j in range(N):
        lj = layer[j]
        if lj == 0: continue
        prev = np.where(layer == lj-1)[0]
        if prev.size == 0: continue
        i_best = prev[np.argmax(P[prev, j])]
        if not selected[i_best, j]:
            selected[i_best, j] = True; cov_added += 1

    for i in range(N):
        li = layer[i]
        if li == Lmax: continue
        nxt = np.where(layer == li+1)[0]
        if nxt.size == 0: continue
        j_best = nxt[np.argmax(P[i, nxt])]
        if not selected[i, j_best]:
            selected[i, j_best] = True; cov_added += 1

    # Top-K 补足到目标边数
    legal = np.zeros_like(P, dtype=bool)
    for i in range(N):
        for j in range(N):
            if layer[j] > layer[i]:
                legal[i, j] = (layer[j]-layer[i] == 1) if neighbor_only else True

    remain = legal & (~selected)
    need = max(0, E_target - int(selected.sum()))
    if need > 0:
        scores = P.copy()
        scores[~remain] = -1.0
        flat = np.argpartition(scores.ravel(), -need)[-need:]
        ii, jj = np.unravel_index(flat, scores.shape)
        order = np.argsort(scores[ii, jj])[::-1]
        for k in order:
            selected[ii[k], jj[k]] = True

    return selected, {"cov_added": cov_added, "topk_added": int(selected.sum()) - cov_added}

def build_graph(selected: np.ndarray, widths: List[int], time_mat: Optional[torch.Tensor]) -> nx.DiGraph:
    N = selected.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    layers: List[int] = []
    for li, w in enumerate(widths):
        layers += [li]*w
    for n, lv in enumerate(layers):
        G.nodes[n]["layer"] = int(lv)
    T = time_mat.detach().cpu().numpy() if time_mat is not None else None
    ii, jj = np.where(selected)
    for i, j in zip(ii.tolist(), jj.tolist()):
        t = float(T[i, j]) if T is not None else 1.0
        G.add_edge(i, j, critical_time=t)
    assert nx.is_directed_acyclic_graph(G)
    return G

def enforce_single_src_sink(G: nx.DiGraph) -> Tuple[nx.DiGraph, int]:
    # 只加不删：把多余源并到最小编号源，把多余汇并到最小编号汇
    G = G.copy(); added = 0
    srcs = [n for n in G if G.in_degree(n) == 0]
    sinks = [n for n in G if G.out_degree(n) == 0]
    if len(srcs) > 1:
        main = min(srcs)
        for s in srcs:
            if s == main: continue
            if not G.has_edge(main, s):
                G.add_edge(main, s, critical_time=float(np.random.uniform(1.0, 3.0))); added += 1
    if len(sinks) > 1:
        main = min(sinks)
        for t in sinks:
            if t == main: continue
            if not G.has_edge(t, main):
                G.add_edge(t, main, critical_time=float(np.random.uniform(1.0, 3.0))); added += 1
    assert nx.is_directed_acyclic_graph(G)
    return G, added

def generate_like_one(refG: nx.DiGraph, model: StructureToGraphDecoder5,
                      neighbor_only: bool, seed: int,
                      equalize_row_time: bool, scale_total_time: bool,
                      post_fix: bool, save_pre: bool,
                      outdir: Path, prefix: str) -> Tuple[nx.DiGraph, Optional[nx.DiGraph], Dict[str, float], Dict[str, int], Tuple[Optional[Path], Path]]:
    # 条件
    s, widths, E_target, T_target = s_from_graph(refG)

    # 前向
    with torch.no_grad():
        logits, time_mat, widths_used = model(s, widths=widths)
        prob = torch.sigmoid(logits)
        prob = torch.where(torch.isfinite(prob), prob, torch.zeros_like(prob))

    # 结构选择
    selected, sel_stats = structured_select(prob, widths_used, E_target, neighbor_only, seed)

    # 时间处理
    if time_mat is not None:
        T = time_mat.detach().cpu().numpy().copy()
        ii, jj = np.where(selected)
        if equalize_row_time and ii.size > 0:
            row_sums = np.bincount(ii, weights=T[ii, jj], minlength=T.shape[0]).astype(np.float32)
            counts   = np.bincount(ii, minlength=T.shape[0]).astype(np.int32)
            row_means = np.where(counts > 0, row_sums / np.maximum(counts, 1), 0.0)
            for i in range(T.shape[0]):
                if counts[i] > 0:
                    T[i, selected[i]] = row_means[i]
        if scale_total_time and ii.size > 0:
            pred_total = float(T[ii, jj].sum())
            if pred_total > 0:
                T[ii, jj] *= (T_target / pred_total)
        time_mat = torch.as_tensor(T, dtype=time_mat.dtype, device=time_mat.device)

    # 构图
    G_pre = build_graph(selected, widths_used, time_mat)

    # 保存 pre
    pre_path = None
    if save_pre:
        pre_path, _ = save_graph(G_pre, outdir, prefix + "_pre")

    # 单源/单汇
    if post_fix:
        G_final, post_added = enforce_single_src_sink(G_pre)
    else:
        G_final, post_added = G_pre, 0

    # 保存 final
    final_path, _ = save_graph(G_final, outdir, prefix)

    # 统计
    Tref, _, _, _ = time_stats(refG)
    Tgen, Tmean, Tmin, Tmax = time_stats(G_final)
    time_dict = {
        "ref_total_T": Tref,
        "gen_total_T": Tgen,
        "gen_mean_T": Tmean,
        "gen_min_T": Tmin,
        "gen_max_T": Tmax,
        "gen_longest_T": longest_path_time(G_final),
    }
    sel_stats["post_added"] = int(post_added)

    return G_final, (G_pre if save_pre else None), time_dict, sel_stats, (pre_path, final_path)

# ============ Batch main ============
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Batch generate graphs and export CSV (defaults are hard-coded)")

    ap.add_argument("--ref_dir", default=DEFAULTS["ref_dir"])
    ap.add_argument("--pattern", default=DEFAULTS["pattern"])
    ap.add_argument("--ckpt", default=DEFAULTS["ckpt"])
    ap.add_argument("--outdir", default=DEFAULTS["outdir"])
    ap.add_argument("--csv", default=DEFAULTS["csv"])

    ap.add_argument("--neighbor_only", action="store_true", default=DEFAULTS["neighbor_only"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])

    ap.add_argument("--no_equalize_row_time", action="store_true", default=DEFAULTS["no_equalize_row_time"])
    ap.add_argument("--no_scale_total_time", action="store_true", default=DEFAULTS["no_scale_total_time"])
    ap.add_argument("--no_post_fix", action="store_true", default=DEFAULTS["no_post_fix"])
    ap.add_argument("--save_pre", action="store_true", default=DEFAULTS["save_pre"])

    ap.add_argument("--limit", type=int, default=DEFAULTS["limit"])

    args = ap.parse_args()

    print("[config] ref_dir=", args.ref_dir)
    print("[config] outdir =", args.outdir)
    print("[config] ckpt   =", args.ckpt)
    print("[config] neighbor_only=", args.neighbor_only,
          "| equalize_row_time=", not args.no_equalize_row_time,
          "| scale_total_time=", not args.no_scale_total_time,
          "| post_fix=", not args.no_post_fix,
          "| save_pre=", args.save_pre)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    ref_dir = Path(args.ref_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 收集文件
    files: List[Path] = sorted(ref_dir.glob(args.pattern))
    # 若 pattern 为 *.gpickle，再额外扫 json/gz
    if args.pattern == "*.gpickle":
        files += sorted(ref_dir.glob("*.json"))
        files += sorted(ref_dir.glob("*.gpickle.gz"))
        files += sorted(ref_dir.glob("*.json.gz"))

    if args.limit > 0:
        files = files[: args.limit]

    if not files:
        print("No reference graphs found.")
        return

    # 载模型（一次）
    model = load_model(args.ckpt)

    # CSV 准备
    csv_path = outdir / args.csv
    fieldnames = [
        "ref_path", "gen_pre_path", "gen_final_path",
        "neighbor_only", "equalize_row_time", "scale_total_time", "post_fix", "seed",
        "N_ref", "E_ref", "N_gen", "E_gen",
        "L", "widths",
        "N_pre", "E_pre",
        "srcs_pre", "sinks_pre",
        "no_in_pre", "no_out_pre",
        "deltaL_pre",
        "time_total_pre", "longest_pre",

        "cov_added", "topk_added", "post_added",
        "srcs_ref", "sinks_ref", "srcs_gen", "sinks_gen",
        "no_in_ref", "no_out_ref", "no_in_gen", "no_out_gen",
        "deltaL_ref", "deltaL_gen",
        "time_total_ref", "time_total_gen", "time_mean_gen", "time_min_gen", "time_max_gen",
        "longest_ref", "longest_gen",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, ref_path in enumerate(files, 1):
            row = {
                "ref_path": str(ref_path),
                "neighbor_only": int(args.neighbor_only),
                "equalize_row_time": int(not args.no_equalize_row_time),
                "scale_total_time": int(not args.no_scale_total_time),
                "post_fix": int(not args.no_post_fix),
                "seed": args.seed,
                "error": "",
            }
            try:
                refG = read_any(ref_path)
                # 参考统计
                N_ref, E_ref = refG.number_of_nodes(), refG.number_of_edges()
                Ls = topological_layers(refG);
                widths = [len(Li) for Li in Ls]
                src_ref, sink_ref = single_src_sink_counts(refG)
                no_in_ref, no_out_ref = degree_coverage_missing(refG)
                H_ref = deltaL_hist(refG)
                Tref, _, _, _ = time_stats(refG)
                LPT_ref = longest_path_time(refG)

                # ===== 生成 =====
                prefix = f"gen_like_{idx:05d}"
                G_final, G_pre, tstats, sstats, (pre_path, final_path) = generate_like_one(
                    refG, model,
                    neighbor_only=args.neighbor_only,
                    seed=args.seed,
                    equalize_row_time=not args.no_equalize_row_time,
                    scale_total_time=not args.no_scale_total_time,
                    post_fix=not args.no_post_fix,
                    save_pre=args.save_pre,
                    outdir=outdir,
                    prefix=prefix
                )

                # ===== pre 的统计（后处理前）=====
                if G_pre is not None:
                    N_pre, E_pre = G_pre.number_of_nodes(), G_pre.number_of_edges()
                    src_pre, sink_pre = single_src_sink_counts(G_pre)
                    no_in_pre, no_out_pre = degree_coverage_missing(G_pre)
                    H_pre = deltaL_hist(G_pre)
                    Tpre, _, _, _ = time_stats(G_pre)
                    LPT_pre = longest_path_time(G_pre)
                else:
                    N_pre = E_pre = src_pre = sink_pre = no_in_pre = no_out_pre = ""
                    H_pre = {}
                    Tpre = LPT_pre = ""

                # ===== final 的统计（后处理后）=====
                N_gen, E_gen = G_final.number_of_nodes(), G_final.number_of_edges()
                src_gen, sink_gen = single_src_sink_counts(G_final)
                no_in_gen, no_out_gen = degree_coverage_missing(G_final)
                H_gen = deltaL_hist(G_final)

                # ===== 写 CSV =====
                row.update({
                    "gen_pre_path": (str(pre_path) if pre_path else ""),
                    "gen_final_path": str(final_path),

                    "N_ref": N_ref, "E_ref": E_ref,
                    "L": len(widths), "widths": json.dumps(widths),

                    "N_pre": N_pre, "E_pre": E_pre,
                    "srcs_pre": src_pre, "sinks_pre": sink_pre,
                    "no_in_pre": no_in_pre, "no_out_pre": no_out_pre,
                    "deltaL_pre": json.dumps(H_pre),
                    "time_total_pre": Tpre, "longest_pre": LPT_pre,

                    "N_gen": N_gen, "E_gen": E_gen,
                    "cov_added": sstats["cov_added"], "topk_added": sstats["topk_added"],
                    "post_added": sstats["post_added"],
                    "srcs_ref": src_ref, "sinks_ref": sink_ref, "srcs_gen": src_gen, "sinks_gen": sink_gen,
                    "no_in_ref": no_in_ref, "no_out_ref": no_out_ref, "no_in_gen": no_in_gen, "no_out_gen": no_out_gen,
                    "deltaL_ref": json.dumps(H_ref), "deltaL_gen": json.dumps(H_gen),
                    "time_total_ref": Tref,
                    "time_total_gen": tstats["gen_total_T"],
                    "time_mean_gen": tstats["gen_mean_T"],
                    "time_min_gen": tstats["gen_min_T"],
                    "time_max_gen": tstats["gen_max_T"],
                    "longest_ref": LPT_ref,
                    "longest_gen": tstats["gen_longest_T"],
                })
            except Exception as e:
                row["error"] = str(e)

            writer.writerow(row)

    print("CSV saved to:", csv_path)

if __name__ == "__main__":
    main()
