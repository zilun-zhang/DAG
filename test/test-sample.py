
from pathlib import Path
import pickle
from collections import Counter

import networkx as nx

from config5 import DATA_DIR
from data5 import load_graph_files
from utils5 import topological_layers


def read_any_gpickle(path: Path):
    try:
        from networkx.readwrite.gpickle import read_gpickle as nx_read_gpickle
        return nx_read_gpickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def main():
    files = load_graph_files(DATA_DIR)
    print(f"Found {len(files)} files under {DATA_DIR}")

  
    dag_ok = 0
    multi_src = 0
    multi_sink = 0
    src_hist = Counter()
    sink_hist = Counter()

 
    out_csv = Path("source_sink_report5.csv")
    with out_csv.open("w", encoding="utf-8") as fcsv:
        fcsv.write("file,N,E,is_dag,L,W,num_sources,num_sinks\n")

        for i, p in enumerate(files):
            G = read_any_gpickle(p)
            if not nx.is_directed(G):
                G = G.to_directed()

            is_dag = nx.is_directed_acyclic_graph(G)
            L = W = -1
            if is_dag:
                dag_ok += 1
                layers = topological_layers(G)
                L = len(layers)
                W = max(layer_widths(layers)) if layers else 0

            N = G.number_of_nodes()
            E = G.number_of_edges()

            sources = [n for n in G.nodes if G.in_degree(n) == 0]
            sinks   = [n for n in G.nodes if G.out_degree(n) == 0]
            ns, nt = len(sources), len(sinks)

            src_hist[ns] += 1
            sink_hist[nt] += 1
            if ns != 1:
                multi_src += 1
            if nt != 1:
                multi_sink += 1

          
            fcsv.write(f"{p.name},{N},{E},{int(is_dag)},{L},{W},{ns},{nt}\n")

            # Print only "abnormal" samples (not 1 source or 1 sink)
            if ns != 1 or nt != 1:
                print(f"[{i:03d}] {p.name} | N={N}, E={E}, DAG={is_dag}, L={L}, W={W}, "
                      f"sources={ns}, sinks={nt}")

    
    print("\n==== SUMMARY ====")
    print(f"DAG count: {dag_ok}/{len(files)}")
    print(f"multi-source (ns!=1): {multi_src}/{len(files)}")
    print(f"multi-sink   (nt!=1): {multi_sink}/{len(files)}")
    print("num_sources histogram:", dict(src_hist))
    print("num_sinks   histogram:", dict(sink_hist))
    print(f"\nWrote detail CSV to: {out_csv.resolve()}")

if __name__ == "__main__":
    main()

