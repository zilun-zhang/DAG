## Installation
Python 3.9+
PyTorch, NetworkX, NumPy, Matplotlib, pandas

## src
1. config.py
Unified management of paths, training hyperparameters, loss weights, and devices（CPU/GPU）

DATA_DIR：Data directory for training/exploration
CHECKPOINT_DIR：Model weight storage directory
Loss Weight：W_BCE, W_TIME, W_TOTALT, W_LONGEST, W_DEG_COV, W_SRC_SINK_SOFT, W_TIME_NODE 

2. data.py
Robust image reading: support .gpickle、.gpickle.gz、.json（NetworkX node-link）

Graph → Tensor/Stats: Output
A_target（N×N，边 0/1）
T_target（N×N，time）
widths（Width of each layer）
s_vec（图级特征向量 [Normalized version of the graph-level feature vector [N,E,L,W,T_total]）
total_T（total time）

3. models.py
StructureToGraphDecoder5

Input: Input: graph-level s_vec and widths

Output:
A_logits (N×N, logits of edge probabilities)
time_mat (N×N, edge time predictions)
widths (the final layer width used)
Internally, predictions are only made on legal node pairs (same-level/reverse-order ones are masked) and the time is guaranteed to be positive.


4. losses.py

LossPack5 combines multiple losses (weights controlled via config5.py):

Structure: Edge BCE (predicted vs. actual edges), degree coverage penalty, soft single-source/single-sink, DAG smoothing term

Time: Edge time L1, node row mean alignment (making outgoing edge times from the same node consistent with the reference row mean)

Path: Constraints on total time/weighted longest path time

5.train.py

Function: Read data → Forward → Calculate loss → Backward and optimize → Periodically save optimal weights.

6.infer5.py
Given a reference graph and training weights, generate a DAG with similar structure and temporal style, and save it in .gpickle and .json.


Extract s_vec and widths (including T_total) from the reference graph.
Model outputs edge probabilities and times.
Structural edge selection (three steps).
Degree coverage cov_added: Ensures ≥1 input for non-first layers and ≥1 output for non-last layers.
Top-K completion topk_added: Fills in the target number of edges from the remaining legal pairs in descending order of probability.
Single-source/single-sink patching post_added: If multiple sources/sinks still exist, only edges are added for merging.
(Optional) Time intra-row equalization and total scaling (alignment with the total time of the reference graph).
Save the graph and print statistics.

6. show.py
Draw the graph (optionally save as a PNG).
Edges crossing layers (ΔL > 1) will be highlighted with arcs and labeled with the critical_time (if any).
The console prints N/E/TotalT/LongestPathT.


7.utils.py

General tool (used for both training and inference):

Topological layers topological_layers
Layer index/width processing
Legal edge position generation and matrix-assisted calculations

8. probe.py
Quickly browse the N/E/L/W overview of the first few images under DATA_DIR and count suspicious samples (such as E==0 or L<=1)


## test
1. udit_postfix.py
Heuristically "guess" which edges in a graph may have been added during the single-source/single-sink patching phase; and compares the ΔL histogram with the change in edge count.

2. test_sample
Checks whether the sample is a reasonable DAG.

3. test.py
Checks the differences between the original and generated graphs.

4.batch_report.csv
dataset

5. audit_post_readme
Explanation of the dataset title

## model

decoder_final1.pt: The final weight of model training
decoder_best1.pt： The best weight of model training

## data

training and test sample
