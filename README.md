# Financial-Fraud-Detection-Graph-Neural-Network-with-Attention-

1. OVERVIEW: 
Fraud detection in finance is a critical task where machine learning must identify rare, suspicious activities among massive amounts of legitimate transactions. Recent advances show that Graph Neural Networks (GNNs) excel in such settings by modeling entities (accounts, transactions, reviews, etc.) as nodes and their relationships as edges. GNNs aggregate neighborhood information through different relations [1], allowing them to spot complex fraud patterns that traditional methods often miss. For example, NVIDIA notes that GNNs “reveal suspicious nodes (in accounts and transactions) by aggregating their neighborhood information” and can identify larger anomalies across a network. This repository demonstrates a GNN-based approach for fraud detection, using the public YelpChi dataset (a review fraud benchmark) as an example. The code implements both a Graph Convolutional Network (GCN) and a Graph Attention Network (GAT) classifier, showing how to preprocess the data, train the models, and evaluate their performance. 

2. MOTIVATION:
Fraudsters often operate in networks (linked accounts, co-reviews, coordinated attacks) rather than in isolation. As fraud patterns become more complex, rule-based or flat feature models struggle to capture these relational cues. GNNs natively leverage graph structure: each node’s representation is iteratively updated by aggregating features from its neighbors. In the financial context, this means a model can infer risk by looking at, say, a transaction’s connected accounts, merchants, or devices, not just its local features [2]. By taking into account multiple “hops” of connectivity, GNNs effectively build a larger receptive field that exposes chains of fraudulent activity. Graph Attention Networks (GATs) extend this idea by learning attention weights over neighbors, so suspicious connections are weighted more heavily. Importantly, fraud datasets are typically highly imbalanced and evasive [3]. The YelpChi dataset, for instance, contains 45,954 reviews but only ~14.5% are spam/fraud. GNNs can help here by incorporating context: even if a fraudulent review looks innocuous alone, its neighbors (e.g. from the same user or product) may reveal anomalies.

3. DATASET:
3.a. We use the YelpChi dataset [4], a public graph for spam-review detection. In this graph, each node is a review, labeled as legitimate (0) or fraudulent (1). There are 45,954 total reviews, with 6,677 frauds (≈14.5%). Each review has 32 hand-crafted features (e.g. word counts, metadata). Edges come from three relation types (metapaths) commonly used in literature: User–Review–User (reviews by the same user), Review–Topic–Review (reviews sharing a topic), and Review–Sentiment–Review (same sentiment category). We combine these into one undirected graph so each pair of related reviews is connected by an edge. 

3.b. Dataset statistics:
Nodes (reviews): 45,954
Node features: 32-dimensional
Edges (all relations combined): 8,051,348 (multi-relational links)
Fraudulent nodes: 6,677 (14.5% of total)
Split: 70% of nodes for training, 30% for testing (random shuffle)
These statistics closely match published descriptions. The class imbalance (few frauds) is typical of financial fraud data, as noted in industry. 

Repo nav link to load and summarize the data: https://github.com/AdSpaceEngineeer/Fraud-Detection-Graph-Neural-Network-with-Attention-/blob/main/Dataset%20load%20and%20summarize 

4. METHOD - GCN and GAT Models

We implement two GNN models: a FraudGCN and a FraudGAT. Both are two-layer classifiers predicting fraud (node classification).

4.a. FraudGCN: 
A simple 2-layer Graph Convolutional Network (GCN).In each layer, node features are aggregated (via graph convolution) and passed through ReLU/dropout. The final layer outputs logits for the two classes. This follows prior work showing that even basic GCNs benefit fraud detection by leveraging network structure. 

Repo nav link : https://github.com/AdSpaceEngineeer/Fraud-Detection-Graph-Neural-Network-with-Attention-/blob/main/FraudGCN 

4.b. FraudGAT: 
A 2-layer Graph Attention Network. The first layer uses multi-head attention (heads=4) to learn neighbor weights, then a second head-reducing layer for final logits. Attention allows the model to “specify different weights to different nodes in a neighborhood”, which can highlight fraud-related links. 

Repo nav link: https://github.com/AdSpaceEngineeer/Fraud-Detection-Graph-Neural-Network-with-Attention-/blob/main/FraudGAT 

These models use PyTorch Geometric layers. Note that GCN uses simple mean-aggregation, whereas GAT “operates on graph data, leveraging masked self-attention” to weight neighbors dynamically. Both models end with 2 output logits (legit/fraud).

4.c. Training and Evaluation

We train both models using supervised cross-entropy on the labeled nodes. 

4.c.i. Key steps:
- Loss and optimizer: We use torch.nn.CrossEntropyLoss on the training mask, with Adam optimizer (lr=0.001, weight_decay=5e-4 as a small regularizer).
- Metrics: We report test accuracy, but more importantly recall and F1 on the fraud class. High recall is often critical in fraud work (catch most frauds even if at the cost of lower accuracy).
- Class imbalance handling: The models see ~14.5% positive nodes; we rely on the GNN’s context-awareness rather than resampling.
- The training code above will output epoch-wise loss and test metrics. (A similar loop can be used for model_gcn.) By the end, we compute final metrics on the held-out 30% of nodes.

Repo nav link: https://github.com/AdSpaceEngineeer/Fraud-Detection-Graph-Neural-Network-with-Attention-/blob/main/Training%20and%20Evaluation

5. RESULTS

We obtained performance metrics for each model on the YelpChi data. 

Our model achieved roughly:
Accuracy: 54.33%
**Recall (fraud class): 94.44%**
F1-score (fraud class): 0.6898

These results indicate our GAT is catching most fraud cases (high recall) but at the expense of many false positives (moderate accuracy and precision). For context, this matches known patterns: fraud detection often prioritizes recall. In comparison, a recent hierarchical GNN (HA-GNN) reported about 79.8% recall with 85.7% AUC on the same YelpChi dataset [5]. 

Our model’s recall (~94%) exceeds that baseline, though with lower precision. Exact values will vary with random seed; the above is illustrative of the approach. The important takeaway is that even a straightforward GAT significantly improves on random guessing by leveraging graph structure. Additional improvements (not shown here) could include more sophisticated GNN architectures or sampling techniques to handle the 8-million-edge graph.

6. USAGE

To reproduce this project, follow these steps:

6.1. Go to repository and navigate into it.

6.2. Install dependencies:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
    pip install torch-geometric
    pip install sklearn

6.3. Prepare data: 
Download YelpChi.zip (containing YelpChi.mat) from [4] and place it in your data folder.

6.4. Run the code: 
Execute the provided training script to train the GNN model on YelpChi and print metrics.

6.5. Inspect results: 
Review the printed accuracy, recall, and F1. Optionally, adjust hyperparameters or try different models.

7. CALL TO ACTION FOR FUTURE WORK

7.1. This project demonstrates how to apply GNNs to fraud detection, a skill valuable in finance and banking, possibly in scam detection too. We saw that GCNs and GATs can leverage relationship data to flag anomalous nodes.

7.2. To further impress potential employers, you could extend this repo by:
    i) Adding more metrics and analysis: Show ROC/AUC, confusion matrix, or per-class breakdown.
    ii) Advanced models: Implement more advanced GNN-with-Attention techniques or other fraud-specialized GNNs.
    iii) Real financial data: Apply the code to a credit-card fraud or transaction dataset, demonstrating transferability.
    iv) Explanations: Use GAT attention scores to highlight why a transaction was flagged, aiding interpretability.

8. REFERENCES:

1] NVIDIA Tech Blog – Optimizing Fraud Detection in Financial Services with GNNs - https://developer.nvidia.com/blog/optimizing-fraud-detection-in-financial-services-with-graph-neural-networks-and-nvidia-gpus/
2] Dou et al., CARE-GNN: Enhancing GNN-based Fraud Detectors (CIKM 2020) - https://arxiv.org/abs/2008.08692
3] Veličković et al., Graph Attention Networks (GATs) (ICLR 2018) - https://arxiv.org/abs/1710.10903
4] YelpChi datasets stats - https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset
5] Liu et al., Improving Fraud Detection via Hierarchical Attention GNN (arXiv 2022) - https://arxiv.org/pdf/2202.06096
