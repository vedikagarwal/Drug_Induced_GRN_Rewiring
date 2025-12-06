import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import polars as pl
import numpy as np
import random
from tqdm import *


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")


def get_drug_metrics(drug_perturbed_weight, disease_edge_weight, healthy_edge_weight):
    """
    Compute the performance of a drug to move the sick GRN towards the control healthy GRN
    Note: All inputs are torch.TEnsor
    Note: Caluations are done over the MSE of edges
    Returns dict with:
    1. healthy_weight_restoration_percentage: 1 - [(drug_perturbed_weight - healthy_edge_weight) / (disease_edge_weight - healthy_edge_weight)]
    2. pred_diff_from_disease: (drug_perturbed_weight - disease_edge_weight)
    3. diff_disease_healthy_weight: (disease_edge_weight - healthy_edge_weight)
    """
    def MSE(w1, w2):
        return torch.mean((w1.float - w2.float)**2)
    
    diff_disease_healthy_weight = MSE(disease_edge_weight, healthy_edge_weight)
    pred_diff_from_disease = MSE(drug_perturbed_weight, disease_edge_weight)
    healthy_weight_restoration_percentage = 1.0 - (MSE(drug_perturbed_weight, healthy_edge_weight) / (diff_disease_healthy_weight + 1e-10)) # to avoid div by zero

    return {"healthy_weight_restoration_percentage": float(healthy_weight_restoration_percentage.item()), "pred_diff_from_disease": float(pred_diff_from_disease.item()), "healthy_weight_restoration_percentage": float(healthy_weight_restoration_percentage.item)}


def rank_drugs(drug_perturbed_weight, disease_edge_weight, healthy_edge_weight):
    """
    Rank drugs from best to worse according to their metrics
    """
    drug_rankings = []
    for drug_code, w_pred in drug_perturbed_weight.items():
        drug_metrics = get_drug_metrics(w_pred, disease_edge_weight, healthy_edge_weight)
        drug_metrics.update({"drug_code": drug_code})
        drug_rankings.append(drug_metrics)
    df = pd.DataFrame(drug_rankings).sort_values("healthy_weight_restoration_percentage", ascending=False, ignore_index=True)
    return df

def build_dataset(healthy_tf_to_gene_csv_path, disease_tf_to_gene_csv_path, tf_to_drug_csv_path, drug_columns=None):
    """
    Build dataset from GRAND database using provided csv files
    Expected csv file structures:
        healthy_tf_to_gene_csv_path/disease_tf_to_gene_csv_path:
            1. columns are TF, Gene, Sample1... , SampleN
            2. rows are TF-Gene edges and cell values are edge weights for each sample
        tf_to_drug_csv_path:
            1. columns are TF, Drug1... , DrugN
            2. rows are TF, cellrs are normalized z-score per drug
    """
    df_healthy = pl.read_csv(healthy_tf_to_gene_csv_path)
    df_disease  = pl.read_csv(disease_tf_to_gene_csv_path)

    # Take average across columns to make a single healthy GRN that will due to the averaging have less noise
    df_healthy_avg_cols = (df_healthy.with_columns(pl.mean_horizontal(pl.all().exclude(["TF", "gene"])).alias("w_control")).select(["TF", "gene", "w_control"]))
    df_disease_avg_cols = (df_disease.with_columns(pl.mean_horizontal(pl.all().exclude(["TF", "gene"])).alias("w_disease")).select(["TF", "gene", "w_disease"]))

    # Handle discrepancies between disease and healthy GRNs
    # Outer join edges (union)
    edges = df_healthy_avg_cols.join(df_disease_avg_cols, on=["TF", "gene"], how="outer")
    # If edge only exists in one GRN then set the weight to 0 in the missing edge
    edges = edges.with_columns([pl.col("w_control").fill_null(0.0), pl.col("w_disease").fill_null(0.0)])

    # Build bipartite graph that the GCN can take as input:
    tf_list   = sorted(edges["TF"].unique().to_list()) # sort to make sure out is always same
    gene_list = sorted(edges["gene"].unique().to_list()) 

    # Make node IDs for Pythorch Geometric:
    tf_count   = len(tf_list)
    gene_count = len(gene_list)
    num_nodes = tf_count + gene_count
    # IDs dicts
    tf_to_idx = {tf: i for i, tf in enumerate(tf_list)}
    gene_to_idx = {gene: tf_count + i for i, gene in enumerate(gene_list)}

    # Map tf and gene names to node ids 
    tf_idx_np   = np.array([tf_to_idx[tf] for tf in edges["TF"].to_list()], dtype=np.int64)
    gene_idx_np = np.array([gene_to_idx[gene] for gene in edges["gene"].to_list()], dtype=np.int64)

    tf_per_edge = torch.from_numpy(tf_idx_np).long()
    gene_per_edge = torch.from_numpy(gene_idx_np).long()

    # Since pytotch geometric cannot handle negative weights do an undirected graph with +/- features
    # Features will look like this: [|w_disease|, sign(w_disease)]
    edge_index_forward  = torch.stack([tf_per_edge, gene_per_edge], dim=0) #These two lines make both directins
    edge_index_backward = torch.stack([gene_per_edge, tf_per_edge], dim=0)
    edge_index = torch.cat([edge_index_forward, edge_index_backward], dim=1)
    patient_edge_weights_np = np.array(edges["disease_weights"].to_list(), dtype=np.float32)
    patient_edge_weights = torch.from_numpy(patient_edge_weights_np)

    edge_mag  = patient_edge_weights.abs().unsqueeze(-1)
    edge_sign = torch.sign(patient_edge_weights).unsqueeze(-1)
    edge_features = torch.cat([edge_mag, edge_sign], dim=-1)

    healthy_edge_weights_np = np.array(edges["healthy_weights"].to_list(), dtype=np.float32)
    healthy_edge_weights = torch.from_numpy(healthy_edge_weights_np)
    gene_nodes = torch.arange(tf_count, num_nodes, dtype=torch.long)
    gene_node_idx = gene_nodes.clone()
    gene_expr_targets = None
    df_drug_pl = pl.read_csv(drug_csv_path)

    all_drug_cols = [c for c in df_drug_pl.columns if c != "Row"]
    if drug_columns is None:
        drug_columns = all_drug_cols
    df_drug = df_drug_pl.to_pandas().set_index("Row")

    in_channels = 2
    data_list = []
    drug_names = []

    for col in drug_columns:
        x = torch.zeros((num_nodes, in_channels), dtype=torch.float32)
        tf_nodes   = torch.arange(0, tf_count, dtype=torch.long)
        drug_series = df_drug[col]
        tf_drug_z = torch.zeros(tf_count, dtype=torch.float32)
        for tf, idx in tf_to_idx.items():
            if tf in drug_series.index:
                tf_drug_z[idx] = float(drug_series.loc[tf])
            else:
                tf_drug_z[idx] = 0.0
        x[tf_nodes, 0] = tf_drug_z
        x[tf_nodes, 1] = 1.0 

        data = Data(
            x=x,
            edge_index=edge_index,
        )

        data_list.append(data)
        drug_names.append(col)

    return (
        data_list,
        tf_per_edge,
        gene_per_edge,
        edge_features,
        healthy_edge_weights,
        gene_node_idx,
        gene_expr_targets,
        drug_names,
        patient_edge_weights,
    )


def train(
    control_csv,
    disease_csv,
    drug_csv,
    drug_columns = None,
    epochs = 200,
    lr = 1e-3,
    train_frac = 0.8,
    val_frac = 0.1,
    seed = 0,
):
    (
        data_list,       
        tf_edges,
        g_edges,         
        edge_features,  
        edge_targets,    
        drug_names,     
        disease_edge_weights,
    ) = build_dataset(
        control_csv=control_csv,
        disease_csv=disease_csv,
        drug_csv=drug_csv,
        drug_columns=drug_columns,
    )

    n_drugs = len(drug_names)
    indices = list(range(n_drugs))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(train_frac * n_drugs)
    n_val = int(val_frac * n_drugs)
    # train with at least 1
    n_train = max(1, n_train)
    # do not go over total
    if n_train + n_val > n_drugs:
        n_val = max(0, n_drugs - n_train)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    data_train = [data_list[i] for i in train_idx]
    data_val   = [data_list[i] for i in val_idx]

    drug_train = [drug_names[i] for i in train_idx]
    drug_val   = [drug_names[i] for i in val_idx]
    drug_test  = [drug_names[i] for i in test_idx]

    print(f"Total drugs: {n_drugs}, train: {len(drug_train)}, val: {len(drug_val)}, test: {len(drug_test)}")

    tf_edges       = tf_edges.to(device)
    g_edges        = g_edges.to(device)
    edge_features  = edge_features.to(device)
    edge_targets   = edge_targets.to(device)
    disease_edge_weights = disease_edge_weights.to(device)

    in_channels = data_list[0].x.size(1)
    edge_feat_dim = edge_features.size(1)

    model = GRNDrugGCN(
        in_channels=in_channels,
        hidden_channels=32,
        gcn_layers=2,
        edge_hidden_channels=32,
        gene_head_hidden=32,
        predict_gene_expression=False,
        edge_feat_dim=edge_feat_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    edge_loss_fn = nn.MSELoss()

    for epoch in trange(1, epochs + 1, desc="Epochs"):
        model.train()
        total_train_loss = 0.0
        for data in data_train:
            data = data.to(device)

            optimizer.zero_grad()

            edge_pred, _ = model(
                data=data,
                tf_edge_idx=tf_edges,
                gene_edge_idx=g_edges,
                edge_features=edge_features,
                gene_node_idx=None,
            )

            loss_edge = edge_loss_fn(edge_pred, edge_targets)
            loss_edge.backward()
            optimizer.step()

            total_train_loss += loss_edge.item()

        mean_train_loss = total_train_loss / len(data_train)
        mean_val_loss = None
        if len(data_val) > 0:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for data in data_val:
                    data = data.to(device)
                    edge_pred, _ = model(
                        data=data,
                        tf_edge_idx=tf_edges,
                        gene_edge_idx=g_edges,
                        edge_features=edge_features,
                        gene_node_idx=None,
                    )
                    total_val_loss += edge_loss_fn(edge_pred, edge_targets).item()
            mean_val_loss = total_val_loss / len(data_val)

        if epoch % 20 == 0:
            if mean_val_loss is not None:
                print(f"Epoch {epoch:03d}, train loss: {mean_train_loss:.4f}, val loss: {mean_val_loss:.4f}")
            else:
                print(f"Epoch {epoch:03d}, train loss: {mean_train_loss:.4f}")
    model.eval()
    all_edge_pred = {}
    with torch.no_grad():
        for data, drug_name in zip(data_list, drug_names):
            data = data.to(device)
            edge_pred, _ = model(
                data=data,
                tf_edge_idx=tf_edges,
                gene_edge_idx=g_edges,
                edge_features=edge_features,
                gene_node_idx=None,
            )
            all_edge_pred[drug_name] = edge_pred.cpu()
    splits = {"train": drug_train, "val": drug_val, "test": drug_test}
    return model, all_edge_pred, edge_targets.cpu(), disease_edge_weights.cpu(), drug_names, splits


class GRNDrugGCN(nn.Module):
    """
    GCN on unsigned weights durign convolution (due to pytorch limitations)
    """
    def __init__(
        self,
        in_channels,
        hidden_channels = 32,
        gcn_layers = 2,
        edge_hidden_channels = 32,
        gene_head_hidden = 32,
        predict_gene_expression = True,
        edge_feat_dim = 2  # [w, sign]
    ):
        super().__init__()

        self.predict_gene_expression = predict_gene_expression
        self.edge_feat_dim = edge_feat_dim
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(gcn_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels + edge_feat_dim, edge_hidden_channels),
            nn.ReLU(),
            nn.Linear(edge_hidden_channels, 1)
        )
        if predict_gene_expression:
            self.gene_mlp = nn.Sequential(
                nn.Linear(hidden_channels, gene_head_hidden),
                nn.ReLU(),
                nn.Linear(gene_head_hidden, 1)
            )

    def forward(
        self,
        data,
        tf_edge_idx,
        gene_edge_idx,
        edge_features,
        gene_node_id = None,
    ):
        x, edge_index = data.x, data.edge_index

        h = x
        for layer in self.gcn_layers:
            h = layer(h, edge_index)
            h = F.relu(h)
        h_tf   = h[tf_edge_idx]
        h_gene = h[gene_edge_idx]
        h_edge_input = torch.cat([h_tf, h_gene, edge_features], dim=-1)
        edge_pred = self.edge_mlp(h_edge_input).squeeze(-1)
        gene_expr_pred = None
        if self.predict_gene_expression:
            h_gene_nodes = h[gene_node_idx]
            gene_expr_pred = self.gene_mlp(h_gene_nodes).squeeze(-1)

        return edge_pred, gene_expr_pred

if __name__ == "__main__":
    control_csv_path = "/content/drive/MyDrive/BioInf/df_c_final.csv"
    disease_csv_path = "/content/drive/MyDrive/BioInf/df_d_final.csv"
    drug_csv_path    = "/content/drive/MyDrive/BioInf/drugs_normalized_avgs.csv"

    selected_drugs = None 
    # selected_drugs = ["ASG001_MCF7_24H:F13"]

    model, all_edge_pred, w_healthy, w_disease, drug_names, splits = train(
        control_csv=control_csv_path,
        disease_csv=disease_csv_path,
        drug_csv=drug_csv_path,
        drug_columns=selected_drugs,
        epochs=200,
        lr=1e-3,
        train_frac=0.8,
        val_frac=0.1,
        seed=0,
    )
    test_drugs = splits["test"]
    test_edge_pred = {d: all_edge_pred[d] for d in test_drugs}
    df_scores = rank_drugs(
        all_edge_pred=test_edge_pred,
        w_disease=w_disease,
        w_healthy=w_healthy,
    )

    print("Top 10 drugs to reconstruct healthy GRN:")
    print(df_scores.head(10))
