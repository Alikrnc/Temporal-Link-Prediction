import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import TemporalLinkPredictor
import time
import csv

# --- Configuration & Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001
EPOCHS = 10
HISTORY_WINDOW_T_REL = 0.15 # Define the history window size (e.g., use last 1% of t_rel range)
VALIDATION_SPLIT_RATIO = 0.2
MODEL_PARAMS = {
    # These should match the ones used in model.py's __main__
    "num_nodes": 19442,
    "num_edge_types": 248,
    "node_categorical_cardinalities": [34, 20, 16, 285, 56, 220, 17, 26],
    "edge_type_categorical_cardinalities": [248, 20, 143],
    "node_continuous_feature_dim": 2,
    "node_cat_embed_dim": 10,
    "edge_type_itself_embed_dim": 16,
    "edge_type_cat_embed_dim": 8,
    "time_embed_dim": 16,
    "gnn_hidden_dim": 128,
    "prediction_hidden_dim": 64,
}

NODE_FEATURES_PATH = "data/final/node_final.parquet"
EDGE_FEATURES_PATH = "data/final/edge_final.parquet"
TRAINING_SAMPLES_PATH = "data/final/train_neg_samp.parquet"
POSITIVE_EDGES_PATH = "data/final/train_final.parquet"

def load_data():
    print("Loading data...")
    node_df = pd.read_parquet(NODE_FEATURES_PATH)
    edge_type_df = pd.read_parquet(EDGE_FEATURES_PATH)
    training_samples_df = pd.read_parquet(TRAINING_SAMPLES_PATH)
    positive_edges_df = pd.read_parquet(POSITIVE_EDGES_PATH)

    ORIGINAL_NODE_ID_COLUMN_NAME = 'node_id' # <<< --- VERIFY AND CHANGE THIS IF NEEDED
    original_to_remapped_id_map = pd.Series(node_df.index, index=node_df[ORIGINAL_NODE_ID_COLUMN_NAME]).to_dict()
    
    positive_edges_df['src_id'] = positive_edges_df['src_id'].map(original_to_remapped_id_map)
    positive_edges_df['dst_id'] = positive_edges_df['dst_id'].map(original_to_remapped_id_map)
    positive_edges_df.dropna(subset=['src_id', 'dst_id'], inplace=True)
    positive_edges_df['src_id'] = positive_edges_df['src_id'].astype(int)
    positive_edges_df['dst_id'] = positive_edges_df['dst_id'].astype(int)

    training_samples_df['src_id'] = training_samples_df['src_id'].map(original_to_remapped_id_map)
    training_samples_df['dst_id'] = training_samples_df['dst_id'].map(original_to_remapped_id_map)
    training_samples_df.dropna(subset=['src_id', 'dst_id'], inplace=True)
    training_samples_df['src_id'] = training_samples_df['src_id'].astype(int)
    training_samples_df['dst_id'] = training_samples_df['dst_id'].astype(int)
    
    print("Sorting positive_edges_df by t_rel...")
    positive_edges_df.sort_values(by='t_rel', inplace=True)
    positive_edges_df.reset_index(drop=True, inplace=True)

    print("Converting positive edge IDs to tensors")
    if not positive_edges_df.empty:
        positive_edges_src_tensor_cpu = torch.tensor(positive_edges_df['src_id'].values, dtype=torch.long)
        positive_edges_dst_tensor_cpu = torch.tensor(positive_edges_df['dst_id'].values, dtype=torch.long)
    else:
        # Create empty tensors if positive_edges_df is empty
        positive_edges_src_tensor_cpu = torch.empty((0,), dtype=torch.long)
        positive_edges_dst_tensor_cpu = torch.empty((0,), dtype=torch.long)
    print("Finished converting positive edge IDs.")

    # --- Validation split ---
    split_index = int(len(training_samples_df) * (1 - VALIDATION_SPLIT_RATIO))
    train_df = training_samples_df.iloc[:split_index]
    validation_samples_df = training_samples_df.iloc[split_index:]

    return node_df, edge_type_df, train_df, validation_samples_df, positive_edges_df, \
           positive_edges_src_tensor_cpu, positive_edges_dst_tensor_cpu

def prepare_input_tensors(node_df, edge_type_df, device):
    node_cont_features = torch.tensor(node_df[['in_deg_norm', 'out_deg_norm']].values, dtype=torch.float32).to(device)
    
    node_cat_feature_cols = [f"node_feature_{i}_code" for i in range(1, 9)]
    node_cat_features_list = [
        torch.tensor(node_df[col].values, dtype=torch.long).to(device) for col in node_cat_feature_cols
    ]

    edge_type_cat_feature_cols = [f"edge_feature_{i}_code" for i in range(1, 4)]
    edge_type_cat_features_list = [
        torch.tensor(edge_type_df[col].values, dtype=torch.long).to(device) for col in edge_type_cat_feature_cols
    ]

    return node_cont_features, node_cat_features_list, edge_type_cat_features_list

def train():
    # Unpack the tensors
    node_df, edge_type_df, train_df, validation_samples_df, positive_edges_df, \
    positive_edges_src_tensor_cpu, positive_edges_dst_tensor_cpu = load_data()
    
    if train_df.empty:
        print("No training samples to process after loading. Exiting training.")
        return

    raw_node_cont_feats, raw_node_cat_feats_list, raw_edge_type_cat_feats_list = prepare_input_tensors(node_df, edge_type_df, DEVICE)

    model = TemporalLinkPredictor(**MODEL_PARAMS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    log_path = 'training_log.csv'
    with open(log_path, 'w', newline='') as logfile:
        writer = csv.writer(logfile)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'duration_min'])

        print("\nStarting training...")
        if not positive_edges_df.empty:
            historical_t_rel_values = positive_edges_df['t_rel'].values
        else:
            historical_t_rel_values = np.array([])
        for epoch in range(EPOCHS):
            start_time = time.time()
            model.train()
            h_prev = torch.zeros(MODEL_PARAMS["num_nodes"], MODEL_PARAMS["gnn_hidden_dim"]).to(DEVICE)
            c_prev = torch.zeros(MODEL_PARAMS["num_nodes"], MODEL_PARAMS["gnn_hidden_dim"]).to(DEVICE)
            epoch_loss = 0
            correct = 0
            total = 0
            grouped_by_time_train = train_df.groupby('t_rel')
            num_time_groups_train = len(grouped_by_time_train)
            print(f"Epoch {epoch+1}/{EPOCHS} - Iterating over {num_time_groups_train} unique t_rel groups for training.")
            
            for i, (t_rel_group, batch_df) in enumerate(tqdm(grouped_by_time_train, desc=f"Epoch {epoch+1}/{EPOCHS} Training", leave=False)):
                optimizer.zero_grad()
                
                # --- historical edge selection using sliding window ---
                current_graph_edge_index = torch.empty((2, 0), dtype=torch.long).to(DEVICE) # Default empty
                if len(historical_t_rel_values) > 0:
                    # Define the window for historical edges
                    lower_t_rel_bound_for_window = max(0, t_rel_group - HISTORY_WINDOW_T_REL)
                    
                    # Find indices for the window
                    lower_bound_idx = np.searchsorted(historical_t_rel_values, lower_t_rel_bound_for_window, side='right') # Use 'right' to get elements > bound
                    upper_bound_idx = np.searchsorted(historical_t_rel_values, t_rel_group, side='right') # Use 'right' to include t_rel_group

                    if upper_bound_idx > lower_bound_idx:
                        src_snapshot = positive_edges_src_tensor_cpu[lower_bound_idx:upper_bound_idx].to(DEVICE)
                        dst_snapshot = positive_edges_dst_tensor_cpu[lower_bound_idx:upper_bound_idx].to(DEVICE)
                        current_graph_edge_index = torch.stack([src_snapshot, dst_snapshot], dim=0)
               
                src_nodes = torch.tensor(batch_df['src_id'].values, dtype=torch.long).to(DEVICE)
                dst_nodes = torch.tensor(batch_df['dst_id'].values, dtype=torch.long).to(DEVICE)
                edge_types = torch.tensor(batch_df['edge_id'].values, dtype=torch.long).to(DEVICE)
                t_rel_batch = torch.tensor(batch_df['t_rel'].values, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
                labels = torch.tensor(batch_df['label'].values, dtype=torch.float32).to(DEVICE)

                predictions, h_new, c_new = model(
                    src_node_ids=src_nodes,
                    dst_node_ids=dst_nodes,
                    edge_type_ids=edge_types,
                    t_rel=t_rel_batch,
                    raw_all_node_features_cont=raw_node_cont_feats,
                    raw_all_node_features_cat_codes_list=raw_node_cat_feats_list,
                    raw_all_edge_type_features_cat_codes_list=raw_edge_type_cat_feats_list,
                    current_graph_edge_index=current_graph_edge_index,
                    h_prev=h_prev,
                    c_prev=c_prev
                )

                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_df)
                preds_binary = (predictions >= 0.5).float()
                correct += (preds_binary == labels).sum().item()
                total += len(batch_df)
                
                h_prev = h_new.detach()
                c_prev = c_new.detach()

            avg_epoch_loss = epoch_loss / len(train_df) if len(train_df) > 0 else 0
            train_acc = correct / total if total > 0 else 0
            print(f"Epoch {epoch+1}/{EPOCHS}, Average Training Loss: {avg_epoch_loss:.4f}, Training Accuracy: {train_acc:.4f}")
            val_loss, val_acc = 0, 0
            if not validation_samples_df.empty:
                # Validation loss and accuracy
                model.eval()
                val_loss_total = 0
                val_correct = 0
                val_total = 0
                h_val = torch.zeros(MODEL_PARAMS["num_nodes"], MODEL_PARAMS["gnn_hidden_dim"]).to(DEVICE)
                c_val = torch.zeros(MODEL_PARAMS["num_nodes"], MODEL_PARAMS["gnn_hidden_dim"]).to(DEVICE)
                grouped_by_time_val = validation_samples_df.groupby('t_rel')
                with torch.no_grad():
                    for t_rel_group, batch_df in grouped_by_time_val:
                        current_graph_edge_index = torch.empty((2, 0), dtype=torch.long).to(DEVICE)
                        if len(historical_t_rel_values) > 0:
                            lower_t_rel_bound_for_window = max(0, t_rel_group - HISTORY_WINDOW_T_REL)
                            lower_bound_idx = np.searchsorted(historical_t_rel_values, lower_t_rel_bound_for_window, side='right')
                            upper_bound_idx = np.searchsorted(historical_t_rel_values, t_rel_group, side='right')
                            if upper_bound_idx > lower_bound_idx:
                                src_snapshot = positive_edges_src_tensor_cpu[lower_bound_idx:upper_bound_idx].to(DEVICE)
                                dst_snapshot = positive_edges_dst_tensor_cpu[lower_bound_idx:upper_bound_idx].to(DEVICE)
                                current_graph_edge_index = torch.stack([src_snapshot, dst_snapshot], dim=0)
                        src_nodes = torch.tensor(batch_df['src_id'].values, dtype=torch.long).to(DEVICE)
                        dst_nodes = torch.tensor(batch_df['dst_id'].values, dtype=torch.long).to(DEVICE)
                        edge_types = torch.tensor(batch_df['edge_id'].values, dtype=torch.long).to(DEVICE)
                        t_rel_batch = torch.tensor(batch_df['t_rel'].values, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
                        labels = torch.tensor(batch_df['label'].values, dtype=torch.float32).to(DEVICE)
                        predictions, h_new_val, c_new_val = model(
                            src_node_ids=src_nodes,
                            dst_node_ids=dst_nodes,
                            edge_type_ids=edge_types,
                            t_rel=t_rel_batch,
                            raw_all_node_features_cont=raw_node_cont_feats,
                            raw_all_node_features_cat_codes_list=raw_node_cat_feats_list,
                            raw_all_edge_type_features_cat_codes_list=raw_edge_type_cat_feats_list,
                            current_graph_edge_index=current_graph_edge_index,
                            h_prev=h_val, 
                            c_prev=c_val
                        )
                        loss = criterion(predictions, labels)
                        val_loss_total += loss.item() * len(batch_df)
                        preds_binary = (predictions >= 0.5).float()
                        val_correct += (preds_binary == labels).sum().item()
                        val_total += len(batch_df)
                        h_val = h_new_val.detach()
                        c_val = c_new_val.detach()
                val_loss = val_loss_total / len(validation_samples_df) if len(validation_samples_df) > 0 else 0
                val_acc = val_correct / val_total if val_total > 0 else 0
                print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
            else:
                print("Validation set is empty. Skipping validation.")
            duration_min = (time.time() - start_time) / 60
            print(f"Epoch {epoch+1} finished in {duration_min:.2f} minutes.")
            writer.writerow([epoch+1, avg_epoch_loss, train_acc, val_loss, val_acc, duration_min])
            logfile.flush()
    print("\nTraining finished.")
    model_save_path = "temporal_link_predictor_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved")

if __name__ == '__main__':
    train()
