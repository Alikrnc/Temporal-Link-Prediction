import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import os
from tqdm import tqdm
from model import TemporalLinkPredictor 

MODEL_PATH = os.path.join("temporal_link_predictor_best_model.pth")
SUBMISSION_RAW_PATH = os.path.join("data/processed/submission.parquet")
NODE_FINAL_PATH = os.path.join("data/final/node_final.parquet")
EDGE_FINAL_PATH = os.path.join("data/final/edge_final.parquet")
TRAIN_A_PATH = os.path.join("data/processed/train_a.parquet") # For t_min, t_max
HISTORICAL_EDGES_PATH = os.path.join("data/final/train_final.parquet")
OUTPUT_PATH = os.path.join("output_A.csv")

# These should match the ones used in model.py
MODEL_PARAMS_SUBMISSION = {
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

NODE_ID_COL = "node_id"
NODE_CAT_FEATURE_COLS_FINAL = [f"node_feature_{i}_code" for i in range(1, 9)]
NODE_CONT_FEATURE_COLS_FINAL = ['in_deg_norm', 'out_deg_norm']

EDGE_ID_COL_SUBMISSION = "edge_id"
EDGE_ID_COL_FEATURES = "edge_id"
EDGE_CAT_FEATURE_COLS_FINAL = [f"edge_feature_{i}_code" for i in range(1, 4)]


TIMESTAMP_COL_TRAIN_A = "timestamp"
START_TIME_COL_SUBMISSION = "start_time"
END_TIME_COL_SUBMISSION = "end_time"
LABEL_COL_SUBMISSION = "label"

NUM_PREDICTION_POINTS_IN_INTERVAL = 1 # Use single point matching training
HISTORY_WINDOW_T_REL = 0.15 # Window for historical graph construction (match train.py)

def load_and_preprocess_data():
    node_final_df = pd.read_parquet(NODE_FINAL_PATH)
    edge_final_df = pd.read_parquet(EDGE_FINAL_PATH)
    submission_raw_df = pd.read_parquet(SUBMISSION_RAW_PATH)
    submission_raw_df['original_order_col'] = range(len(submission_raw_df))
    
    train_a_df = pd.read_parquet(TRAIN_A_PATH)
    historical_edges_df = pd.read_parquet(HISTORICAL_EDGES_PATH)

    original_to_remapped_id_map = pd.Series(node_final_df.index, index=node_final_df[NODE_ID_COL]).to_dict()
    node_final_df['idx'] = node_final_df[NODE_ID_COL].map(original_to_remapped_id_map)
    node_final_df = node_final_df.sort_values(by='idx').set_index('idx', drop=False) # Keep 'idx' as a column
    num_nodes = len(original_to_remapped_id_map)
    print(f"Remapped node IDs. Total unique nodes: {num_nodes}")

    edge_final_df = edge_final_df.sort_values(by=EDGE_ID_COL_FEATURES).reset_index(drop=True)
    edge_type_to_code = pd.Series(edge_final_df.index, index=edge_final_df[EDGE_ID_COL_FEATURES]).to_dict()
    num_edge_types = len(edge_type_to_code)
    print(f"Mapped edge types. Total unique edge types: {num_edge_types}")

    print("Performing feature engineering")
    t_min = train_a_df[TIMESTAMP_COL_TRAIN_A].min()
    t_max = train_a_df[TIMESTAMP_COL_TRAIN_A].max()
    
    # Calculate t_rel for start_time and end_time
    submission_raw_df['t_rel_start'] = (submission_raw_df[START_TIME_COL_SUBMISSION] - t_min) / (t_max - t_min)
    submission_raw_df['t_rel_start'] = submission_raw_df['t_rel_start'].fillna(0).clip(0, 1)
    
    submission_raw_df['t_rel_end'] = (submission_raw_df[END_TIME_COL_SUBMISSION] - t_min) / (t_max - t_min)
    submission_raw_df['t_rel_end'] = submission_raw_df['t_rel_end'].fillna(0).clip(0, 1)
    submission_raw_df['t_rel_end'] = np.maximum(submission_raw_df['t_rel_start'], submission_raw_df['t_rel_end'])
    submission_raw_df['t_rel'] = submission_raw_df['t_rel_start']

    submission_raw_df['src_idx'] = submission_raw_df['src_id'].map(original_to_remapped_id_map)
    submission_raw_df['dst_idx'] = submission_raw_df['dst_id'].map(original_to_remapped_id_map)
    submission_raw_df['edge_type_code'] = submission_raw_df[EDGE_ID_COL_SUBMISSION].map(edge_type_to_code)

    submission_raw_df.dropna(subset=['src_idx', 'dst_idx', 'edge_type_code'], inplace=True)
        
    submission_raw_df['src_idx'] = submission_raw_df['src_idx'].astype(int)
    submission_raw_df['dst_idx'] = submission_raw_df['dst_idx'].astype(int)
    submission_raw_df['edge_type_code'] = submission_raw_df['edge_type_code'].astype(int)

    # Prepare node features
    raw_node_cont_features = torch.tensor(node_final_df[NODE_CONT_FEATURE_COLS_FINAL].values, dtype=torch.float32)
    raw_node_cat_features_list = [
        torch.tensor(node_final_df[col].values, dtype=torch.long) for col in NODE_CAT_FEATURE_COLS_FINAL
    ]

    # Prepare edge type features
    raw_edge_type_cat_features_list = [
        torch.tensor(edge_final_df[col].values, dtype=torch.long) for col in EDGE_CAT_FEATURE_COLS_FINAL
    ]
    
    # Prepare historical_edges_df for graph construction
    historical_edges_df['src_idx'] = historical_edges_df['src_id'].map(original_to_remapped_id_map)
    historical_edges_df['dst_idx'] = historical_edges_df['dst_id'].map(original_to_remapped_id_map)
    historical_edges_df.dropna(subset=['src_idx', 'dst_idx'], inplace=True)
    historical_edges_df['src_idx'] = historical_edges_df['src_idx'].astype(int)
    historical_edges_df['dst_idx'] = historical_edges_df['dst_idx'].astype(int)
    historical_edges_df_sorted = historical_edges_df.sort_values(by='t_rel').reset_index(drop=True)
    
    if not historical_edges_df_sorted.empty:
        historical_edges_src_tensor_cpu = torch.tensor(historical_edges_df_sorted['src_idx'].values, dtype=torch.long)
        historical_edges_dst_tensor_cpu = torch.tensor(historical_edges_df_sorted['dst_idx'].values, dtype=torch.long)
    else:
        historical_edges_src_tensor_cpu = torch.empty((0,), dtype=torch.long)
        historical_edges_dst_tensor_cpu = torch.empty((0,), dtype=torch.long)
    
    print("Data loading and preprocessing complete.")
    return (submission_raw_df, 
            raw_node_cont_features, raw_node_cat_features_list,
            raw_edge_type_cat_features_list,
            historical_edges_df_sorted,
            historical_edges_src_tensor_cpu, historical_edges_dst_tensor_cpu,
            num_nodes, num_edge_types)


def predict(model, device, submission_df_processed, 
            raw_node_cont_features_all, raw_node_cat_features_list_all,
            raw_edge_type_cat_features_list_all,
            historical_edges_sorted,
            historical_edges_src_tensor_cpu, historical_edges_dst_tensor_cpu,
            num_nodes, gnn_hidden_dim):
    print("Starting prediction")
    model.eval()
    all_final_predictions = []
    
    submission_df_sorted = submission_df_processed.sort_values(by='t_rel_start').reset_index(drop=True)

    # Move static features to device ONCE and process node features ONCE
    raw_node_cont_features_all_dev = raw_node_cont_features_all.to(device)
    raw_node_cat_features_list_all_dev = [tensor.to(device) for tensor in raw_node_cat_features_list_all]
    raw_edge_type_cat_features_list_all_dev = [tensor.to(device) for tensor in raw_edge_type_cat_features_list_all]

    # Process all node features ONCE (as they are static for all queries)
    processed_all_node_features_dev = model._process_node_features(
        raw_node_cont_features_all_dev,
        raw_node_cat_features_list_all_dev
    )

    h_prev = torch.zeros((num_nodes, gnn_hidden_dim), device=device)
    c_prev = torch.zeros((num_nodes, gnn_hidden_dim), device=device)

    historical_t_rel_sorted_np = historical_edges_sorted['t_rel'].values

    with torch.no_grad():
        for i, query in tqdm(submission_df_sorted.iterrows(), total=len(submission_df_sorted), desc="Predicting Queries"):
            t_rel_start_query = query['t_rel_start']
            t_rel_end_query = query['t_rel_end']

            prediction_time_points = []
            if NUM_PREDICTION_POINTS_IN_INTERVAL == 1:
                prediction_time_points.append(t_rel_start_query)
            elif NUM_PREDICTION_POINTS_IN_INTERVAL == 2:
                prediction_time_points.append(t_rel_start_query)
                prediction_time_points.append(t_rel_end_query)
            else: 
                prediction_time_points = np.linspace(t_rel_start_query, t_rel_end_query, NUM_PREDICTION_POINTS_IN_INTERVAL).tolist()
            prediction_time_points = sorted(list(set(prediction_time_points)))

            lower_bound_t_rel_hist = max(0, t_rel_start_query - HISTORY_WINDOW_T_REL)
            start_idx_hist = np.searchsorted(historical_t_rel_sorted_np, lower_bound_t_rel_hist, side='left')
            upper_bound_idx_hist = np.searchsorted(historical_t_rel_sorted_np, t_rel_start_query, side='left')
            
            if upper_bound_idx_hist > start_idx_hist:
                src_hist = historical_edges_src_tensor_cpu[start_idx_hist:upper_bound_idx_hist].to(device)
                dst_hist = historical_edges_dst_tensor_cpu[start_idx_hist:upper_bound_idx_hist].to(device)
                current_graph_edge_index_for_interval = torch.stack([src_hist, dst_hist], dim=0)
            else:
                current_graph_edge_index_for_interval = torch.empty((2, 0), dtype=torch.long, device=device)

            src_idx_tensor = torch.tensor([query['src_idx']], dtype=torch.long, device=device)
            dst_idx_tensor = torch.tensor([query['dst_idx']], dtype=torch.long, device=device)
            edge_type_code_tensor = torch.tensor([query['edge_type_code']], dtype=torch.long, device=device)

            h_temp, c_temp = h_prev, c_prev 

            h_intermediate_for_interval, _ = model.temporal_gnn(
                processed_all_node_features_dev, 
                current_graph_edge_index_for_interval,
                None, 
                h_temp, 
                c_temp
            )
            
            evolved_src_embeddings_common = h_intermediate_for_interval[src_idx_tensor.long()]
            evolved_dst_embeddings_common = h_intermediate_for_interval[dst_idx_tensor.long()]
            
            processed_batch_edge_features_common = model._process_batch_edge_type_features(
                edge_type_code_tensor,
                raw_edge_type_cat_features_list_all_dev
            )

            query_interval_predictions = []
            for point_t_rel in prediction_time_points:
                t_rel_tensor = torch.tensor([point_t_rel], dtype=torch.float32, device=device)
                if t_rel_tensor.ndim == 1:
                    t_rel_tensor = t_rel_tensor.unsqueeze(-1)

                time_encoded_point = model.time_encoder(t_rel_tensor)

                combined_input = torch.cat([
                    evolved_src_embeddings_common, 
                    evolved_dst_embeddings_common, 
                    processed_batch_edge_features_common, 
                    time_encoded_point
                ], dim=-1)
                
                prob_item = model.predictor(combined_input).squeeze(-1).item() # Ensure squeeze if predictor output is [1,1]
                query_interval_predictions.append(prob_item)

            if not query_interval_predictions:
                final_prob_for_query = 0.0
            else:
                final_prob_for_query = np.max(query_interval_predictions)
            all_final_predictions.append(final_prob_for_query)

            # Update main h_prev, c_prev for the next query
            # This uses the h_prev, c_prev from *before* this query (the ones that are chronologically evolving)
            h_prev_updated, c_prev_updated = model.temporal_gnn(
                processed_all_node_features_dev,
                current_graph_edge_index_for_interval,
                None,
                h_prev, 
                c_prev  
            )
            h_prev, c_prev = h_prev_updated.detach(), c_prev_updated.detach()

    submission_df_sorted['probability'] = all_final_predictions
    print("Prediction finished.")
    return submission_df_sorted

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    (submission_df_processed, 
     raw_node_cont_features, raw_node_cat_features_list,
     raw_edge_type_cat_features_list,
     historical_edges_df_sorted, 
     historical_edges_src_tensor_cpu, historical_edges_dst_tensor_cpu, # New args from load_data
     num_nodes, num_edge_types) = load_and_preprocess_data()

    # Update num_nodes and num_edge_types in MODEL_PARAMS_SUBMISSION from loaded data
    current_model_params = MODEL_PARAMS_SUBMISSION.copy()
    current_model_params["num_nodes"] = num_nodes
    current_model_params["num_edge_types"] = num_edge_types
    
    # Ensure cardinalities match the actual data if necessary
    actual_node_cardinalities = [int(raw_node_cat_features_list[i].max().item() + 1) for i in range(len(raw_node_cat_features_list))]
    if actual_node_cardinalities != current_model_params["node_categorical_cardinalities"]:
        print(f"Warning: Mismatch in node categorical cardinalities. Model expects: {current_model_params['node_categorical_cardinalities']}, Data has: {actual_node_cardinalities}")
        # Override to actual
        current_model_params["node_categorical_cardinalities"] = actual_node_cardinalities

    # For edge_type_categorical_cardinalities
    actual_other_edge_type_cardinalities = [int(raw_edge_type_cat_features_list[i].max().item() + 1) if raw_edge_type_cat_features_list[i].numel() > 0 else 0 for i in range(len(raw_edge_type_cat_features_list))]
    
    # Ensure the number of features matches
    if len(actual_other_edge_type_cardinalities) != len(current_model_params["edge_type_categorical_cardinalities"]):
        print(f"Warning: Mismatch in the number of edge type categorical features. Model expects: {len(current_model_params['edge_type_categorical_cardinalities'])}, Data has: {len(actual_other_edge_type_cardinalities)}")
    elif actual_other_edge_type_cardinalities != current_model_params["edge_type_categorical_cardinalities"]:
       print(f"Warning: Mismatch in edge type categorical cardinalities (for other features). Model expects: {current_model_params['edge_type_categorical_cardinalities']}, Data has: {actual_other_edge_type_cardinalities}")
       current_model_params["edge_type_categorical_cardinalities"] = actual_other_edge_type_cardinalities

    model = TemporalLinkPredictor(**current_model_params).to(device)

    results_df = predict(model, device, submission_df_processed, 
                         raw_node_cont_features, raw_node_cat_features_list,
                         raw_edge_type_cat_features_list,
                         historical_edges_df_sorted, # Pass new args
                         historical_edges_src_tensor_cpu, historical_edges_dst_tensor_cpu,
                         num_nodes, 
                         current_model_params["gnn_hidden_dim"])
    
    output_df = results_df[[
        'src_id', 'dst_id', EDGE_ID_COL_SUBMISSION, 
        START_TIME_COL_SUBMISSION, END_TIME_COL_SUBMISSION, 'probability'
    ]]
    
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved")

    if LABEL_COL_SUBMISSION in results_df.columns:
        valid_labels_df = results_df.dropna(subset=[LABEL_COL_SUBMISSION])
        y_true = valid_labels_df[LABEL_COL_SUBMISSION].astype(int)
        y_pred_proba = valid_labels_df['probability']

        if len(y_true) > 0 and len(y_true) == len(y_pred_proba):
            try:
                auc = roc_auc_score(y_true, y_pred_proba)
                print(f"AUC: {auc:.4f}")
            except ValueError as e:
                print(f"Could not calculate AUC: {e}. Check if labels are all of one class.")

            y_pred_binary = (y_pred_proba >= 0.5).astype(int)
            accuracy = accuracy_score(y_true, y_pred_binary)
            print(f"Accuracy (threshold 0.5): {accuracy:.4f}")
        else:
            print("Not enough data or mismatched lengths for evaluation after dropping NaNs.")
    else:
        print(f"\'{LABEL_COL_SUBMISSION}\' column not found in submission data.")

if __name__ == "__main__":
    main()
