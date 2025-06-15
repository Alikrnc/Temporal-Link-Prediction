import pandas as pd
import numpy as np
import random
from tqdm import tqdm # Import tqdm

NODE_FINAL_PATH = "data/final/node_final.parquet"
TRAIN_FINAL_PATH = "data/final/train_final.parquet"

def load_data():
    node_df = pd.read_parquet(NODE_FINAL_PATH)
    train_df = pd.read_parquet(TRAIN_FINAL_PATH)
    return node_df, train_df

def generate_training_samples(node_df, train_df, num_negative_samples_per_positive=1):
    """
    Generates training samples with positive and negative examples.
    num_negative_samples_per_positive (int): Number of negative samples to generate for each positive sample.

    Returns:
        pd.DataFrame: A DataFrame with columns ['src_id', 'dst_id', 'edge_id', 't_rel', 'label']
    """
    
    # Sort by t_rel to maintain temporal order
    train_sorted = train_df.sort_values(by='t_rel').reset_index(drop=True)
    all_node_ids = node_df['node_id'].tolist()
    print("Building existing edges set")
    existing_edges_set = set(zip(train_df['src_id'], train_df['dst_id'], train_df['edge_id']))

    positive_samples = []
    negative_samples = []
    seen_nodes_set = set()
    seen_nodes_list = []
    print("\nGenerating samples")
    for row in tqdm(train_sorted.itertuples(index=False), total=len(train_sorted), desc="Generating samples"):
        src, dst, edge_id_val, t_rel = row.src_id, row.dst_id, row.edge_id, row.t_rel
        positive_samples.append({'src_id': src, 'dst_id': dst, 'edge_id': edge_id_val, 't_rel': t_rel, 'label': 1})
        # Update seen nodes list and set
        for node in (src, dst):
            if node not in seen_nodes_set:
                seen_nodes_set.add(node)
                seen_nodes_list.append(node)
 
        # Generate negative samples
        neg_count = 0
        while neg_count < num_negative_samples_per_positive:
            # Randomly pick a destination node from seen nodes
            dst_neg = random.choice(seen_nodes_list) if seen_nodes_list else random.choice(all_node_ids)
            
            # Ensure dst_neg is different from dst and the edge (src, dst_neg, type) doesn't exist
            if dst_neg != dst and (src, dst_neg, edge_id_val) not in existing_edges_set:
                # Use the same t_rel as the positive sample to keep timestamp values discrete
                negative_t_rel = t_rel
                negative_samples.append({'src_id': src,
                                         'dst_id': dst_neg,
                                         'edge_id': edge_id_val,
                                         't_rel': negative_t_rel,
                                         'label': 0})
                neg_count += 1

    all_samples_df = pd.DataFrame(positive_samples + negative_samples)
    
    # Sort by t_rel then src_id to maintain temporal order and group by source
    all_samples_df = all_samples_df.sort_values(by=['t_rel', 'src_id']).reset_index(drop=True)
    return all_samples_df

if __name__ == '__main__':
    nodes, train_edges = load_data()
    
    training_data = generate_training_samples(nodes, train_edges, num_negative_samples_per_positive=1)
    print(f"Generated a total of {len(training_data)} samples")
    print("Sample of the generated data:")
    print(training_data.head())

    training_data.to_parquet("data/final/train_neg_samp.parquet", index=False)
