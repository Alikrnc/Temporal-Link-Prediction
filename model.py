import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import GConvLSTM

class TemporalLinkPredictor(nn.Module):
    def __init__(self,
                 num_nodes,
                 num_edge_types,
                 
                 # Cardinalities for categorical features
                 node_categorical_cardinalities, 
                 edge_type_categorical_cardinalities,
                 
                 # Embedding dimensions
                 node_continuous_feature_dim=2, 
                 node_cat_embed_dim=16,        
                 edge_type_itself_embed_dim=32, 
                 edge_type_cat_embed_dim=16,   
                 time_embed_dim=16,
                 
                 # GNN and prediction dimensions
                 gnn_hidden_dim=128, # Dimension of GNN output and LSTM hidden state
                 prediction_hidden_dim=64):
        super(TemporalLinkPredictor, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_edge_types = num_edge_types

        # Node Feature Embedding
        self.node_categorical_embedding_layers = nn.ModuleList([
            nn.Embedding(num_categories, node_cat_embed_dim) for num_categories in node_categorical_cardinalities
        ])
        self.processed_node_feature_dim = node_continuous_feature_dim + (len(node_categorical_cardinalities) * node_cat_embed_dim)

        # Edge Feature Embedding
        self.edge_id_embedding_layer = nn.Embedding(num_edge_types, edge_type_itself_embed_dim)
        self.edge_type_categorical_embedding_layers = nn.ModuleList([
            nn.Embedding(num_categories, edge_type_cat_embed_dim) for num_categories in edge_type_categorical_cardinalities
        ])
        self.processed_edge_type_feature_dim = edge_type_itself_embed_dim + (len(edge_type_categorical_cardinalities) * edge_type_cat_embed_dim)
        
        self.time_encoder = nn.Linear(1, time_embed_dim)

        # Temporal GNN layer
        self.temporal_gnn = GConvLSTM(
            in_channels=self.processed_node_feature_dim,
            out_channels=gnn_hidden_dim, # This is the LSTM hidden size and GNN output embedding size
            K=1 # K=1 for GCN-like convolution within GConvLSTM
        )

        # Prediction head
        predictor_input_dim = (gnn_hidden_dim * 2) + self.processed_edge_type_feature_dim + time_embed_dim
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, prediction_hidden_dim),
            nn.ReLU(),
            nn.Linear(prediction_hidden_dim, 1),
            nn.Sigmoid()
        )

    def _process_node_features(self, raw_node_features_continuous, raw_node_features_categorical_codes):
        """
        Helper function to process raw node features for all nodes.
        Args:
            raw_node_features_continuous (Tensor): Continuous features [num_nodes, node_continuous_feature_dim]
            raw_node_features_categorical_codes (list of Tensors): Categorical codes [num_nodes] for each cat feature.
        Returns:
            Tensor: Processed node features [num_nodes, self.processed_node_feature_dim]
        """
        node_cat_embeds = []
        for i, codes_tensor in enumerate(raw_node_features_categorical_codes):
            # codes_tensor should be LongTensor for nn.Embedding
            node_cat_embeds.append(self.node_categorical_embedding_layers[i](codes_tensor.long())) 
        
        all_node_cat_embedded = torch.cat(node_cat_embeds, dim=-1)
        return torch.cat([raw_node_features_continuous, all_node_cat_embedded], dim=-1)

    def _process_batch_edge_type_features(self, batch_edge_type_ids, raw_edge_type_features_codes_full):
        """
        Helper function to process features for edge types in the current batch.
        Args:
            batch_edge_type_ids (Tensor): Edge type IDs for the current batch [batch_size].
            raw_edge_type_features_codes_full (list of Tensors): Full list of categorical codes for all edge types.
                                                              Each tensor is [num_edge_types].
        Returns:
            Tensor: Processed edge type features for the batch [batch_size, self.processed_edge_type_feature_dim]
        """
        # Embed the edge_ids themselves
        batch_edge_id_embeds = self.edge_id_embedding_layer(batch_edge_type_ids.long()) # [batch_size, edge_type_itself_embed_dim]

        # Embed the categorical features associated with edge_ids
        batch_cat_feature_embeds = []
        for i, full_codes_for_feature_i in enumerate(raw_edge_type_features_codes_full):
            # Get the codes for the edge_ids in the current batch
            codes_for_batch_edges = full_codes_for_feature_i[batch_edge_type_ids.long()]
            batch_cat_feature_embeds.append(self.edge_type_categorical_embedding_layers[i](codes_for_batch_edges.long()))
        
        concatenated_cat_embeds = torch.cat(batch_cat_feature_embeds, dim=-1) # [batch_size, num_edge_cat_feats * edge_type_cat_embed_dim]
        return torch.cat([batch_edge_id_embeds, concatenated_cat_embeds], dim=-1)


    def forward(self, src_node_ids, dst_node_ids, edge_type_ids, t_rel,
                raw_all_node_features_cont,             
                raw_all_node_features_cat_codes_list,   
                raw_all_edge_type_features_cat_codes_list,
                current_graph_edge_index, 
                h_prev, 
                c_prev
                ):
        """
        Forward pass for link prediction.
        Args:
            src_node_ids (Tensor): Source node IDs for the links to predict [batch_size].
            dst_node_ids (Tensor): Destination node IDs [batch_size].
            edge_type_ids (Tensor): Edge type IDs for the links [batch_size].
            t_rel (Tensor): Relative timestamps [batch_size, 1] or [batch_size].
            raw_all_node_features_cont (Tensor): Continuous features for ALL nodes [num_nodes, node_continuous_feature_dim].
            raw_all_node_features_cat_codes_list (list of Tensors): Categorical codes for ALL nodes. Each tensor is [num_nodes].
            raw_all_edge_type_features_cat_codes_list (list of Tensors): Categorical codes for ALL edge types. Each tensor is [num_edge_types].
            current_graph_edge_index (Tensor): Edge index for GNN at this time step [2, num_current_edges].
            h_prev (Tensor): Previous hidden state for GConvLSTM [num_nodes, gnn_hidden_dim].
            c_prev (Tensor): Previous cell state for GConvLSTM [num_nodes, gnn_hidden_dim].
        Returns:
            link_probability (Tensor): Predicted link probabilities [batch_size].
            h_new (Tensor): New hidden state from GConvLSTM [num_nodes, gnn_hidden_dim].
            c_new (Tensor): New cell state from GConvLSTM [num_nodes, gnn_hidden_dim].
        """

        processed_all_node_features = self._process_node_features(
            raw_all_node_features_cont,
            raw_all_node_features_cat_codes_list
        )

        processed_batch_edge_features = self._process_batch_edge_type_features(
            edge_type_ids,
            raw_all_edge_type_features_cat_codes_list
        )
        
        # Encode time
        if t_rel.ndim == 1:
            t_rel = t_rel.unsqueeze(-1)
        time_encoded = self.time_encoder(t_rel) # [batch_size, time_embed_dim]

        # H_new contains the updated node embeddings (hidden states)
        h_new, c_new = self.temporal_gnn(
            processed_all_node_features, # X: node features at current time t
            current_graph_edge_index,    # edge_index for GCN-like convolution
            None,                        # edge_weight
            h_prev,                      # H: previous LSTM hidden state
            c_prev                       # C: previous LSTM cell state
        )
        
        evolved_src_embeddings = h_new[src_node_ids.long()] # [batch_size, gnn_hidden_dim]
        evolved_dst_embeddings = h_new[dst_node_ids.long()] # [batch_size, gnn_hidden_dim]
        
        combined_input = torch.cat([evolved_src_embeddings, evolved_dst_embeddings, 
                                     processed_batch_edge_features, time_encoded], dim=-1)
        
        link_probability = self.predictor(combined_input)
        return link_probability.squeeze(-1), h_new, c_new