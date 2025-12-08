import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=8, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.num_classes = num_classes
        
        # GAT Layer 1
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=self.dropout, add_self_loops=True)
        # GAT Layer 2
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=self.dropout, add_self_loops=True)
        # GAT Layer 3
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=self.dropout, add_self_loops=True)
        
        out_dim = 1 if num_classes == 2 else num_classes
        self.lin = torch.nn.Linear(hidden_channels, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GAT Layer 1
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # GAT Layer 2
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        # GAT Layer 3
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        
        # Pooling (Summary node)
        # summary_nodes = x[data.ptr[:-1]]

        # Select Topic nodes 
        if hasattr(data, 'topic_mask'):
            mask = data.topic_mask 
        else:
            mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
        x_topics = x[mask]
        
        batch_topics = batch[mask]
        out = global_mean_pool(x_topics, batch_topics)

        # Classification
        # out = F.dropout(summary_nodes, p=0.5, training=self.training)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.lin(out)
        
        if self.num_classes == 2:
            out = out.squeeze(-1)
            
        return out