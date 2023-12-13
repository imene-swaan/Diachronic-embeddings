from ast import Not
from semantics.graphs.temporal_graph import TemporalGraph
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN
from typing import Optional
import tqdm




class TemporalGCN(torch.nn.Module):
    def __init__(self, node_features, edge_features):
        super(TemporalGCN, self).__init__()
        # (nodes, node_features) -> (nodes, 32)
        self.node_encoder = torch.nn.Linear(node_features, 32)  # Node feature encoder
        # (edges, edge_features) -> (edges, 32)
        self.edge_encoder = torch.nn.Linear(edge_features, 32)  # Edge feature encoder
        # 
        self.conv = TGCN(in_channels=32, out_channels=32)  # Temporal GNN layer
        # ()
        self.linear = torch.nn.Linear(96, 1)  # Linear layer for final edge prediction

    def forward(self, x, edge_index, edge_attr):
        # Encode node and edge features
        print(f'x: {x.shape}')
        x_encoded = F.relu(self.node_encoder(x))
        print(f'x_encoded: {x_encoded.shape}')

        print(f'edge_attr: {edge_attr.shape}')
        edge_attr_encoded = F.relu(self.edge_encoder(edge_attr))
        print(f'edge_attr_encoded: {edge_attr_encoded.shape}')


        # Apply temporal graph convolution
        h = self.conv(x_encoded, edge_index)
        print(f'h: {h.shape}')

        # Aggregate node features for each edge
        print(f'edge_index: {edge_index.shape}')
        row, col = edge_index
        print(f'row: {row.shape}')
        print(f'col: {col.shape}')

        print(f'h[row]: {h[row].shape}')
        print(f'h[col]: {h[col].shape}')
        edge_h = torch.cat([h[row], h[col]], dim=1)
        print(f'edge_h: {edge_h.shape}')

        # Combine aggregated node features with edge features
        combined_edge_features = torch.cat([edge_h, edge_attr_encoded], dim=1)
        print(f'combined_edge_features: {combined_edge_features.shape}')
        
        # Predict edge similarity
        return self.linear(combined_edge_features)
 

class TemporalGNNTrainer:
    def __init__(
            self, 
            node_features: int = 771,
            edge_features: int = 4,
            epochs: int = 10,
            split_ratio: float = 0.8,
            learning_rate: float = 0.01,
            device: str = "cpu",
            # subset: Optional[int] = None
            ):
        self.node_features = node_features
        self.edge_features = edge_features
        self.epochs = epochs
        self.split_ratio = split_ratio
        # self.learning_rate = learning_rate
        self.device = device
        # self.subset = subset

        self.model = TemporalGCN(node_features, edge_features).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= learning_rate)
    
    def train(
            self, 
            graph: TemporalGraph,
            output_dir: Optional[str] = None
            ) -> None:
        
        dataset = DynamicGraphTemporalSignal(
            graph.edge_indices, graph.edge_features, graph.xs, graph.ys, y_indices= graph.y_indices
        )


        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=self.split_ratio)

        
        for epoch in range(self.epochs):
            total_loss = 0
            self.model.train()

            progress_bar = tqdm.tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False)

            for snapshot in progress_bar:
                print('\nShapes:')
                print(f'x: {snapshot.x.shape}')
                print(f'edge_index: {snapshot.edge_index.shape}')
                print(f'edge_attr: {snapshot.edge_attr.shape}')
                print(f'y: {snapshot.y.shape}')

                print('\n')
                # Move the snapshot data to the specified device
                snapshot = snapshot.to(self.device)
                # Forward pass: compute the model's output
                # Input shape: x: [number_of_nodes, node_features], edge_index: [2, number_of_edges], edge_attr: [number_of_edges, edge_features]
                # Output shape: y_hat: [number_of_edges, 1]
                y_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                # Compute mean squared error loss
                # Input shape: y_hat: [number_of_edges, 1], snapshot.y.view(-1, 1): [number_of_edges, 1]
                loss = F.mse_loss(y_hat, snapshot.y.view(-1, 1))
                # Zero the gradients before backward pass
                self.optimizer.zero_grad()
                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # Perform a single optimization step (parameter update)
                self.optimizer.step()
                # Accumulate the loss
                total_loss += loss.item()

                # Update progress bar description with the current loss
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            # Compute average loss for the epoch
            avg_loss = total_loss / sum(1 for _ in train_dataset)
            # Print average loss for the epoch
            # print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {avg_loss:.4f}")
            val_loss = self.evaluate(test_dataset)
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if output_dir:
            torch.save(self.model.state_dict(), f"{output_dir}.pt")

    def evaluate(self, test_dataset: DynamicGraphTemporalSignal):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for snapshot in test_dataset:
                snapshot = snapshot.to(self.device)
                y_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                loss = F.mse_loss(y_hat, snapshot.y.view(-1, 1))
                total_loss += loss.item()
        return total_loss / sum(1 for _ in test_dataset)
