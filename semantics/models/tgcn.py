import numpy as np
from semantics.graphs.temporal_graph import TemporalGraph
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN
from typing import Optional
import tqdm
import yaml
import os



class TemporalGCN(torch.nn.Module):
    def __init__(self, node_features, edge_features):
        super(TemporalGCN, self).__init__()
        # (nodes, node_features) -> (nodes, 32)
        self.node_encoder = torch.nn.Linear(node_features, 32)  # Node feature encoder
        # (edges, edge_features) -> (edges, 32)
        self.edge_encoder = torch.nn.Linear(edge_features, 32)  # Edge feature encoder
        # ((nodes, 32), (edges, 32)) -> (nodes, 32)
        self.conv = TGCN(in_channels=32, out_channels=32)  # Temporal GNN layer
        # (edges, 96) -> (edges, 1)
        self.linear = torch.nn.Linear(96, 1)  # Linear layer for final edge prediction

    def forward(self, x, edge_index, edge_attr):
        # Encode node and edge features

        # (nodes, node_features) -> (nodes, 32)
        x_encoded = F.relu(self.node_encoder(x))

        # (edges, edge_features) -> (edges, 32)
        edge_attr_encoded = F.relu(self.edge_encoder(edge_attr))

        # Apply temporal graph convolution

        # (nodes, 32)
        h = self.conv(x_encoded, edge_index)
        

        # Aggregate node features for each edge
        row, col = edge_index

        # (edges, 32), (edges, 32) -> (edges, 64)
        edge_h = torch.cat([h[row], h[col]], dim=1)
        
        # Combine aggregated node features with edge features

        # (edges, 64), (edges, 32) -> (edges, 96)
        combined_edge_features = torch.cat([edge_h, edge_attr_encoded], dim=1)
        
        # Predict edge similarity
        return self.linear(combined_edge_features)
 

class TemporalGCNTrainer:
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
                print(f'y_indices: {snapshot.y_indices.shape}')
                print('\n')

                # Move the snapshot data to the specified device
                snapshot = snapshot.to(self.device)

                # Forward pass: compute the model's output
                # Input shape: x: [number_of_nodes, node_features], edge_index: [2, number_of_edges], edge_attr: [number_of_edges, edge_features]
                # Output shape: y_hat: [number_of_edges, 1]
                y_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        
                # Compute mean squared error loss
                labeled_edges = snapshot.y_indices.T.to(torch.int).tolist()
                labeled_edge_index = []
                for idx, edge in enumerate(snapshot.edge_index.T):
                    if edge.tolist() in labeled_edges:
                        labeled_edge_index.append(idx)

                y_hat = y_hat[labeled_edge_index]
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
            config = {
                "node_features": self.node_features,
                "edge_features": self.edge_features,
                "epochs": self.epochs,
                "split_ratio": self.split_ratio,
                "train_loss": avg_loss,
                "val_loss": val_loss
            }
            torch.save(self.model.state_dict(), f"{output_dir}.pt")
            with open(f"{output_dir}.yaml", "w") as f:
                yaml.dump(config, f)

    def evaluate(self, test_dataset: DynamicGraphTemporalSignal):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for snapshot in test_dataset:
                snapshot = snapshot.to(self.device)
                y_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

                # Compute mean squared error loss
                labeled_edges = snapshot.y_indices.T.to(torch.int).tolist()
                labeled_edge_index = []
                for idx, edge in enumerate(snapshot.edge_index.T):
                    if edge.tolist() in labeled_edges:
                        labeled_edge_index.append(idx)

                y_hat = y_hat[labeled_edge_index]
                loss = F.mse_loss(y_hat, snapshot.y.view(-1, 1))

                total_loss += loss.item()
        return total_loss / sum(1 for _ in test_dataset)


class TGCNInference:
    def __init__(
            self,
            pretrained_model_path: str,
            **config
        ):
        self.node_features = config.get("node_features", 771)
        self.edge_features = config.get("edge_features", 4)
        

        self.model_path = pretrained_model_path
        if not os.path.exists(pretrained_model_path):
            raise ValueError(
                f'The path {pretrained_model_path} does not exist'
            )
        

        self.model = None
        self.vocab = False
        self._tgcn_preparation()

    def _tgcn_preparation(self) -> None:
        self.model = TemporalGCN(self.node_features, self.edge_features)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        self.vocab = True


    def predict(self, graph: TemporalGraph) -> TemporalGraph:
        if not self.vocab:
            raise ValueError(
                'The model is not loaded'
            )
        
        dataset = DynamicGraphTemporalSignal(
            graph.edge_indices, graph.edge_features, graph.xs, graph.ys, y_indices= graph.y_indices
        )

        y_hat = []
        for snapshot in dataset:
            snapshot = snapshot.to('cpu')
            y_hat.append(self.model(snapshot.x, snapshot.edge_index, snapshot.edge_attr))
        
        return y_hat


    def mse_loss(self, y_hat, y):
        if isinstance(y_hat, list):
            y_hat = torch.cat(y_hat)
        if isinstance(y, list):
            y = torch.cat(y)
        
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        
        
        return F.mse_loss(y_hat, y)
