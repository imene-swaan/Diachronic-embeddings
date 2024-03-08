import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import TGCN
from typing import Optional, List, Literal
import tqdm
import yaml
import os
import numpy as np
from semantics.graphs.temporal_graph import TemporalGraph
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split




class TemporalLinkPredictionModel(nn.Module):
    def __init__(
            self, 
            num_node_features: int, 
            num_edge_features: int,
            out_channels: int
            ):
        super(TemporalLinkPredictionModel, self).__init__()

        self.node_encoder: nn.Module = nn.Linear(num_node_features, out_channels)
        self.edge_encoder: nn.Module = nn.Linear(num_edge_features, out_channels)

        # self.conv: nn.Module = GCNConv(out_channels, out_channels)
        self.temporal_conv: nn.Module = TGCN(out_channels, out_channels)


        self.decoder = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(
            self, 
            x: torch.Tensor, 
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
            return_embedding: bool = False,
            apply_threshold: bool = False
            ):

        x_encoded = F.relu(self.node_encoder(x))
        edge_attr_encoded = F.relu(self.edge_encoder(edge_attr))
        
        # x_conv: torch.Tensor = self.conv(x_encoded, edge_index, edge_attr_encoded)

        x_temporal: torch.Tensor = self.temporal_conv(
            x_encoded, 
            edge_index)

        

        if return_embedding:
            return x_temporal
        
        num_nodes = x.size(0)
        indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
        
        x_i = x_temporal[indices[0]]
        x_j = x_temporal[indices[1]]

        node_pairs = torch.cat([x_i, x_j], dim=-1)

        predictions = self.decoder(node_pairs)

        if apply_threshold:
            predictions = torch.ge(predictions, 0.5).float()
        
        return predictions
        
        

        
       

class TemporalGCNTrainer:
    def __init__(
            self, 
            node_features: int = 770,
            edge_features: int = 3,
            size: int = 128,
            epochs: int = 10,
            split_ratio: float = 0.8,
            learning_rate: float = 0.01,
            device: str = "cpu"
            # subset: Optional[int] = None
            ):
        self.node_features = node_features
        self.edge_features = edge_features
        self.epochs = epochs
        self.split_ratio = split_ratio
        self.learning_rate = learning_rate
        self.device = device
        self.size = size

        self.criterion = nn.BCELoss()

        self.model = TemporalLinkPredictionModel(node_features, edge_features, size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= float(learning_rate))
    
    def train(
            self, 
            graph: TemporalGraph,
            output_dir: Optional[str] = None,
            ) -> None:
        
        dataset = DynamicGraphTemporalSignal(
            graph.edge_indices,
            graph.edge_features,
            graph.xs,
            graph.ys,
            y_indices = graph.y_indices
        )


        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=self.split_ratio)
        best_val_loss = float('inf')  # For checkpointing
    
        losses = []
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
                snapshot = snapshot.to(self.device)

                y = snapshot.y.view(-1, 1).float()
                y_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

                loss = self.criterion(y_hat, y)

                # Zero the gradients before backward pass
                self.optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # Perform a single optimization step (parameter update)
                self.optimizer.step()

                losses.append(loss.item())
                # Accumulate the loss
                total_loss += loss.item()

                # Update progress bar description with the current loss
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            # Compute average loss for the epoch
            avg_loss = total_loss / sum(1 for _ in train_dataset)
            # Print average loss for the epoch
            # print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {avg_loss:.4f}")
            # val_loss = self.evaluate(test_dataset)
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_loss:.4f}") #, Val Loss: {val_loss:.4f}")
        
        if output_dir:
            config = {
                "node_features": self.node_features,
                "edge_features": self.edge_features,
                "size": self.size,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "split_ratio": self.split_ratio,
                # "loss_func": self.loss_func,
                "train_loss": avg_loss,
                "losses": losses
                # "val_loss": val_loss
            }
            torch.save(self.model.state_dict(), f"{output_dir}.pt")
            with open(f"{output_dir}.yaml", "w") as f:
                yaml.dump(config, f)

    def evaluate(self, test_dataset: DynamicGraphTemporalSignal):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for snapshot in test_dataset:
                print('\nShapes:')
                print(f'x: {snapshot.x.shape}')
                print(f'edge_index: {snapshot.edge_index.shape}')
                print(f'edge_attr: {snapshot.edge_attr.shape}')
                print(f'y: {snapshot.y.shape}')
                print(f'y_indices: {snapshot.y_indices.shape}')
                print('\n')
                snapshot = snapshot.to(self.device)

                y = snapshot.y.view(-1, 1).float().to(self.device)
                print('y', y.shape)
                
                y_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, apply_threshold=False)
                print('y_hat', y_hat.shape)
                

                loss = self.criterion(y_hat, y)
                print('loss', loss)
                total_loss += loss.item()

        return total_loss / sum(1 for _ in test_dataset)
    



class LPInference:
    def __init__(
            self,
            pretrained_model_path: str,
            **config
        ):
        self.node_features = config.get("node_features", 770)
        self.edge_features = config.get("edge_features", 3)
        self.size = config.get("size", 128)
        

        self.model_path = pretrained_model_path
        if not os.path.exists(pretrained_model_path):
            raise ValueError(
                f'The path {pretrained_model_path} does not exist'
            )
        

        self.model = None
        self.vocab = False
        self._tgcn_preparation()

    def _tgcn_preparation(self) -> None:
        self.model = TemporalLinkPredictionModel(self.node_features, self.edge_features, self.size)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        self.vocab = True


    def predict(self, graph: TemporalGraph) -> list:
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
        
        
        return F.mse_loss(y_hat, y).item()
    
    def mae_loss(self, y_hat, y):
        if isinstance(y_hat, list):
            y_hat = torch.cat(y_hat)
        if isinstance(y, list):
            y = torch.cat(y)
        
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        
        return F.l1_loss(y_hat, y).item()


    def get_embedding(
            self, 
            graph: TemporalGraph, 
            to_vector: Optional[Literal['flatten', 'mean', 'max']] = None
            ) -> List[np.ndarray]:
        if not self.vocab:
            raise ValueError(
                'The model is not loaded'
            )
        
        dataset = DynamicGraphTemporalSignal(
            graph.edge_indices, graph.edge_features, graph.xs, graph.ys, y_indices= graph.y_indices
        )

        embeddings = []
        for snapshot in dataset:
            snapshot = snapshot.to('cpu')
            # embeddings.append(self.model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, return_embedding=True))

            conv: torch.Tensor = self.model(
                snapshot.x, 
                snapshot.edge_index, 
                snapshot.edge_attr, 
                return_embedding=True
                )
            
            conv = conv.detach().numpy()
            if to_vector is None:
                embeddings.append(conv)
                continue

            if to_vector == 'mean':
                emb = conv.mean(axis=0)
            elif to_vector == 'max':
                emb = conv.max(axis=0)
            elif to_vector == 'flatten':
                emb = conv.flatten()
            else:
                raise ValueError('Unknown to_vector value')
        
            embeddings.append(emb)
        
        return embeddings
    

    

    
