import numpy as np
from semantics.graphs.temporal_graph import TemporalGraph
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split
import torch
# import torch.nn.functional as F
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import TGCN
from typing import Optional, List, Literal
# import tqdm
import yaml
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score




class TemporalGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalGCN, self).__init__()
        
        self.recurrent = TGCN(in_channels=in_channels, out_channels=out_channels)
        self.linear = torch.nn.Linear(out_channels*2, 1)


    def forward(
            self, 
            x, 
            edge_index, 
            edge_attr, 
            return_embedding=False,
            threshold: float = 0.5, 
            return_binary: bool = False
            ):
        
        h = self.recurrent(x, edge_index, edge_attr)
        # h = F.relu(h)
        h = self.linear(h)
        if return_embedding:
            return h
        
        probabilities = torch.sigmoid(h)
        if return_binary:
            return (probabilities > threshold).int()
        
        return probabilities
 

class TemporalGCNTrainer:
    def __init__(
            self, 
            node_features: int = 771,
            size: int = 128,
            epochs: int = 10,
            split_ratio: float = 0.8,
            learning_rate: float = 0.01,
            device: str = "cpu",
            criterion: str = 'BCELoss'
            ):
        self.node_features = node_features
        self.epochs = epochs
        self.split_ratio = split_ratio
        self.size = size

        if criterion == 'BCELoss':
            self.criterion = nn.BCELoss()
        
        else:
            raise ValueError('Unknown criterion value')

        self.model = TemporalGCN(node_features, size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= float(learning_rate))

        self.device = next(self.model.parameters()).device
    
    def train(
            self, 
            graph: TemporalGraph,
            output_dir: Optional[str] = None,
            ) -> None:
        
        dataset = DynamicGraphTemporalSignal(
            graph.edge_indices, graph.edge_features, graph.xs, graph.ys, y_indices= graph.y_indices
        )


        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=self.split_ratio)



        for epoch in range(self.epochs):
            total_loss = 0
            self.model.train()

            # progress_bar = tqdm.tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False)

            for time, snapshot in enumerate(train_dataset):
                # print('\nShapes:')
                # print(f'x: {snapshot.x.shape}')
                # print(f'edge_index: {snapshot.edge_index.shape}')
                # print(f'edge_attr: {snapshot.edge_attr.shape}')
                # print(f'y: {snapshot.y.shape}')
                # print(f'y_indices: {snapshot.y_indices.shape}')
                # print('\n')
                snapshot = snapshot.to(self.device)
                x = snapshot.x
                edge_index = snapshot.edge_index
                edge_attr = snapshot.edge_attr
                y= snapshot.y
            

                y_hat = self.model(x, edge_index, edge_attr).view(-1)
                y = y.float().view(-1)

                loss = self.criterion(y_hat, y)
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

 
            avg_loss = total_loss / len(train_dataset)
           
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        if output_dir:
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)

            config = {
                "node_features": self.node_features,
                "size": self.size,
                "epochs": self.epochs,
                "split_ratio": self.split_ratio,
                "loss_func": self.criterion,
                "train_loss": avg_loss,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
            }
            torch.save(self.model.state_dict(), f"{output_dir}.pt")
            with open(f"{output_dir}.yaml", "w") as f:
                yaml.dump(config, f)

    


class TGCNInference:
    def __init__(
            self,
            pretrained_model_path: str,
            node_features: int = 771,
            size: int = 128
        ):
        

        self.model_path = pretrained_model_path
        if not os.path.exists(pretrained_model_path):
            raise ValueError(
                f'The path {pretrained_model_path} does not exist'
            )
        
        self.node_features = node_features
        self.size = size
        

        self.model = None
        self.vocab = False
        self._tgcn_preparation()

    def _tgcn_preparation(self) -> None:
        self.model = TemporalGCN(self.node_features, self.size)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        self.vocab = True


    def predict(self, graph: TemporalGraph, return_binary=False, threshold: float = 0.5) -> list:
        if not self.vocab:
            raise ValueError(
                'The model is not loaded'
            )
        
        dataset = DynamicGraphTemporalSignal(
            graph.edge_indices, graph.edge_features, graph.xs, graph.ys, y_indices= graph.y_indices
        )

        device = next(self.model.parameters()).device
        self.model.to(device)

        ys, y_hats = None, None
    
        with torch.no_grad():
            for time, snapshot in enumerate(dataset):
                last_snapshot = snapshot
            
            last_snapshot = last_snapshot.to(device)
            x, edge_index, edge_attr, y = last_snapshot.x, last_snapshot.edge_index, last_snapshot.edge_attr, last_snapshot.y

            y_hat = self.model(x, edge_index, edge_attr, return_binary, threshold).squeeze()
            
        
            ys = y.cpu()
            y_hats = y_hat.cpu()

        return y_hats, ys

    def evaluate(self, graph: TemporalGraph, on_binary: False, threshold: float = 0.5) -> dict:
        y_hats, ys = self.predict(graph, return_binary=on_binary, threshold=threshold)

        metrics = self.evaluate_classification(ys, y_hats, on_binary=on_binary)
        return metrics
    
    def evaluate_classification(
            self, 
            y_true: torch.Tensor, 
            y_pred: torch.Tensor,
            on_binary: bool = False,
        ) -> dict:

        y_true = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
        y_pred = y_pred.numpy() if isinstance(y_pred, torch.Tensor) else y_pred

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)
        }

        if not on_binary:
            metrics["auc_roc"] = roc_auc_score(y_true, y_pred)

        return metrics





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
    

    

    
