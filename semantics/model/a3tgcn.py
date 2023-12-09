from semantics.graphs.temporal_graph import TemporalGraph
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN
from typing import Optional
import tqdm




class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=32, 
                           periods=1)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h
 

class TemporalGNNTrainer:
    def __init__(
            self, 
            node_features: int = 3,
            epochs: int = 10,
            split_ratio: float = 0.8,
            learning_rate: float = 0.01,
            device: str = "cpu",
            subset: Optional[int] = None
            ):
        self.node_features = node_features
        self.epochs = epochs
        self.split_ratio = split_ratio
        self.learning_rate = learning_rate
        self.device = device
        self.subset = subset
    
    def train(
            self, 
            graph: TemporalGraph
            ) -> None:
        
        dataset = DynamicGraphTemporalSignal(
            graph.edge_indices, graph.edge_features, graph.xs, graph.ys, y_indices= graph.y_indices
        )
        print("Dataset type:  ", dataset)
        print(next(iter(dataset)))

        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=self.split_ratio)




        sample = self.subset if self.subset else sum(1 for _ in train_dataset)

        device = torch.device(self.device)
        model = TemporalGNN(node_features=self.node_features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        # self.criterion = torch.nn.MSELoss()

        model.train()

        progress_bar = tqdm.tqdm(
            range(sample * self.epochs), 
            desc="Training", 
            dynamic_ncols=True
            )
        
        for epoch in range(self.epochs):
            loss = 0
            step = 0
            for snapshot in train_dataset:
                snapshot = snapshot.to(device)
                # Get model predictions
                y_hat = model(snapshot.x, snapshot.edge_index)
                # Mean squared error
                loss = loss + torch.mean((y_hat-snapshot.y)**2) 
                step += 1
                if step > sample:
                    break

            loss = loss / (step + 1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))



        model.eval()
        loss = 0
        step = 0

        # Store for analysis
        predictions = []
        labels = []

        for snapshot in test_dataset:
            snapshot = snapshot.to(device)
            # Get predictions
            y_hat = model(snapshot.x, snapshot.edge_index)
            # Mean squared error
            loss = loss + torch.mean((y_hat-snapshot.y)**2)
            # Store for analysis below
            labels.append(snapshot.y)
            predictions.append(y_hat)
            step += 1
        

        loss = loss / (step+1)
        loss = loss.item()
        print("Test MSE: {:.4f}".format(loss))
