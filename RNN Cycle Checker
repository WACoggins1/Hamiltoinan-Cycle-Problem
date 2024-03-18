import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RNNHamiltonianNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNHamiltonianNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_dim)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

# Function to read edge list from a .txt file and construct the adjacency matrix
def read_edge_list(file_path):
    edges = []
    with open(file_path, 'r') as file:
        for line in file:
            edge = tuple(map(int, line.strip().split()))
            edges.append(edge)
    max_node = max(max(edge) for edge in edges)
    adj_matrix = np.zeros((max_node, max_node), dtype=np.float32)
    for edge in edges:
        adj_matrix[edge[0]-1, edge[1]-1] = 1
        adj_matrix[edge[1]-1, edge[0]-1] = 1
    return torch.tensor(adj_matrix)

# Read the edge list from a .txt file
edge_list_file = "II_7932.hcp.txt"  # Update with the path to your .txt file
adj_matrix = read_edge_list(edge_list_file)


# Create input data as sequences of node visits
# For simplicity, we'll create sequences of length equal to the number of nodes, representing a possible Hamiltonian cycle
# Each sequence starts from a different node
input_data = torch.eye(adj_matrix.shape[0]).unsqueeze(0)  # One-hot encoding of starting nodes

# Labels: 1 if the sequence represents a Hamiltonian cycle, 0 otherwise
labels = torch.tensor([[1]], dtype=torch.float)

# Instantiate the RNN model
input_dim = adj_matrix.shape[0]  # Dimension of one-hot encoded node representation
hidden_dim = 16
output_dim = 1  # Binary classification: Hamiltonian cycle present or not
model = RNNHamiltonianNet(input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(input_data)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()

# Evaluate the trained model
model.eval()
with torch.no_grad():
    pred = model(input_data)
    pred_binary = torch.round(pred)  # Round predictions to 0 or 1
    print("Predicted:", pred_binary.item(), "Actual:", labels.item())
