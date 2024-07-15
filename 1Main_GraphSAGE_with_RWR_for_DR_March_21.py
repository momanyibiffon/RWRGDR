#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# reading datasets
A = pd.read_csv('dataset_processed/CDatasets/drug_disease_adjacency_matrix.csv', index_col=0) # adjacency matrix
DS = pd.read_csv('dataset_processed//Cdatasets/disease_similarity_matrix.csv', index_col=0) # disease similarity matrix
RS = pd.read_csv('dataset_processed/CDatasets/drug_similarity_matrix.csv', index_col=0) # drug similarity matrix
association_df = pd.read_csv('dataset_processed/CDatasets/D_R_association_df.csv', index_col=None)


# In[3]:


# data shapes
DS.shape, RS.shape, A.shape, association_df.shape


# In[4]:


# DS visualizations
DS


# In[5]:


# RS visualizations
RS


# In[6]:


# plotting lncrna and disease similarity figures

fig, (fig1, fig2) = plt.subplots(1, 2, figsize=(13,13))
fig1.title.set_text('Drugs similarity')
fig2.title.set_text('Diseases similarity')

# Adjust vmin and vmax to control the color intensity
fig1.imshow(RS, cmap="Blues", interpolation="none", vmin=0.1, vmax=0.9)
fig2.imshow(DS, cmap="Blues", interpolation="none", vmin=0.1, vmax=0.9)
plt.savefig('drug_disease_similarities.png', dpi=500) # Save the figure with high dpi


# In[7]:


# Adjacency matrix visualization
A


# In[8]:


# lncRNA-disease association visualization
association_df


# In[9]:


# creating a dataframe of drugs and diseases using index
drugs = []
diseases = []

for i in RS.index:
    drugs.append(i)
    
for i in DS.index:
    diseases.append(i)

# converting diseases and lncrnas lists to dataframe with a unique index (0 to n-1)
drugs_df = pd.DataFrame(drugs, index=range(len(drugs)), columns=['drugs'])
diseases_df = pd.DataFrame(diseases, index=range(len(diseases)), columns=['disease'])
len(drugs), len(diseases)


# In[10]:


drugs_df


# In[11]:


diseases_df


# ### GAT model

# In[12]:


import torch


# In[13]:


# mapping a unique disease ID to the disease ID
unique_disease_id = association_df['disease'].unique()
unique_disease_id = pd.DataFrame(data={
    'disease': unique_disease_id,
    'mappedID': pd.RangeIndex(len(unique_disease_id)),
})
print("Mapping of disease IDs to consecutive values:")
print("*********************************************")
print(unique_disease_id.head())

# mapping a unique lncrna ID to the lncrna ID
unique_drug_id = association_df['drug'].unique()
unique_drug_id = pd.DataFrame(data={
    'drug': unique_drug_id,
    'mappedID': pd.RangeIndex(len(unique_drug_id)),
})
print("Mapping of drug IDs to consecutive values:")
print("*********************************************")
print(unique_drug_id.head())

# Perform merge to obtain the edges from lncrna and diseases:
association_disease_id = pd.merge(association_df["disease"], unique_disease_id,
                            left_on='disease', right_on='disease', how='left')
association_disease_id = torch.from_numpy(association_disease_id['mappedID'].values)


association_drug_id = pd.merge(association_df['drug'], unique_drug_id,
                            left_on='drug', right_on='drug', how='left')
association_drug_id = torch.from_numpy(association_drug_id['mappedID'].values)

# construct `edge_index` in COO format
edge_index_disease_to_drug = torch.stack([association_disease_id, association_drug_id], dim=0)
print("Final edge indices from diseases to drugs")
print("*********************************************")
print(edge_index_disease_to_drug)
print(edge_index_disease_to_drug.shape)


# In[14]:


# disease and lncrna features
disease_feat = torch.from_numpy(DS.values).to(torch.float) # total disease features
drug_feat = torch.from_numpy(RS.values).to(torch.float) # total drug features
disease_feat.size(), drug_feat.size()


# In[15]:


# HeteroData object initialization and passing necessary info
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

data = HeteroData()
# Saving node indices
data["disease"].node_id = torch.arange(len(unique_disease_id))
data["drug"].node_id = torch.arange(len(RS))
# Adding node features and edge indices
data["disease"].x = disease_feat
data["drug"].x = drug_feat

data["disease", "associates_with", "drug"].edge_index = edge_index_disease_to_drug
# Adding reverse edges(GNN used this to pass messages in both directions)
data = T.ToUndirected()(data)
print(data)


# In[16]:


# confirmation of data in the HeteroData object
data.edge_index_dict, data.x_dict, data.edge_types


# In[17]:


# import networkx as nx
# import torch_geometric
# import matplotlib.pyplot as plt

# # Extracting the necessary data
# edge_index = data["disease", "associates_with", "drug"].edge_index.numpy()
# disease_nodes = data["disease"].node_id.numpy()
# drug_nodes = data["drug"].node_id.numpy()

# G = torch_geometric.utils.to_networkx(data.to_homogeneous()) # Convert HeteroData object to a NetworkX graph
# pos = nx.kamada_kawai_layout(G) # Get positions of nodes using Kamada-Kawai layout

# # Create figure and subplots
# fig, axs = plt.subplots(1, 3, figsize=(18, 6))
# # Draw the original graph
# nx.draw(G, pos=pos, with_labels=False, node_color="#77AADD", edgecolors="#000000", linewidths=0.5, node_size=200, ax=axs[0])
# # Draw a subgraph for diseases, reusing the same node positions
# nx.draw(G.subgraph(disease_nodes), pos=pos, node_color="#FFA500", edgecolors="#000000", linewidths=0.5, node_size=300, ax=axs[0])
# axs[0].set_title("Heterogeneous Graph")
# axs[0].axis('off')

# # Degree rank plot
# degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
# axs[1].plot(degree_sequence, "b-", marker="o")
# axs[1].set_title("Degree Rank Plot")
# axs[1].set_ylabel("Degree")
# axs[1].set_xlabel("Rank")

# # Degree histogram
# axs[2].hist(degree_sequence, bins='auto', color='b')
# axs[2].set_title("Degree Histogram")
# axs[2].set_xlabel("Degree")
# axs[2].set_ylabel("# of Nodes")

# # Adjust layout
# plt.tight_layout()

# # Save or display the figure
# plt.savefig('HetGraph_Degree_Plot.png', dpi=300)
# plt.show()


# In[18]:


# G.number_of_edges(), G.number_of_nodes()


# In[19]:


# split associations into training, validation, and test splits
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.5,
    neg_sampling_ratio=0.5,
    add_negative_train_samples=False,
    is_undirected = True, # added
    edge_types=("disease", "associates_with", "drug"),
    rev_edge_types=("drug", "rev_associates_with", "disease"),
)
train_data, val_data, test_data = transform(data)
print(train_data)


# In[20]:


# creating a mini-batch loader for generating subgraphs used as input into our GNN
import torch_sparse
from torch_geometric.loader import LinkNeighborLoader

# Defining seed edges
edge_label_index = train_data["disease", "associates_with", "drug"].edge_label_index
edge_label = train_data["disease", "associates_with", "drug"].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20] * 50,
    neg_sampling_ratio=0.5,
    edge_label_index=(("disease", "associates_with", "drug"), edge_label_index),
    edge_label=edge_label,
    batch_size=64,
    shuffle=True,
)  
sampled_data = next(iter(train_loader)) # Inspecting a sample mini-batch

print("Sampled mini-batch:")
print("===================")
print(sampled_data)


# In[21]:


sampled_data['disease'].x


# In[22]:


sampled_data['drug'].x


# ### Heterogeneous network Model based on GraphSAGE and RWR

# In[23]:


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import SAGEConv, GATConv, to_hetero
# from torch import Tensor
# from torch.nn import Linear
# from torch.nn import Transformer

# import networkx as nx
# import numpy as np

# def random_walk_with_restart(data, restart_prob=0.1, max_steps=10):
#     # Create a NetworkX graph from the provided data
#     graph = nx.Graph()
#     # Add nodes
#     for node_type, node_data in data.items():
#         for node_id in range(node_data.num_nodes):
#             graph.add_node((node_type, node_id))
#     # Add edges
#     for edge_type, edge_index in data.edge_index_dict.items():
#         for src, dst in edge_index.t().tolist():
#             graph.add_edge((edge_type[0], src), (edge_type[2], dst))
#             graph.add_edge((edge_type[2], dst), (edge_type[0], src))

#     rwr_scores = {}
#     for node_type, node_data in data.items():
#         for node_id in range(node_data.num_nodes):
#             current_node = (node_type, node_id)
#             scores = np.zeros(len(graph.nodes()))

#             # Perform random walk with restart from the current node
#             for _ in range(max_steps):
#                 next_scores = np.zeros(len(graph.nodes()))
#                 for neighbor in graph.neighbors(current_node):
#                     next_scores[neighbor] += restart_prob * scores[current_node] / len(list(graph.neighbors(current_node)))
#                 next_scores += (1 - restart_prob) * scores
#                 if np.allclose(next_scores, scores):
#                     break
#                 scores = next_scores
#             rwr_scores[(node_type, node_id)] = scores

#     return rwr_scores


# class MLPClassifier(torch.nn.Module):
#     def __init__(self, mlp_hidden_channels, mlp_out_channels):
#         super(MLPClassifier, self).__init__()
#         self.fc1 = torch.nn.Linear(mlp_hidden_channels, mlp_hidden_channels)
#         self.fc2 = torch.nn.Linear(mlp_hidden_channels, mlp_out_channels)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# class GNN(torch.nn.Module):
#     def __init__(self, hidden_channels, out_channels=32, num_heads=24):
#         super(GNN, self).__init__()
#         self.attn1 = GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False)
#         self.attn2 = GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False)
#         self.attn3 = GATConv((-1, -1), out_channels, heads=num_heads, add_self_loops=False)
#         self.num_heads = num_heads
#         self.penalty_linear = nn.Linear(out_channels, 1)

#         # GraphSAGE layers
#         self.sage1 = SAGEConv(hidden_channels, hidden_channels)
#         self.sage2 = SAGEConv(hidden_channels, hidden_channels)
#         self.sage3 = SAGEConv(hidden_channels, out_channels)

#     def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
#         x1 = self.attn1(x, edge_index)
#         x1 = x1.relu()
#         x1 = x1.view(-1, self.num_heads, x1.shape[1] // self.num_heads)
#         x1 = x1.mean(dim=1)

#         x2 = self.attn2(x1, edge_index)
#         x2 = x2.relu()
#         x2 = x2.view(-1, self.num_heads, x2.shape[1] // self.num_heads)
#         x2 = x2.mean(dim=1)

#         x3 = self.attn3(x2, edge_index)
#         x3 = x3.relu()
#         x3 = x3.view(-1, self.num_heads, x3.shape[1] // self.num_heads)
#         x3 = x3.mean(dim=1)

#         penalty = self.penalty_linear(x3)
#         x3 = x3 * torch.exp(penalty)
        
#         # GraphSAGE message passing
#         x1 = F.relu(self.sage1(x, edge_index))
#         x2 = F.relu(self.sage2(x1, edge_index))
#         x3 = F.relu(self.sage3(x2, edge_index))

#         return x3

# class Classifier(torch.nn.Module):
#     def __init__(self, mlp_hidden_channels, mlp_out_channels):
#         super(Classifier, self).__init__()
#         self.mlp = MLPClassifier(mlp_hidden_channels, mlp_out_channels)

#     def forward(self, x_disease: Tensor, x_drug: Tensor, edge_label_index: Tensor) -> Tensor:
#         edge_feat_disease = x_disease[edge_label_index[0]]
#         edge_feat_drug = x_drug[edge_label_index[1]]
#         concat_edge_feats = torch.cat((edge_feat_disease, edge_feat_drug), dim=-1)
#         return self.mlp(concat_edge_feats)


# class Model(torch.nn.Module):
#     def __init__(self, hidden_channels=2048, num_graphs=3, mlp_hidden_channels=64, mlp_out_channels=1):
#         super(Model, self).__init__()
#         self.num_graphs = num_graphs
#         self.graphs = torch.nn.ModuleList()
#         for i in range(num_graphs):
#             self.graphs.append(GNN(hidden_channels))
#         self.disease_lin = torch.nn.Linear(409, hidden_channels)
#         self.drug_lin = torch.nn.Linear(663, hidden_channels)
#         self.disease_emb = torch.nn.Embedding(data["disease"].num_nodes, hidden_channels)
#         self.drug_emb = torch.nn.Embedding(data["drug"].num_nodes, hidden_channels)
#         self.gnn = GNN(hidden_channels)
#         self.gnn = to_hetero(self.gnn, metadata=data.metadata())
#         self.classifier = Classifier(mlp_hidden_channels, mlp_out_channels)

#     def forward(self, data: HeteroData) -> Tensor:
#         # Compute RWR scores
#         rwr_scores = random_walk_with_restart(data)  # Implement this function

#         # Concatenate RWR scores with original features
#         for key, value in rwr_scores.items():
#             data[key].x = torch.cat((data[key].x, torch.tensor(value).unsqueeze(1)), dim=1)

#         # Forward pass through the model
#         x_dict = {
#             "disease": self.disease_lin(data["disease"].x) + self.disease_emb(data["disease"].node_id),
#             "drug": self.drug_lin(data["drug"].x) + self.drug_emb(data["drug"].node_id),
#         }
#         x_dict = self.gnn(x_dict, data.edge_index_dict)
#         pred = self.classifier(
#             x_dict["disease"],
#             x_dict["drug"],
#             data["disease", "associates_with", "drug"].edge_label_index,
#         )
#         return pred

# # Instantiate the model
# model = Model(hidden_channels=2048, mlp_hidden_channels=64, mlp_out_channels=1)
# print(model)


# In[24]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, to_hetero
from torch import Tensor
from torch.nn import Linear
from torch.nn import Transformer

import networkx as nx
import numpy as np

def random_walk_with_restart(data, restart_prob=0.1, max_steps=10):
    # Create a NetworkX graph from the provided data
    graph = nx.Graph()
    # Add nodes
    for node_type, node_data in data.items():
        for node_id in range(node_data.num_nodes):
            graph.add_node((node_type, node_id))
    # Add edges
    for edge_type, edge_index in data.edge_index_dict.items():
        for src, dst in edge_index.t().tolist():
            graph.add_edge((edge_type[0], src), (edge_type[2], dst))
            graph.add_edge((edge_type[2], dst), (edge_type[0], src))

    rwr_scores = {}
    for node_type, node_data in data.items():
        for node_id in range(node_data.num_nodes):
            current_node = (node_type, node_id)
            scores = np.zeros(len(graph.nodes()))

            # Perform random walk with restart from the current node
            for _ in range(max_steps):
                next_scores = np.zeros(len(graph.nodes()))
                for neighbor in graph.neighbors(current_node):
                    next_scores[neighbor] += restart_prob * scores[current_node] / len(list(graph.neighbors(current_node)))
                next_scores += (1 - restart_prob) * scores
                if np.allclose(next_scores, scores):
                    break
                scores = next_scores
            rwr_scores[(node_type, node_id)] = scores

    return rwr_scores


class MLPClassifier(torch.nn.Module):
    def __init__(self, mlp_hidden_channels, mlp_out_channels):
        super(MLPClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(mlp_hidden_channels, mlp_hidden_channels)
        self.fc2 = torch.nn.Linear(mlp_hidden_channels, mlp_out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels=32, num_heads=24, dropout=0.05):  # Add dropout argument
        super(GNN, self).__init__()
        self.attn1 = GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False)
        self.attn2 = GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False)
        self.attn3 = GATConv((-1, -1), out_channels, heads=num_heads, add_self_loops=False)
        self.num_heads = num_heads
        self.penalty_linear = nn.Linear(out_channels, 1)

        # GraphSAGE layers
        self.sage1 = SAGEConv(hidden_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.sage3 = SAGEConv(hidden_channels, out_channels)

        # Dropout layers
        self.dropout = nn.Dropout(dropout)  # Add dropout layer

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x1 = self.attn1(x, edge_index)
        x1 = self.dropout(x1)  # Apply dropout
        x1 = x1.relu()
        x1 = x1.view(-1, self.num_heads, x1.shape[1] // self.num_heads)
        x1 = x1.mean(dim=1)

        x2 = self.attn2(x1, edge_index)
        x2 = self.dropout(x2)  # Apply dropout
        x2 = x2.relu()
        x2 = x2.view(-1, self.num_heads, x2.shape[1] // self.num_heads)
        x2 = x2.mean(dim=1)

        x3 = self.attn3(x2, edge_index)
        x3 = self.dropout(x3)  # Apply dropout
        x3 = x3.relu()
        x3 = x3.view(-1, self.num_heads, x3.shape[1] // self.num_heads)
        x3 = x3.mean(dim=1)

        penalty = self.penalty_linear(x3)
        x3 = x3 * torch.exp(penalty)
        
        # GraphSAGE message passing
        x1 = F.relu(self.sage1(x, edge_index))
        x2 = F.relu(self.sage2(x1, edge_index))
        x2 = self.dropout(x2)  # Apply dropout
        x3 = F.relu(self.sage3(x2, edge_index))
        x3 = self.dropout(x3)  # Apply dropout

        return x3

class Classifier(torch.nn.Module):
    def __init__(self, mlp_hidden_channels, mlp_out_channels):
        super(Classifier, self).__init__()
        self.mlp = MLPClassifier(mlp_hidden_channels, mlp_out_channels)

    def forward(self, x_disease: Tensor, x_drug: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_disease = x_disease[edge_label_index[0]]
        edge_feat_drug = x_drug[edge_label_index[1]]
        concat_edge_feats = torch.cat((edge_feat_disease, edge_feat_drug), dim=-1)
        return self.mlp(concat_edge_feats)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels=512, num_graphs=3, mlp_hidden_channels=64, mlp_out_channels=1):
        super(Model, self).__init__()
        self.num_graphs = num_graphs
        self.graphs = torch.nn.ModuleList()
        for i in range(num_graphs):
            self.graphs.append(GNN(hidden_channels))
        self.disease_lin = torch.nn.Linear(409, hidden_channels)
        self.drug_lin = torch.nn.Linear(663, hidden_channels)
        self.disease_emb = torch.nn.Embedding(data["disease"].num_nodes, hidden_channels)
        self.drug_emb = torch.nn.Embedding(data["drug"].num_nodes, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier(mlp_hidden_channels, mlp_out_channels)

    def forward(self, data: HeteroData) -> Tensor:
        # Compute RWR scores
        rwr_scores = random_walk_with_restart(data)  # Implement this function

        # Concatenate RWR scores with original features
        for key, value in rwr_scores.items():
            data[key].x = torch.cat((data[key].x, torch.tensor(value).unsqueeze(1)), dim=1)

        # Forward pass through the model
        x_dict = {
            "disease": self.disease_lin(data["disease"].x) + self.disease_emb(data["disease"].node_id),
            "drug": self.drug_lin(data["drug"].x) + self.drug_emb(data["drug"].node_id),
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["disease"],
            x_dict["drug"],
            data["disease", "associates_with", "drug"].edge_label_index,
        )
        return pred

# Instantiate the model
model = Model(hidden_channels=512, mlp_hidden_channels=64, mlp_out_channels=1)
print(model)


# In[25]:


train_loader.data['disease', 'associates_with', 'drug'].edge_label.size()


# In[26]:


# validation loader
edge_label_index = val_data["disease", "associates_with", "drug"].edge_label_index
edge_label = val_data["disease", "associates_with", "drug"].edge_label

val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20] * 10,
    edge_label_index=(("disease", "associates_with", "drug"), edge_label_index),
    edge_label=edge_label,
    batch_size=64,
    shuffle=True,
)
sampled_data = next(iter(val_loader))
print(sampled_data)


# In[27]:


import torch_geometric
import networkx as nx

# evaluate the GNN model
# The new LinkNeighborLoader iterates over edges in the validation set
# obtaining predictions on validation edges
# then evaluate the performance by computing the AUC

# Define the validation seed edges:
edge_label_index = test_data["disease", "associates_with", "drug"].edge_label_index
edge_label = test_data["disease", "associates_with", "drug"].edge_label

test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=[20] * 10,
    edge_label_index=(("disease", "associates_with", "drug"), edge_label_index),
    edge_label=edge_label,
    batch_size=64,
    shuffle=True,
)
sampled_data = next(iter(test_loader))
print(sampled_data)


# In[35]:


import torch
import tqdm
import torch.nn.functional as F

# best parameters so far: lr: 0.0001, epochs 500/1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: {}".format(device))

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)  # Adding L2 regularization (weight decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

num_epochs = 500
total_batches = len(train_loader)  # Total number of batches in the training data

fold_loss_values = []  # List to store loss values for each fold
learning_curve = []  # List to store performance metric values during training
learning_rates = []  # List to store learning rates
true_labels = []  # List to store true labels
predicted_probs = []  # List to store predicted probabilities
validation_loss_values = []  # List to store validation loss values

# Create a single progress bar for all epochs
training_bar = tqdm.tqdm(total=num_epochs * total_batches, position=0, leave=True)

for epoch in range(1, num_epochs + 1):
    total_loss = total_examples = 0
    model.train()
    current_lr = optimizer.param_groups[0]['lr']

    # Training loop
    for batch_idx, sampled_data in enumerate(train_loader):
        optimizer.zero_grad()
        sampled_data = sampled_data.to(device)

        pred = model(sampled_data)
        ground_truth = sampled_data["disease", "associates_with", "drug"].edge_label
        pred_prob = torch.sigmoid(pred)
        loss = F.binary_cross_entropy(pred_prob.squeeze(), ground_truth)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * pred_prob.numel()
        total_examples += pred_prob.numel()

        # Update true_labels and predicted_probs
        true_labels.extend(ground_truth.tolist())
        predicted_probs.extend(pred_prob.squeeze().tolist())

        # Update progress bar
        training_bar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss / total_examples:.4f}")
        training_bar.update(1)  # Update progress bar

    fold_loss_values.append(total_loss / total_examples)  # Update fold_loss_values

    # Evaluate model performance on validation set
    model.eval()
    with torch.no_grad():
        total_val_loss = total_val_examples = 0
        for sampled_data in val_loader:
            sampled_data = sampled_data.to(device)
            pred = model(sampled_data)
            ground_truth = sampled_data["disease", "associates_with", "drug"].edge_label
            pred_prob = torch.sigmoid(pred)
            loss = F.binary_cross_entropy(pred_prob.squeeze(), ground_truth)

            total_val_loss += loss.item() * pred_prob.numel()
            total_val_examples += pred_prob.numel()

        val_loss = total_val_loss / total_val_examples
        learning_curve.append(1.0 - (total_loss / total_examples))  # Update learning_curve
        validation_loss_values.append(val_loss)  # Update validation_loss_values

    learning_rates.append(current_lr)  # Update learning_rates

    scheduler.step()

training_bar.close()  # Close progress bar

torch.save(model.state_dict(), 'trained_model.pth')


# In[36]:


# import torch
# import tqdm
# import torch.nn.functional as F

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Device: {}".format(device))

# model = model.to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)  # Adding L2 regularization (weight decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# num_epochs = 100
# total_batches = len(train_loader)  # Total number of batches in the training data

# fold_loss_values = []  # List to store loss values for each fold
# learning_curve = []  # List to store performance metric values during training
# learning_rates = []  # List to store learning rates
# true_labels = []  # List to store true labels
# predicted_probs = []  # List to store predicted probabilities

# # Create a single progress bar for all epochs
# training_bar = tqdm.tqdm(total=num_epochs * total_batches, position=0, leave=True)

# for epoch in range(1, num_epochs + 1):
#     total_loss = total_examples = 0
#     model.train()
#     current_lr = optimizer.param_groups[0]['lr']

#     # Training loop
#     for batch_idx, sampled_data in enumerate(train_loader):
#         optimizer.zero_grad()
#         sampled_data = sampled_data.to(device)

#         pred = model(sampled_data)
#         ground_truth = sampled_data["disease", "associates_with", "drug"].edge_label
#         pred_prob = torch.sigmoid(pred)
#         loss = F.binary_cross_entropy(pred_prob.squeeze(), ground_truth)

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item() * pred_prob.numel()
#         total_examples += pred_prob.numel()

#         # Update true_labels and predicted_probs
#         true_labels.extend(ground_truth.tolist())
#         predicted_probs.extend(pred_prob.squeeze().tolist())

#         # Update progress bar
#         training_bar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss / total_examples:.4f}")
#         training_bar.update(1)  # Update progress bar

#     fold_loss_values.append(total_loss / total_examples)  # Update fold_loss_values

#     learning_curve.append(1.0 - (total_loss / total_examples))  # Update learning_curve
#     learning_rates.append(current_lr)  # Update learning_rates

#     scheduler.step()

# training_bar.close()  # Close progress bar

# torch.save(model.state_dict(), 'trained_model.pth')


# In[37]:


from sklearn.calibration import calibration_curve

# Training Loss and Validation Loss Curves
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.plot(range(1, num_epochs + 1), fold_loss_values, label='Training Loss')
plt.plot(range(1, num_epochs + 1), validation_loss_values, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor('white')
plt.gca().set_frame_on(True)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')

# Learning Rate Schedule
plt.subplot(1, 3, 2)
plt.plot(range(1, num_epochs + 1), learning_rates)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor('white')
plt.gca().set_frame_on(True)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')

# Calibration Curve
plt.subplot(1, 3, 3)
fraction_of_positives, mean_predicted_value = calibration_curve(true_labels, predicted_probs, n_bins=10)
plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label='Calibration Curve')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve')
plt.legend()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor('white')
plt.gca().set_frame_on(True)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')

plt.tight_layout()

# Save the figure
plt.savefig('training_validation_calibration_curves_with_border.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


# In[38]:


# testing accuracy
from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay, roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, f1_score
from scipy.interpolate import interp1d
import seaborn as sns

# this function converts predictions from continous to binary specifically for 
# use in the classification report which doesn't accept continuous labels
def binary_predictions(threshold, x):
    predictions_binary = (x > threshold).astype(int)
    return predictions_binary
    
# main model for training and testing
def test_val_accuracy(loader):
    tv_preds = []
    tv_ground_truths = []
    for sampled_data in tqdm.tqdm(loader):
        with torch.no_grad():
            sampled_data.to(device)
            tv_preds.append(model(sampled_data))
            tv_ground_truths.append(sampled_data["disease", "associates_with", "drug"].edge_label)
    tv_preds = torch.cat(tv_preds, dim=0).cpu().numpy()
    tv_ground_truths = torch.cat(tv_ground_truths, dim=0).cpu().numpy()
    # tv_auc = roc_auc_score(tv_ground_truths, tv_preds)
    
    # plotting AUC Curve
    binary_ground_truths = np.array([1 if label == 2.0 or label == 1.0 else 0 for label in tv_ground_truths]) # converting ground truth values to {0, 1}
    
    # plotting the AUC using seaborn
    sns.set_style('white')
    sfpr, stpr, _ = roc_curve(binary_ground_truths, tv_preds)
    roc_auc = round(auc(sfpr, stpr), 2)
    sns.lineplot(x=sfpr, y=stpr, label=f'RWRGDR (AUC = {roc_auc})', errorbar=('ci', 99))
    sns.lineplot(x=[0,1], y=[0,1], color='black', linestyle='dashed')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC')
    plt.legend(loc='lower right')
    plt.savefig('AUC.png', dpi=500)
    plt.show()
    
    # Calculate AUPRC
    auprc = average_precision_score(binary_ground_truths, tv_preds)
    
    # Plotting AUPRC Curve
    sns.set_style('white')
    precision, recall, _ = precision_recall_curve(binary_ground_truths, tv_preds)
    auprc = round(auprc, 2)
    sns.lineplot(x=recall, y=precision, label=f'RWRGDR (AUPRC = {auprc})', errorbar=('ci', 99))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig('AUPRC.png', dpi=500)
    plt.show()
    
    # converting predictions to binary so as to print a classification report
    binary_preds = binary_predictions(0.5, tv_preds)
    # classification report
    clf_report = classification_report(binary_ground_truths, binary_preds)
    print("Classification report")
    print(clf_report)
#     print(binary_preds)
#     print(binary_ground_truths)
#     print(tv_preds)

    # plotting confusion matrix
    cm = confusion_matrix(binary_ground_truths, binary_preds)    
    class_labels = ["Negative", "Positive"] # Define class labels for the confusion matrix
    sns.set(font_scale=1.2)  # Adjusting font size
    plt.figure(figsize=(6, 4))  # Adjusting figure size
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig('onfusion_matrix.png', dpi=500)
    plt.show()
    
    # f1-score
    f1 = f1_score(binary_ground_truths, binary_preds)
    print("F1-score:", f1)
    
#     print(tv_preds)
#     print(tv_ground_truths)
    
    return roc_auc


# In[41]:


# Evaluate the final model on evaluation data
# print("Validation performance")
# evaluation_roc_auc = train_val_accuracy(final_val_loader)

# Evaluate the final model on test data
print("Testing performance")
test_roc_auc = test_val_accuracy(train_loader)
test_roc_auc = test_val_accuracy(test_loader)


# In[ ]:





# In[ ]:




