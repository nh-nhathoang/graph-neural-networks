import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import networkx as nx
from sklearn.preprocessing import StandardScaler
import numpy as np

    
def is_undirected(data):
    # Extract the edge index
    edge_index = data.edge_index
    
    # Create a set to store unique edges
    edge_set = set()
    for i in range(edge_index.shape[1]):
        # Get nodes from edge index
        node_a = edge_index[0, i].item()
        node_b = edge_index[1, i].item()

        # Check if the reverse edge exists
        if (node_b, node_a) not in edge_set:
            edge_set.add((node_a, node_b))
        else:
            edge_set.remove((node_b, node_a))
            
    # If all edges had their reverse, the edge_set should be empty
    return len(edge_set) == 0

def make_undirected(data):
    # Extract edge index
    edge_index = data.edge_index

    # Convert edge_index to a set of tuples for O(1) look-up time
    edge_set = {(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.shape[1])}

    # Identify missing reverse edges
    missing_edges = [(b, a) for a, b in edge_set if (b, a) not in edge_set]

    # Convert lists of missing edges to tensor format and add them to the original edge_index
    if len(missing_edges) > 0:
        missing_edges_tensor = torch.tensor(missing_edges, dtype=torch.long).t()
        data.edge_index = torch.cat([edge_index, missing_edges_tensor], dim=1)
    
    return data

def feature_engineering(data_list):
    engineered_data_list = []

    for data in data_list:
        new_data = data.clone()

        # center coordinates (x, y) 
        center_coords = new_data.x[:, :2]

        # distance to centroid
        centroid = torch.mean(center_coords, dim=0, keepdim=True)
        dist_to_center = torch.norm(center_coords - centroid, dim=1, keepdim=True)

        # graph construction
        G = nx.Graph()
        G.add_edges_from(new_data.edge_index.t().cpu().numpy())

        # node degree
        degrees = torch.tensor(
            [G.degree[i] for i in range(new_data.num_nodes)], dtype=torch.float
        ).view(-1, 1)

        # closeness centrality (may be replaced with eigenvector centrality)
        closeness = torch.tensor(
            list(nx.closeness_centrality(G).values()), dtype=torch.float
        ).view(-1, 1)

        # clustering
        clustering = torch.tensor(
            list(nx.clustering(G).values()), dtype=torch.float
        ).view(-1, 1)

        # add new features
        new_data.x = torch.cat([center_coords, dist_to_center, degrees, 
                                closeness, clustering], dim=1)

        engineered_data_list.append(new_data)

    return engineered_data_list

def normalize_planar_info(training_list, validation_list = None, test_list = None):
    all_train_x = torch.cat([data.x for data in training_list], dim=0)
    num_features = all_train_x.shape[1]
    
    global_mins = torch.min(all_train_x, dim=0)[0]
    global_maxs = torch.max(all_train_x, dim=0)[0]

    def scale_split(data_list):
        for data in data_list:
            for i in range(num_features):
                denom = global_maxs[i] - global_mins[i]
                if denom != 0:
                    data.x[:, i] = (data.x[:, i] - global_mins[i]) / denom
        return data_list
    
    training_list = scale_split(training_list)
    if validation_list is not None:
        validation_list = scale_split(validation_list)
    if test_list is not None:
        test_list = scale_split(test_list)

    return training_list, validation_list, test_list


def normalize_E(data_list):

    # Assume `data_list` is your list of graph data objects
    E_vals = np.array([data.E for data in data_list]).reshape(-1, 1)  # shape: (n, 1)

    # Fit scaler
    scaler = StandardScaler()
    E_scaled = scaler.fit_transform(E_vals)

    for i, data in enumerate(data_list):
        data.E = float(E_scaled[i])

    return data_list

def prepare_dataset(data_list, batch_size, train_percentage=0.80, valid_percentage=0.1):
    dataset_size = len(data_list)
    train_size = int(train_percentage * dataset_size)
    valid_size = int(valid_percentage * dataset_size)
    test_size = dataset_size - train_size - valid_size

    train_set, valid_set, test_set = random_split(data_list, [train_size, valid_size, test_size])
   
    print(f'Number of training graphs: {len(train_set)}')
    print(f'Number of validation graphs: {len(valid_set)}')
    print(f'Number of test graphs: {len(test_set)}')
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader

def check_duplicate_graphs(dataset):
    seen = {}
    duplicates = []

    for i, data in enumerate(dataset):
        x_key = tuple(map(tuple, torch.round(data.x, decimals=8).tolist()))

        edge_index = data.edge_index.t().tolist()
        edge_key = tuple(sorted(tuple(edge) for edge in edge_index))

        e_key = round(float(data.E), 8)

        graph_key = (x_key, edge_key, e_key)

        if graph_key in seen:
            duplicates.append((seen[graph_key], i))
        else:
            seen[graph_key] = i

    print(f"Number of duplicate graphs with same coordinates, edges, and E: {len(duplicates)}")

    if duplicates:
        print("Duplicate pairs:")
        for a, b in duplicates[:20]:
            print(f"Graph {a} and Graph {b}")

    return duplicates
