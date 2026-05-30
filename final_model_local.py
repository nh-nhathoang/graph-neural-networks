import argparse
import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool

from utils.data_processing import *
from utils.architecture import GIN, Transformer
from utils.train_model import train_model
from utils.evaluate_model import *


def load_inp_data(file_path, label):
    nodes, edges = [], []

    with open(file_path, "r") as file:
        lines = file.readlines()

    node_section = False
    element_section = False

    for line in lines:
        line = line.strip()

        if "*Node" in line:
            node_section = True
            element_section = False
            continue

        if "*Element" in line:
            node_section = False
            element_section = True
            continue

        if "*Nset" in line or "*Elset" in line:
            node_section = False
            element_section = False
            continue

        if node_section:
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    x = float(parts[1].strip())
                    y = float(parts[2].strip())
                    nodes.append([x, y])
                except ValueError:
                    continue

        if element_section:
            parts = line.split(",")
            if (
                len(parts) >= 3
                and parts[0].strip().isdigit()
                and parts[1].strip().isdigit()
                and parts[2].strip().isdigit()
            ):
                start = int(parts[1].strip()) - 1
                end = int(parts[2].strip()) - 1

                if start == end:
                    continue

                if start < len(nodes) and end < len(nodes):
                    edges.append([start, end])
                else:
                    print(f"Skipping invalid edge ({start + 1}, {end + 1}) in {file_path}")

    node_tensor = torch.tensor(nodes, dtype=torch.float)
    edge_tensor = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )

    return Data(x=node_tensor, edge_index=edge_tensor, E=np.float64(label * 200.00))


def load_dataset(data_dir, folder_list):
    dataset = []

    for folder in folder_list:
        results_path = os.path.join(data_dir, folder, "Results.txt")
        inp_folder = os.path.join(data_dir, folder, "inp_files")

        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Missing Results.txt: {results_path}")

        if not os.path.isdir(inp_folder):
            raise FileNotFoundError(f"Missing inp_files folder: {inp_folder}")

        file_labels = {}
        with open(results_path, "r") as file:
            for line in file:
                if line.strip() == "" or "E/Es" in line:
                    continue
                try:
                    file_name, label = line.strip().split(", ")
                    filename = file_name.strip().split(".")[0]
                    file_labels[filename] = float(label)
                except ValueError:
                    print(f"[{folder}] Skipping invalid line: {line.strip()}")

        for file, label in file_labels.items():
            file_path = os.path.join(inp_folder, f"{file}.inp")
            if os.path.exists(file_path):
                dataset.append(load_inp_data(file_path, label))
            else:
                print(f"[{folder}] Missing file: {file_path}")

    if len(dataset) == 0:
        raise RuntimeError("No graph data was loaded. Check --data_dir and folder names.")

    return dataset


def plot_dataset(dataset, output_dir):
    E_vals = [data.E for data in dataset]

    plt.figure(figsize=(5, 5), dpi=300)
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 13

    labels = ["Hexagonal", "Kagome", "Demi-A", "Demi-B", "Demi-C"]
    markers = ["o", "s", "^", "D", "x"]
    colors = ["blue", "green", "red", "orange", "purple"]

    size = 4951
    for i in range(5):
        start = i * size
        end = min((i + 1) * size, len(E_vals))
        if start >= len(E_vals):
            break

        x = range(start, end)
        y = E_vals[start:end]

        plt.plot( x, y, linestyle="none", marker=markers[i], color=colors[i], markersize=3, label=labels[i], alpha=0.4)

    plt.ylabel(r"Elastic Modulus $E$ (GPa)")
    plt.xlabel("Samples")
    plt.xlim(right=25000)
    plt.ylim(top=3.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "dataset_distribution.png"))
    plt.close()


def plot_losses(train_losses, valid_losses, epoch_num, output_dir):
    plt.figure(figsize=(5, 5), dpi=400)
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 12

    plt.plot(np.array(train_losses), label="Train")
    plt.plot(np.array(valid_losses), label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.xlim(0, epoch_num)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".", help="Folder containing Hexagonal, Kagome, Demi-A, Demi-B, and Demi-C")
    parser.add_argument("--batch_num", type=int, default=32, help="Batch size")
    parser.add_argument("--epoch_num", type=int, default=150, help="Epoch number")
    parser.add_argument("--cover_interval", type=int, default=20, help="Cover interval")
    parser.add_argument("--overlap", type=float, default=0.3, help="Cover overlap")
    parser.add_argument("--save_model_dir", type=str, default="./saved_model", help="Output directory")
    parser.add_argument("--model", type=str, default="transformer", choices=["transformer", "gin"], help="Model type")
    args = parser.parse_args()

    folder_list = ["Hexagonal", "Kagome", "Demi-A", "Demi-B", "Demi-C"]

    os.makedirs(args.save_model_dir, exist_ok=True)

    dataset = load_dataset(args.data_dir, folder_list)

    check_duplicate_graphs(dataset)

    batch_size = args.batch_num
    epoch_num = args.epoch_num
    learning_rate = 0.001
    overlap = args.overlap

    all_undirected_before = all(is_undirected(data) for data in dataset)
    print(f"Graphs are undirected (before conversion): {all_undirected_before}")

    dataset = [make_undirected(data) for data in dataset]

    all_undirected_after = all(is_undirected(data) for data in dataset)
    print(f"Graphs are undirected (after conversion): {all_undirected_after}")

    dataset = normalize_planar_info(dataset)
    train_loader, valid_loader, test_loader = prepare_dataset(
        dataset, batch_size, train_percentage=0.80, valid_percentage=0.1
    )

    plot_dataset(dataset, args.save_model_dir)

    no_node_feature = dataset[0].x.shape[1]

    if args.model == "transformer":
        model = Transformer(dim_h=64, node_feature=no_node_feature)
    else:
        model = GIN(dim_h=64, node_feature=no_node_feature)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-5)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model.to(device)

    print(model)
    print("Device:", device)

    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters:", num_params)

    train_losses, valid_losses, R2_trainings, R2_valids, best_state_dict = train_model(
        model, train_loader, valid_loader, criterion, 
        optimizer, scheduler, device=device, num_epochs=epoch_num,
    )

    plot_losses(train_losses, valid_losses, epoch_num, args.save_model_dir)

    best_model_path = os.path.join(args.save_model_dir, f"epoch_{args.epoch_num}.pt")
    torch.save(best_state_dict, best_model_path)
    print(f"Saved best model to: {best_model_path}")

    model.load_state_dict(best_state_dict)

    evaluate_model(model, train_loader, device, args.cover_interval, overlap, args.save_model_dir, split_name="train")
    evaluate_model(model, valid_loader, device, args.cover_interval, overlap, args.save_model_dir, split_name="validation")
    evaluate_model(model, test_loader, device, args.cover_interval, overlap, args.save_model_dir, split_name="test")


if __name__ == "__main__":
    main()
