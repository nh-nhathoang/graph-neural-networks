import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score
import numpy as np

def visualize_performance(true_E, pred_E, R2, cover_interval, overlap, save_dir):
    plt.rcParams['font.size'] = 12
    # Plotting for E
    fig, ax = plt.subplots(figsize=(5,5), dpi=400)   

    ax.plot(true_E, pred_E, 'bo', markersize=3, label=f'$R^2 = %.3f$' %R2)
    ax.axline((0, 0), slope=1, color='red', linestyle='--')
    ax.set_ylabel(r'Predicted $E$ (GPa)')
    ax.set_xlabel(r'Ground Truth $E$ (GPa)')


    ax.set_xlim([min(true_E), max(true_E)])
    ax.set_ylim([min(true_E), max(true_E)])
    ax.legend()

    plt.tight_layout()
    
# Evaluate model performance on the training, validation, and test sets
def evaluate_model(model, loader, device, cover_interval, overlap, save_dir, split_name = "test"):
    model.eval()

    E_pred_whole = []
    E_whole = []

    with torch.no_grad():
        for data in loader:
            E_pred = model(data.x.to(device), data.edge_index.long().to(device), data.batch.to(device))
            data.E = data.E.to(torch.float32)

            E_pred_whole.append(E_pred.cpu().numpy())
            E_whole.append(data.E.cpu().numpy())

    E_pred_whole = np.concatenate(E_pred_whole, axis=0)
    E_whole = np.concatenate(E_whole, axis=0)

    R2 = r2_score(E_whole, E_pred_whole)

    print(f'{split_name.capitalize()} R2 for E: {R2}')

    # Return values if needed later
    visualize_performance(E_whole, E_pred_whole, R2, cover_interval, overlap, save_dir)
    plt.savefig(f"{save_dir}/{split_name}_prediction_vs_ground_truth.png")
    plt.close()
    return R2
