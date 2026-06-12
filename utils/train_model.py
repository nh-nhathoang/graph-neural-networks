import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score
import copy

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, device='cuda', num_epochs=1):
    model.to(device)

    train_losses = []
    valid_losses = []
    R2_trainings = []
    R2_valids = []
    best_state_dict = None
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        total_graphs = 0
        pred_whole = []
        E_whole = []
        
        for data in train_loader:
            pred = model(data.x.to(device), data.edge_index.long().to(device), data.batch.to(device))
            data.E = data.E.to(torch.float32)
            
            optimizer.zero_grad()
            loss = criterion(pred, data.E.to(device))
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Gradient clipping
            optimizer.step()
            
            num_graphs_in_batch = data.num_graphs
            train_loss += loss.item() * num_graphs_in_batch
            total_graphs += num_graphs_in_batch
            
            pred_whole.append(pred.detach().cpu().numpy())
            E_whole.append(data.E.cpu().numpy())
        
        out_whole = np.concatenate(pred_whole, axis=0)
        E_whole = np.concatenate(E_whole, axis=0)  
        R2_train = r2_score(E_whole, out_whole)  
        R2_trainings.append(R2_train)  
        
        epoch_loss = train_loss / total_graphs
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            pred_whole = []
            E_whole = []
            valid_loss = 0
            total_graphs = 0
            
            for data in valid_loader:
                pred = model(data.x.to(device), data.edge_index.long().to(device), data.batch.to(device))
                data.E = data.E.to(torch.float32)
                
                num_graphs_in_batch = data.num_graphs
                valid_loss += criterion(pred, data.E.to(device)).item() * num_graphs_in_batch
                total_graphs += num_graphs_in_batch
                
                pred_whole.append(pred.cpu().numpy())
                E_whole.append(data.E.cpu().numpy())
            
            pred_whole = np.concatenate(pred_whole, axis=0)
            E_whole = np.concatenate(E_whole, axis=0)  
            R2_valid = r2_score(E_whole, pred_whole)
            R2_valids.append(R2_valid)
            
            valid_loss = valid_loss / total_graphs
            valid_losses.append(valid_loss)  
        
        scheduler.step(valid_loss)  # Step the scheduler based on validation loss
        current_lr = scheduler.get_last_lr()[0]

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state_dict = copy.deepcopy(model.state_dict())
        print(f'Epoch [{epoch+1}], LR [{current_lr}], Loss[Train: {epoch_loss:.3f}, Valid: {valid_loss:.3f}], R2[Train: {R2_train:.3f}, Valid: {R2_valid:.3f}]')
    
    print(f'Final Valid Loss: {valid_losses[-1]:.4f}')
    
    return train_losses, valid_losses, R2_trainings, R2_valids, best_state_dict
