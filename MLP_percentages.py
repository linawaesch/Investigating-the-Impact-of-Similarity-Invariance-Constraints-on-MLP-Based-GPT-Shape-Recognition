import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np

import random
import numpy as np
import torch
from openpyxl import load_workbook


def set_seed(seed=41):
    random.seed(seed)                        # Python RNG
    np.random.seed(seed)                     # NumPy RNG
    torch.manual_seed(seed)                  # CPU RNG
    torch.cuda.manual_seed_all(seed)         # All CUDA devices
    # make cuDNN deterministic (may slow you down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =======================
# Editable Hyperparameters
# =======================
ORDER=1
INPUT_SIZE = 24*ORDER*ORDER          # 3 matrices × 4×4 entries × (real+imag) = 96
HIDDEN_SIZES = [20, 32]
OUTPUT_SIZE = 6*ORDER      # number of invariants columns in mean_invariants_by_shape.csv
LAMBDA = 0.3           # weight of the consistency penalty
LEARNING_RATE =0.5e-3
BATCH_SIZE = 5
EPOCHS = 2000
# Use Smooth L1 (Huber) loss instead of plain MSE
#LOSS_FN = nn.L1Loss()
LOSS_FN = nn.SmoothL1Loss()
EPSILON = 0.00
FILE1 = 'mean_invariants_by_shape.csv'
FILE2 = 'data-generation.csv'

# =======================
# Model Definition
# =======================

import os
import csv

def append_to_csv_line(csv_path, row_num, value):
    """
    Appends `value` to the end of line `row_num` (1-based) in a CSV.
    If the file doesn’t exist, it’s created. If row_num>1, earlier rows
    are created as empty.
    """
    rows = []
    if os.path.exists(csv_path):
        # Read existing rows
        with open(csv_path, newline='') as f:
            rows = list(csv.reader(f))
    # Ensure we have at least row_num rows
    while len(rows) < row_num:
        rows.append([])
    # Append the value to the desired row
    rows[row_num-1].append(str(value))
    # Write back (creates file if missing)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            # use LeakyReLU so that units never completely die
            layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# =======================
# Training & Evaluation
# =======================
def compute_full_loss(y1, y2, y_true):
    """
    Computes combined supervised + consistency loss.
    """
    loss1 = LOSS_FN(y1, y_true)
    loss2 = LOSS_FN(y2, y_true)
    diff = torch.abs(y1 - y2)
    insensitive = torch.clamp(diff - EPSILON, min=0.0)
    loss_c = torch.mean(insensitive**2)
    return loss1 + loss2 + LAMBDA * loss_c


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for X1, X2, y in loader:
        X1, X2, y = X1.to(device), X2.to(device), y.to(device)
        optimizer.zero_grad()
        y1 = model(X1)
        y2 = model(X2)
        loss = compute_full_loss(y1, y2, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X1.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X1, X2, y in loader:
            X1, X2, y = X1.to(device), X2.to(device), y.to(device)
            y1 = model(X1)
            y2 = model(X2)
            loss = compute_full_loss(y1, y2, y)
            total_loss += loss.item() * X1.size(0)
    return total_loss / len(loader.dataset)

import csv
import numpy as np

# =======================
# Utility: Parse CGPT entries into a complex matrix
# =======================
def parse_cgpt_entries(entry_str, size=2*ORDER):
    """
    Parse a whitespace-separated list of real and imaginary parts (with trailing 'i')
    into a (size x size) numpy complex matrix.
    """
    tokens = entry_str.strip().split()
    vals = []
    for i in range(0, len(tokens), 2):
        real = float(tokens[i])
        imag_str = tokens[i+1]
        # remove trailing 'i' if present
        if imag_str.endswith('i'):
            imag = float(imag_str[:-1])
        else:
            imag = float(imag_str)
        vals.append(real + 1j * imag)
    return np.array(vals).reshape((size, size))


import numpy as np

def classify_and_check(pred_invariants: np.ndarray,
                       true_shape: str,
                       mean_map: dict) -> bool:
    """
    Determine which shape’s mean invariants are closest to the predicted invariants,
    then check whether that predicted shape matches the true shape.

    Parameters
    ----------
    pred_invariants : np.ndarray, shape (n_invariants,)
        The predicted invariants vector.
    true_shape : str
        The ground‐truth shape label.
    mean_map : dict[str, np.ndarray]
        Mapping from shape name to its mean invariants vector.

    Returns
    -------
    bool
        True if the closest‐mean shape equals true_shape, else False.
    """
    # Compute Euclidean distance to each shape’s mean invariants
    distances = {
        shape: np.linalg.norm(pred_invariants - mean_vec)
        for shape, mean_vec in mean_map.items()
    }
    # Find the shape with minimal distance
    predicted_shape = min(distances, key=distances.get)

    # Check correctness
    is_correct = (predicted_shape == true_shape)

    # Print result
    #print(f"True: {true_shape!r}, Predicted: {predicted_shape!r} → Correct? {is_correct}")

    return is_correct


# =======================
# Main Script: Simple CSV Loop to Build X1, X2, and y
# =======================
def main():
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Data Loading (temp.py style) ---
    X1_samples, X2_samples, y_samples = [], [], []
    buffer, group_count = [], 0
    # load invariants map
    invariants_map = {}
    with open(FILE1, newline='') as invfile:
        for row in csv.DictReader(invfile):
            key = row['shape']
            invariants_map[key] = np.array([float(row[c]) for c in row if c!='shape'], dtype=np.float32)
    # stream through data-generation.csv
    with open(FILE2, newline='') as datafile:
        for row in csv.DictReader(datafile):
            if row['type']=='theoretical':
                continue
            buffer.append(row)
            if len(buffer)==3:
                # flatten three matrices
                vec = []
                for ent in buffer:
                    M = parse_cgpt_entries(ent['CGPT_entries'], size=2*ORDER)
                    vec.extend(M.real.ravel())
                    vec.extend(M.imag.ravel())
                arr = np.array(vec, dtype=np.float32)
                if group_count%2==0:
                    X1_samples.append(arr)
                else:
                    X2_samples.append(arr)
                    y_samples.append(invariants_map[buffer[0]['shape']])
                group_count+=1
                buffer.clear()
    # 1️⃣ Flatten into single NumPy arrays
    X1_np = np.stack(X1_samples, axis=0)   # shape (N,96)
    X2_np = np.stack(X2_samples, axis=0)   # shape (N,96)
    y_np  = np.stack(y_samples,  axis=0)   # shape (N,12)

    # ——————————————
    # 2️⃣ Compute mean/std on all X’s (so both views share the same scaler)
    X_all = np.vstack([X1_np, X2_np])
    mean = X_all.mean(axis=0, keepdims=True)
    std  = X_all.std(axis=0, keepdims=True) + 1e-8


    #y_all = np.vstack([y_np, y_np])
    #mean_y = y_all.mean(axis=0, keepdims=True)
    #std_y  = y_all.std(axis=0, keepdims=True) + 1e-8

    # 3️⃣ Normalize each
    X1_np = (X1_np - mean) / std
    X2_np = (X2_np - mean) / std
    #y_np = (y_np - mean_y) / std_y


    # 4️⃣ Turn into fast torch Tensors
    X1 = torch.from_numpy(X1_np).float()
    X2 = torch.from_numpy(X2_np).float()
    y  = torch.from_numpy(y_np).float()


    # move back to CPU & convert to NumPy
    y_np = y.cpu().numpy()

    # turn into a DataFrame and write out
    dataset = TensorDataset(X1, X2, y)
    n_train = int(0.8*len(dataset))
    train_ds, test_ds = random_split(dataset, [n_train, len(dataset)-n_train])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # --- Model & Training ---
    model = MLP(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(1, EPOCHS+1):
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        te_loss = evaluate(model, test_loader, device)
        #print(f"Epoch {epoch}/{EPOCHS}  Train: {tr_loss:.9f}  Test: {te_loss:.9f}")
        # === Predictions on Test Set (formatted with error) ===
        preds_list, trues_list = [], []
        model.eval()
        with torch.no_grad():
            for X1b, X2b, yb in test_loader:
                X1b = X1b.to(device)
                out = model(X1b).cpu().numpy()
                preds_list.append(out)
                trues_list.append(yb.numpy())
        preds_arr = np.vstack(preds_list)
        trues_arr = np.vstack(trues_list)

        # === Model predictions on the two test‐set views ===preds_list, trues_list = [], []
        model.eval()
        with torch.no_grad():
            for X1b, X2b, yb in test_loader:
                # Move inputs to device and predict on the first view
                X1b = X1b.to(device)
                out = model(X1b).cpu().numpy()
                preds_list.append(out)
                trues_list.append(yb.numpy())

        # Stack into arrays of shape (N_samples, OUTPUT_SIZE)
        preds_arr = np.vstack(preds_list)
        trues_arr = np.vstack(trues_list)

        # Compute element‐wise errors
        errors = preds_arr - trues_arr        # shape (N, OUTPUT_SIZE)
        sq_errors = errors ** 2               # squared errors

        # Mean over all samples and outputs
        mse = np.mean(sq_errors)
        rmse = np.sqrt(mse)

        
        #print (f"Overall RMSE on test set: {rmse:.4f}")

        # === Classification on the test set ===# === Shape Classification on Test Set ===

        # === Shape Classification on Test Set ===

        # build the mean_map
        mean_map = {}
        with open(FILE1, newline='') as mf:
            reader = csv.DictReader(mf)
            inv_cols = [c for c in reader.fieldnames if c != 'shape']
            for row in reader:
                shape = row['shape']
                vec = np.array([float(row[c]) for c in inv_cols], dtype=np.float32)
                mean_map[shape] = vec

        model.eval()

        correct = 0
        total = 0

        for X1b, X2b, yb in test_loader:
            # move to device once
            X1b = X1b.to(device)
            X2b = X2b.to(device)

            # predict both views as batches
            pred1 = model(X1b).detach().cpu().numpy()  # shape (batch, 12)
            pred2 = model(X2b).detach().cpu().numpy()

            # true invariants batch
            true_batch = yb.numpy()  # shape (batch, 12)

            # for each sample in this batch
            for i in range(pred1.shape[0]):
                # recover true shape by matching the mean_map
                yb_vec = true_batch[i]
                true_shape = next(
                    (s for s, m in mean_map.items() if np.allclose(yb_vec, m)),
                    None
                )

                # first view
                pred_inv = pred1[i]
                if classify_and_check(pred_inv, true_shape, mean_map):
                    correct += 1
                total += 1

                # second view
                pred_inv = pred2[i]
                if classify_and_check(pred_inv, true_shape, mean_map):
                    correct += 1
                total += 1

        print(f"\nOverall: {correct} / {total} correct ({correct/total*100:.1f}%)")
        append_to_csv_line('new2.csv', epoch, correct/total*100)



    # save model
    torch.save(model.state_dict(), "mlp_with_data_merge.pth")
    print("Model saved.")


    return

if __name__=='__main__':
    set_seed(2)
    main()