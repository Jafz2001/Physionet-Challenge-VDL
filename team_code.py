#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import numpy as np
import os
import sys

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import csv

from helper_code import *
from features_extractor import *

import random
import pandas as pd

from scipy.signal import butter, filtfilt

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from model import ECGConv2D

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

def train_model(data_folder, model_folder, verbose):
    if verbose:
        print('Finding the Challenge data...')

    # Load all records from the data folder
    records = find_records(data_folder)
    records, num_records = select_records(data_folder, records)
    print('Number of samples:', num_records)
    
    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')
    
    signals = []
    labels = []

    # Iterate over the records.
    error = 0
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        signal = extract_signal(record)

        signals.append(signal)
        labels.append(load_label(record))
        
    X = np.stack(signals)
    y = np.stack(labels)

    if verbose:
        print("Signal error counter:", error)

    # ----------------------------
    # Entrenamiento del modelo
    # ----------------------------
    if verbose:
        print('Training the model on the data...')

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Dataset y DataLoader
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

    model = ECGConv2D(n_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(20):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
    print(f"\nEpoch {epoch+1} | Train loss: {train_loss:.4f}")

    best_f1 = -1.0
    best_epoch = -1
    os.makedirs(model_folder, exist_ok=True)

    # Training loop
    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)            # (B, 12, T)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)            # adapta tu forward a (B,1,12,T) si hace falta
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # --- Validación & métrica (F1) ---
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_true.append(yb.numpy())
                y_pred.append(pred)
        y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
        val_f1 = f1_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0

        if verbose:
            print(f"Epoch {epoch+1:02d} | Train loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")

        # Guardar mejor por F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            save_model(model_folder, model, optimizer, epoch=best_epoch, extra={"val_f1": best_f1})

    if verbose:
        print('Training complete.')


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filename = os.path.join(model_folder, "model.pt")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file not found: {filename}")

    model = ECGConv2D(n_classes=2).to(device)
    ckpt = torch.load(filename, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    if verbose:
        info = ckpt.get("extra", {})
        print(f"Loaded model from {filename} (epoch={ckpt.get('epoch')} | info={info})")
    return {"model": model, "device": device}

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    model_net = model["model"]
    device    = model["device"]

    # Extraer señal y preprocesar
    sig = extract_signal(record)  # (12, T)
    if sig is None or np.isnan(sig).any() or not np.isfinite(sig).all():
        return 0, 0.0

    x = torch.from_numpy(sig.astype(np.float32)).unsqueeze(0)  # (1, 12, T)
    x = x.to(device)

    with torch.no_grad():
        logits = model_net(x)               # (1, 2)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        cls    = int(np.argmax(probs))
        p1     = float(probs[1])            # prob clase positiva

    return cls, p1

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################
def bandpass_filter(ecg, fs=500, lowcut=0.5, highcut=40, order=4):
        b, a = butter(order, [lowcut / (fs/2), highcut / (fs/2)], btype='band')
        return filtfilt(b, a, ecg)

def normalize_channels(ecg):
    return (ecg - ecg.mean(axis=1, keepdims=True)) / (ecg.std(axis=1, keepdims=True) + 1e-8)

def pad_or_truncate(ecg, target_length=4096):
    current_length = ecg.shape[1]
    if current_length > target_length:
        return ecg[:, :target_length]
    elif current_length < target_length:
        pad_width = target_length - current_length
        return np.pad(ecg, ((0, 0), (0, pad_width)), mode='constant')
    else:
        return ecg

def extract_signal(record, target_length=4096):
    try:
        header = load_header(record)
        sfreq = int(get_sampling_frequency(header))

        signal, fields = load_signals(record)  # signal shape: (num_samples, num_leads) ó similar
        signal = signal.T                      # (12, T)  ← CORRECTO (antes usabas record.p_signal)

        # 1) Filtrado (si quieres)
        ecg_filtered = np.array(signal, dtype=np.float32)

        # 2) Normalizar por canal
        ecg_normalized = normalize_channels(ecg_filtered)

        # 3) Pad/Truncate a T=4096
        fixed = pad_or_truncate(ecg_normalized, target_length)

        return fixed
    except Exception as e:
        print(f"Error processing {record}: {e}")
        return None
    
# Save your trained model.
def save_model(model_folder, model, optimizer=None, epoch=None, extra=None):
    ckpt = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "extra": extra or {}
    }
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    filename = os.path.join(model_folder, "model.pt")
    torch.save(ckpt, filename)
    
#####################################################################################

def is_invalid(x):
    if x is None:
        return True
    try:
        return not np.isfinite(float(x))
    except (ValueError, TypeError):
        return True

def filter_records_by_folder(records, folder_keyword):
    """Return all records that contain a specific folder in their path."""
    return [f for f in records if folder_keyword in os.path.normpath(os.path.dirname(f))]

def sample_records(records, max_samples):
    """Return a random subset of records, up to max_samples."""
    return random.sample(records, min(max_samples, len(records)))

def select_records(data_folder, records, max_sami = 3000, max_ptb = 3000, max_negative_code = 6000, max_positive_code = 6000):
    #Check folders existance
    if not any(folder in os.listdir(data_folder) for folder in ["CODE-15%", "PTB-XL", "SaMi-Trop"]):
        print('Data folders not found: CODE-15%, PTB-XL, SaMi-Trop')
        return records, len(records)
    # Identify SaMi-Trop records and remove them from the list
    Sami_records = filter_records_by_folder(records, "SaMi-Trop")
    records = list(set(records) - set(Sami_records))
    Sami_records = sample_records(Sami_records, max_sami)
    
    # Identify PTB-XL records and remove them from the list
    PTB_records = filter_records_by_folder(records, "PTB-XL")
    records = list(set(records) - set(PTB_records))
    PTB_records = sample_records(PTB_records, max_ptb)

    # Identify CODE-15% records and remove them from the list
    code15_records = [f for f in records if os.path.dirname(f).endswith("CODE-15%")]
    records = list(set(records) - set(code15_records))
    
    positives, negatives = [], []
    #print(code15_records)
    for rc in code15_records:
        rc_path = os.path.join(data_folder, rc)
        label_rc = load_label(rc_path)
        #print(label_rc)
        if(label_rc == True):
            positives.append(rc)
        elif(label_rc == False):
            negatives.append(rc)
        else:
            continue;      
    
    positives = sample_records(positives, max_positive_code)
    negatives = sample_records(negatives, max_negative_code)
    # Combine all selected records
    final_records = Sami_records + PTB_records + positives + negatives
    
    return final_records, len(final_records)
