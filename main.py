import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
import torchaudio
import librosa
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle

from pre_data import *
from model import *

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for (inputs, labels) in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)

def pad_collate(batch):
    max_length = max([item[0].shape[0] for item in batch])
    padded_inputs = []
    labels = []

    for item in batch:
        input_tensor, label = item
        padded_input = torch.zeros(max_length, input_tensor.shape[1])
        padded_input[: input_tensor.shape[0], :] = input_tensor
        padded_inputs.append(padded_input.unsqueeze(0)) # 수정된 부분
        labels.append(label)

    padded_inputs = torch.cat(padded_inputs, dim=0) # 수정된 부분
    labels = torch.tensor(labels)
    return padded_inputs, labels

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return running_loss / len(val_loader), accuracy

def main():
    with open("data/path.pkl", "rb") as file:
        train_file_paths = pickle.load(file)
    with open("data/label.pkl", "rb") as file:
        train_labels = pickle.load(file)
    with open("data/val_path.pkl", "rb") as file:
        val_file_paths = pickle.load(file)
    with open("data/val_label.pkl", "rb") as file:
        val_labels = pickle.load(file)

    input_size = 128 # Set to the number of Mel features used
    num_classes = 209 # Set to the number of classes in your dataset
    
    train_dataset = AudioDataset(train_file_paths, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=pad_collate)

    val_dataset = AudioDataset(val_file_paths, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=pad_collate)

    model = TransformerAudioClassifier(input_size, num_classes).to("cuda:0")

    criterion = nn.CrossEntropyLoss()
    op = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    best_loss = float('inf')
    save_path = "best_model.pth"
    scheduler = torch.optim.lr_scheduler.StepLR(op, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, criterion, op, "cuda:0")
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, "cuda:0")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save the model with the best validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        
        scheduler.step()

if __name__ == "__main__":
    main()