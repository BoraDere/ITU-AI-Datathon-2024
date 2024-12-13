import os
from tqdm import tqdm
import argparse
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision.transforms import Normalize

from resnet50 import ResNet50Model, init_weights

DATA_PATH = os.path.join(os.getcwd(), 'data')
TRAIN_CSV_PATH = os.path.join(os.getcwd(), 'data', 'train_data.csv')
TEST_CSV_PATH = os.path.join(os.getcwd(), 'data', 'test.csv')
MODEL_PATH = os.path.join(os.getcwd(), 'models')
RESULTS_PATH = os.path.join(os.getcwd(), 'results')

class ImageDataset(Dataset):
    """PyTorch dataset for image data."""

    def __init__(self, data_path, csv_path, classes, mode='train', random_state=42):
        """
        zadanie 1
        """
        self.data_path = data_path
        self.csv_path = csv_path
        self.classes = classes
        # self.target_size = target_size
        self.mode = mode
        self.random_state = random_state
        self.samples = self.load_samples_from_mode_()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def load_samples_from_mode_(self):
        samples = []
        df = pd.read_csv(self.csv_path)

        city_map = {
            'Istanbul': 0,
            'Ankara': 1,
            'Izmir': 2
        }

        if self.mode == 'train':
            for _, row in df.iterrows():
                samples.append((os.path.join(DATA_PATH, self.mode, row['filename']), city_map[row['city']]))
        else:
            for _, row in df.iterrows():
                samples.append((os.path.join(DATA_PATH, self.mode, row['filename']), ''))

        return samples

    def __len__(self):
        return len(self.samples)

    def __num_classes__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        image = ToTensor()(image)
        image = self.normalize(image)

        if label == '':  # For test mode where labels are not provided
            return image
        else:
            return image, label
    

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


def test_model(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc="Testing"):
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    
    return predictions


def run_model():
    """
    Argument parser for a complete model runnning configuration.

    :return: Parsed arguments, including device, experiment ID, learning rate, batch size and number of epochs.
    """

    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model', type=str, default='resnet50')

    # Device configuration (e.g., 'cuda:0' or 'cpu')
    parser.add_argument('--device', type=str, default='cuda:0')

    # Experiment ID for saving checkpoints and results
    parser.add_argument('--exp_id', type=str, default='exp_0')

    # Learning rate for the optimizer
    parser.add_argument('--lr', type=float, default=1e-3)

    # Batch size
    parser.add_argument('--bs', type=int, default=32)

    # Number of epochs
    parser.add_argument('--epoch', type=int, default=20)

    # Random state
    parser.add_argument('--random_state', type=int, default=42)

    # Mode
    parser.add_argument('--mode', type=str, default='train')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = run_model()

    torch.manual_seed(args.random_state)
    device = args.device
    classes = ['Istanbul', 'Ankara', 'Izmir']

    if args.mode == 'train':
        data = ImageDataset(
            data_path=DATA_PATH,
            csv_path=TRAIN_CSV_PATH, 
            classes=classes, 
            mode='train'
        )

        train_size = int(0.8 * len(data))
        valid_size = len(data) - train_size
        train_data, valid_data = random_split(data, [train_size, valid_size])

        train_loader = DataLoader(
            train_data, 
            batch_size=args.bs, 
            shuffle=True,
            num_workers=4
        )

        valid_loader = DataLoader(
            valid_data, 
            batch_size=args.bs, 
            shuffle=False,
            num_workers=4
        )

        model = ResNet50Model(num_classes=len(classes))
        model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # optimizer = optim.Adam(model.resnet50.fc.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        model.to(device)

        num_epochs = args.epoch
        best_val_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
            
            val_loss, val_acc = validate_epoch(model, valid_loader, criterion, device)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(MODEL_PATH, f'{args.model}_{args.exp_id}.pth'))
                print("Model saved.")

    elif args.mode == 'test':
        data = ImageDataset(
            data_path=DATA_PATH,
            csv_path=TEST_CSV_PATH, 
            classes=classes, 
            mode='test'
        )

        test_loader = DataLoader(
            data, 
            batch_size=args.bs, 
            shuffle=False,
            num_workers=4
        )

        model = ResNet50Model(num_classes=len(classes), pretrained=True)
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f'{args.model}_{args.exp_id}.pth')))
        model.to(device)

        predictions = test_model(model, test_loader, device)

        df = pd.read_csv(TEST_CSV_PATH)
        df['city'] = predictions

        city_map_r = {
            0: 'Istanbul',
            1: 'Ankara',
            2: 'Izmir'
        }

        df['city'] = df['city'].map(city_map_r)

        df.to_csv(os.path.join(RESULTS_PATH, f'{args.model}_{args.exp_id}.csv'), index=False)