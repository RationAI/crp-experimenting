import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import argparse


# ==========================================
# 1. Define the "One Layer" Models
# ==========================================
class OneConvLayerClassifier(nn.Module):
    def __init__(self):
        super(OneConvLayerClassifier, self).__init__()
        # 1 input channel (grayscale), 32 output channels (features), kernel size 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Flatten: 28x28 image -> maxpooled to 14x14. 32 channels * 14 * 14 = 6272
        self.fc = nn.Linear(32 * 14 * 14, 10) # 10 outputs for digits 0-9

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = self.fc(x)
        return x


class OneLinearLayerClassifier(nn.Module):
    def __init__(self):
        super(OneLinearLayerClassifier, self).__init__()
        # Single linear layer on flattened input: 28*28 -> 10
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_model(model_type: str):
    model_type = model_type.lower()
    if model_type == "conv":
        return OneConvLayerClassifier(), "one_layer_cnn_weights.pth"
    if model_type == "linear":
        return OneLinearLayerClassifier(), "one_layer_linear_weights.pth"
    raise ValueError(f"Unknown model_type '{model_type}', expected 'conv' or 'linear'.")

# ==========================================
# 2. Download or Load Dataset (MNIST)
# ==========================================
def get_data():
    print("Downloading/Loading MNIST Data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST Normalization
    ])
    
    # This automatically downloads MNIST if you don't have it
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    return train_data, test_data

# ==========================================
# 3. "Import" Weights (Training Simulation)
# ==========================================
def train_and_save_weights(model, train_loader, weights_path: str):
    """
    Since standard repositories don't host '1-layer-cnn' weights, 
    we create them in 20 seconds here.
    """
    print("Training model to generate weights...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    # Train for just 1 epoch (sufficient for ~95% accuracy on MNIST with this model)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"   Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved to '{weights_path}'")

# ==========================================
# 4. Main Execution Flow
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load a one-layer MNIST model (conv or linear).")
    parser.add_argument("--model-type", choices=["conv", "linear"], default="conv", help="Choose model architecture")
    args = parser.parse_args()

    # A. Setup Data
    train_data, test_data = get_data()
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    # B. Instantiate Model
    model, weights_path = get_model(args.model_type)

    # C. Check if we have weights, if not, create them
    try:
        model.load_state_dict(torch.load(weights_path))
        print(f"Loaded pre-existing weights from '{weights_path}'.")
    except FileNotFoundError:
        print(f"No weights found at '{weights_path}'. Creating them now...")
        train_and_save_weights(model, train_loader, weights_path)

    # D. Run on Dataset (Inference)
    print("\nRunning inference on 5 random test images...")
    model.eval()
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            if i >= 5:
                break
            output = model(image)
            prediction = output.argmax(dim=1, keepdim=True).item()
            print(f"Image {i+1}: True Label = {label.item()}, Model Prediction = {prediction}")
            
            # Optional: Visualize if you have matplotlib installed
            # plt.imshow(image.squeeze(), cmap='gray')
            # plt.show()
