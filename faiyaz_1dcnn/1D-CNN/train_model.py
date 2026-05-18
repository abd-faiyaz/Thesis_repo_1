"""Training script for ByteCNN malware detector.

Trains the 1-D CNN model on APK samples from benign/ and malware/ directories.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

# Import ByteCNN from malware_checker
sys.path.insert(0, str(Path(__file__).parent))
from malware_checker import ByteCNN


class APKDataset(Dataset):
    """Dataset for loading APK files as byte sequences with labels."""

    def __init__(self, benign_dir: Path, malware_dir: Path, byte_length: int = 1024, from_end: bool = True):
        self.byte_length = byte_length
        self.from_end = from_end
        self.samples = []
        
        # Load benign APKs (label 0)
        for path in sorted(benign_dir.glob("*.apk")):
            self.samples.append((path, 0))
        
        # Load malware APKs (label 1)
        for path in sorted(malware_dir.glob("*.apk")):
            self.samples.append((path, 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        data = path.read_bytes()
        
        # Extract segment (from end or beginning)
        if self.from_end:
            segment = data[-self.byte_length:] if len(data) >= self.byte_length else data.rjust(self.byte_length, b"\0")
        else:
            segment = data[:self.byte_length].ljust(self.byte_length, b"\0")
        
        # Convert bytes to tensor
        tensor = torch.tensor(list(segment), dtype=torch.long)
        return tensor, label


def train_epoch(model: ByteCNN, train_loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(train_loader)


def evaluate(model: ByteCNN, val_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Evaluate on validation set. Returns (accuracy, loss)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    accuracy = correct / total
    avg_loss = total_loss / len(val_loader)
    return accuracy, avg_loss


def main() -> int:
    # Configuration
    samples_dir = Path("1D CNN") / "Samples"
    benign_dir = samples_dir / "benign"
    malware_dir = samples_dir / "malware"
    output_path = Path("1D CNN") / "trained_model.pth"
    
    byte_length = 1024
    batch_size = 8
    num_epochs = 50
    learning_rate = 0.001
    
    # Check directories exist
    if not benign_dir.exists() or not malware_dir.exists():
        raise SystemExit(f"Directories not found: benign={benign_dir.exists()}, malware={malware_dir.exists()}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    print("Loading dataset...")
    dataset = APKDataset(benign_dir, malware_dir, byte_length=byte_length, from_end=True)
    print(f"Total samples: {len(dataset)}")
    
    # Split into train (80%) and validation (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = ByteCNN(embed_dim=8, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nTraining for {num_epochs} epochs...")
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_acc, val_loss = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"  ✓ Saved best model (acc: {val_acc:.4f})")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
