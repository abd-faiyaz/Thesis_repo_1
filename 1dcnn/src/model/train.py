import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from model.dataset import APKDataset
from model.bytecnn import ByteCNN
from logger import get_logger


logger = get_logger("Model Train")


def train_epoch(model, train_loader, optimizer, device, verbose=True):
    """
    Train model for one epoch and return average loss.
    """
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
        if verbose and (batch_idx + 1) % 5 == 0:
            logger.info(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(train_loader)


def evaluate(model, val_loader, device):
    """
    Evaluate on validation set. Returns (accuracy, loss).
    """
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


def train_model(year_dir, model_save_path, byte_length=1024, batch_size=8, 
                epochs=50, learning_rate=0.001, from_end=True, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading dataset...")
    dataset = APKDataset(year_dir, byte_length=byte_length, from_end=from_end)
    logger.info(f"Total samples: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ByteCNN(embed_dim=8, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logger.info(f"\nTraining for {epochs} epochs...")
    best_val_acc = 0.0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, verbose=verbose)
        val_acc, val_loss = evaluate(model, val_loader, device)
        
        if verbose:
            logger.info(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved best model (acc: {val_acc:.4f})")
    
    logger.info(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Model saved to: {model_save_path}")