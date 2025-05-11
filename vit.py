import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import math
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
from utils.metrics import calculate_metrics, plot_confusion_matrix
from utils.visualization import plot_training_history

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PatchEmbedding(nn.Module):
    def __init__(self, seq_length=23, patch_size=1, in_channels=4, embed_dim=64):
        super().__init__()
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.num_patches = (seq_length // patch_size)
        
        self.proj = nn.Linear(patch_size * in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unfold(1, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, self.num_patches, -1)
        x = self.proj(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_length, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        x = (attn_probs @ v).transpose(1, 2).reshape(batch_size, seq_length, embed_dim)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, seq_length=23, patch_size=1, in_channels=4, 
                 embed_dim=64, depth=6, num_heads=8, mlp_ratio=4, 
                 dropout=0.1, num_classes=1):
        super().__init__()
        self.patch_embed = PatchEmbedding(seq_length, patch_size, in_channels, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        x = self.head(cls_token)
        return x

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    val_metrics = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Calculate metrics using your framework's function
        metrics = calculate_metrics(all_labels, all_preds)
        val_metrics.append(metrics)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Validation Metrics: {metrics}')
    
    # Final evaluation on test set
    test_preds, test_labels = evaluate_model(model, test_loader)
    test_metrics = calculate_metrics(test_labels, test_preds)
    
    # Visualization
    plot_training_history(train_losses, val_losses, val_metrics)
    plot_confusion_matrix(test_labels, test_preds)
    
    return model, test_metrics

def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels

def main():
    # Load your data
    # X_train, y_train, X_val, y_val, X_test, y_test = load_your_data()
    
    # Example with random data (replace with your actual data)
    X_train = np.random.randn(268410, 23, 4)  # 80% of 335513
    y_train = np.random.randint(0, 2, (268410,))
    X_val = np.random.randn(33551, 23, 4)  # 10%
    y_val = np.random.randint(0, 2, (33551,))
    X_test = np.random.randn(33552, 23, 4)  # 10%
    y_test = np.random.randint(0, 2, (33552,))
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    # Create dataloaders
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = VisionTransformer(
        seq_length=23,
        patch_size=1,
        in_channels=4,
        embed_dim=64,
        depth=6,
        num_heads=8,
        num_classes=1
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Train the model
    trained_model, test_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=20
    )
    
    print(f'\nFinal Test Metrics: {test_metrics}')
    torch.save(trained_model.state_dict(), 'vit_timeseries_model.pth')

if __name__ == "__main__":
    main()
