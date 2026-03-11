# ================================================================
# Face Attribute Classifier — CelebA + MobileNetV2
# Transfer Learning | Multi-label Classification
# Attributes: Eyeglasses, Wearing_Hat, Narrow_Eyes, Smiling, Male
# ================================================================

import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

SEED       = 42
N_SAMPLES  = 15000
BATCH_SIZE = 64
EPOCHS     = 15
LR         = 1e-3
ATTRS      = ["Eyeglasses", "Wearing_Hat", "Narrow_Eyes", "Smiling", "Male"]

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Paths ────────────────────────────────────────────────────────
ATTR_CSV = "/content/celeba/list_attr_celeba.csv"
IMG_DIR  = "/content/celeba/img_align_celeba/img_align_celeba"

# ── Load & Prepare CSV ───────────────────────────────────────────
df = pd.read_csv(ATTR_CSV)
print(f"Total rows: {len(df)}")

df[ATTRS] = (df[ATTRS] == 1).astype(int)

print("\nAttribute distribution (% positive):")
print((df[ATTRS].mean() * 100).round(1))

df = df.sample(n=N_SAMPLES, random_state=SEED).reset_index(drop=True)
print(f"\nSampled: {len(df)} rows")

# ── Split ────────────────────────────────────────────────────────
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED)
val_df,   test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── Dataset ──────────────────────────────────────────────────────
class CelebADataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        image    = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = torch.FloatTensor(row[ATTRS].values.astype(float))
        return image, labels

train_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_loader = DataLoader(
    CelebADataset(train_df, IMG_DIR, train_tf),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    CelebADataset(val_df, IMG_DIR, val_tf),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)
test_loader = DataLoader(
    CelebADataset(test_df, IMG_DIR, val_tf),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

print("Dataloaders ready.")

# ── Model ────────────────────────────────────────────────────────
def build_model(n_attrs=5):
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, n_attrs),
    )
    return model

model      = build_model(len(ATTRS)).to(device)
pos_weight = torch.tensor([2.0, 5.0, 8.0, 1.0, 1.0]).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=2, factor=0.5
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model ready — trainable params: {trainable:,}")

# ── Training Loop ────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs     = model(images)
            total_loss += criterion(outputs, labels).item()
            preds       = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    acc = (all_preds == all_labels).mean()
    return total_loss / len(loader), acc, all_preds, all_labels

print("\n── Training ────────────────────────────────────────────")
best_val_loss = float("inf")
best_state    = None

for epoch in range(1, EPOCHS + 1):
    train_loss              = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion)
    scheduler.step(val_loss)

    print(f"Epoch {epoch:2d}/{EPOCHS}  "
          f"Train Loss: {train_loss:.4f}  "
          f"Val Loss: {val_loss:.4f}  "
          f"Val Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}

# ── Test Evaluation ──────────────────────────────────────────────
print("\n── Test Evaluation ─────────────────────────────────────")
model.load_state_dict(best_state)
_, test_acc, test_preds, test_labels = eval_epoch(
    model, test_loader, criterion
)

print(f"Overall Test Accuracy: {test_acc:.4f}\n")
for i, attr in enumerate(ATTRS):
    print(f"── {attr} ──")
    print(classification_report(
        test_labels[:, i], test_preds[:, i],
        target_names=[f"No {attr}", attr],
        zero_division=0
    ))

# ── Save ─────────────────────────────────────────────────────────
SAVE_PATH = "/content/drive/MyDrive/face_attributes.pt"
torch.save({
    "model_state" : best_state,
    "attrs"       : ATTRS,
    "val_loss"    : best_val_loss,
    "test_acc"    : test_acc,
}, SAVE_PATH)

print(f"\nSaved: {SAVE_PATH}")
print("Download face_attributes.pt to your local project folder.")