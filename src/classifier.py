"""CNN-based character classifier using PyTorch.

Trains and evaluates a deep Convolutional Neural Network for OCR character
recognition. Architecture uses residual connections, batch normalization,
and aggressive augmentation for real-world robustness.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class ResidualBlock(nn.Module):
    """Residual block with skip connection for deeper networks."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class CNNClassifier(nn.Module):
    """Deep CNN for character classification with residual connections.

    Architecture:
        Conv(1→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool → Dropout(0.2)
        ResBlock(64) → Conv(64→128) → BN → ReLU → MaxPool → Dropout(0.2)
        ResBlock(128) → Conv(128→256) → BN → ReLU → MaxPool → Dropout(0.2)
        ResBlock(256) → Conv(256→256) → BN → ReLU → Dropout(0.2)
        GAP → FC(256→512) → ReLU → Dropout(0.5) → FC(512→num_classes)
    """

    def __init__(self, num_classes: int = 72):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            ResidualBlock(64),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            ResidualBlock(128),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            ResidualBlock(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class EmnistDataset(Dataset):
    """PyTorch Dataset wrapper for pre-loaded numpy arrays with augmentation."""

    def __init__(self, data: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.data = torch.from_numpy(data).unsqueeze(1).float()
        self.labels = torch.from_numpy(labels).long()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.data[idx]
        label = self.labels[idx]
        if self.augment:
            img = self._augment(img)
        return img, label

    @staticmethod
    def _augment(img: torch.Tensor) -> torch.Tensor:
        """Applies random augmentations during training."""
        import torch.nn.functional as tnf

        # Rotation ±15°
        angle = (torch.rand(1).item() - 0.5) * 30.0 * 3.14159 / 180.0
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Translation ±15%
        tx = (torch.rand(1).item() - 0.5) * 0.3
        ty = (torch.rand(1).item() - 0.5) * 0.3

        # Scale 0.85–1.15
        scale = 0.85 + torch.rand(1).item() * 0.3

        theta = torch.empty(1, 2, 3)
        theta[0, 0, 0] = cos_a * scale
        theta[0, 0, 1] = -sin_a * scale
        theta[0, 1, 0] = sin_a * scale
        theta[0, 1, 1] = cos_a * scale
        theta[0, 0, 2] = tx
        theta[0, 1, 2] = ty

        grid = tnf.affine_grid(theta, (1, 1, 28, 28), align_corners=False)
        img = tnf.grid_sample(
            img.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False
        ).squeeze(0)

        # Random erasing
        if torch.rand(1).item() < 0.4:
            erase_size = torch.randint(3, 8, (1,)).item()
            x = int(torch.randint(0, max(1, 28 - erase_size), (1,)).item())
            y = int(torch.randint(0, max(1, 28 - erase_size), (1,)).item())
            img[:, y:y + erase_size, x:x + erase_size] = 0.0

        return img


class OCRClassifier:
    """High-level wrapper for training, loading, and predicting with the CNN."""

    def __init__(
        self,
        model_path: str = "models/ocr_cnn.pth",
        num_classes: int = 72,
        device: str | None = None,
    ):
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNClassifier(num_classes=num_classes).to(self.device)

    def train_model(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
        batch_size: int = 128,
        epochs: int = 60,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        augment: bool = True,
    ) -> dict[str, list[float]]:
        """Trains the CNN with augmentation, LR scheduling, and early stopping."""
        train_dataset = EmnistDataset(train_data, train_labels, augment=augment)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        )

        val_loader = None
        if val_data is not None and val_labels is not None:
            val_dataset = EmnistDataset(val_data, val_labels, augment=False)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
            )

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        }
        best_val_acc = 0.0
        best_epoch = 0
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)

            epoch_train_loss = train_loss / train_total
            epoch_train_acc = train_correct / train_total
            history["train_loss"].append(epoch_train_loss)
            history["train_acc"].append(epoch_train_acc)

            if val_loader:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * images.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == labels).sum().item()
                        val_total += labels.size(0)

                epoch_val_loss = val_loss / val_total
                epoch_val_acc = val_correct / val_total
                history["val_loss"].append(epoch_val_loss)
                history["val_acc"].append(epoch_val_acc)
                scheduler.step()

                if epoch_val_acc > best_val_acc:
                    best_val_acc = epoch_val_acc
                    best_epoch = epoch + 1
                    patience_counter = 0
                    self.save_model(self.model_path.replace(".pth", "_best.pth"))
                else:
                    patience_counter += 1
            else:
                history["val_loss"].append(0.0)
                history["val_acc"].append(0.0)
                scheduler.step()

            print(
                f"Epoch {epoch + 1:2d}/{epochs} | "
                f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
                f"Val Loss: {history['val_loss'][-1]:.4f} Acc: {history['val_acc'][-1]:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.save_model()
        print(f"Training complete. Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")
        return history

    @torch.no_grad()
    def predict(self, char_images: np.ndarray) -> list[int]:
        """Predicts class indices for a batch of character images."""
        if len(char_images.shape) != 3 or char_images.shape[1:] != (28, 28):
            raise ValueError(f"Expected (N, 28, 28) array, got {char_images.shape}.")
        self.model.eval()
        tensor = torch.from_numpy(char_images).unsqueeze(1).float().to(self.device)
        outputs = self.model(tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.cpu().tolist()

    @torch.no_grad()
    def predict_proba(self, char_images: np.ndarray) -> np.ndarray:
        """Returns prediction probabilities for a batch of character images."""
        self.model.eval()
        tensor = torch.from_numpy(char_images).unsqueeze(1).float().to(self.device)
        outputs = self.model(tensor)
        probs = F.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    def save_model(self, path: str | None = None) -> str:
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        return path

    def load_model(self, path: str | None = None) -> None:
        path = path or self.model_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
