import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DATASET_DIR = "dataset/train"
MODEL_PATH = "plastic_detector_custom.pth"
NEW_MODEL_PATH = "plastic_detector_indian_v2.pth"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

def main():
    print("=" * 50)
    print("🚀 EcoScanIndia: Model Fine-tuning Pipeline")
    print("=" * 50)

    # 1. Check if dataset exists
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        print("Please run annotation_app.py first to label your images.")
        return

    # 2. Data Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load Dataset
    print("\nLoading dataset from disk...")
    full_dataset = ImageFolder(root=DATASET_DIR, transform=transform)
    num_classes = len(full_dataset.classes)
    print(f"Found {len(full_dataset)} total images across {num_classes} classes.")
    print(f"Classes: {full_dataset.classes}")

    # 4. Split Dataset (70/15/15)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    print(f"Split: Train({train_size}) | Val({val_size}) | Test({test_size})")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Load Existing Model
    print("\nLoading existing MobileNetV2 architecture...")
    model = models.mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    
    # We must first load it exactly as it was saved (2 classes)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, 2))
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    else:
        print("Existing model weights not found, starting from scratch.")

    # Now replace the classifier to match our new number of classes
    print(f"Replacing classification head to support {num_classes} classes.")
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, num_classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training Loop
    print("\nStarting Training...")
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / train_size
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Val Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            print(f"   -> Saving new best model ({best_acc:.2f}%)")
            torch.save(model.state_dict(), NEW_MODEL_PATH)

    # 7. Testing
    print("\nEvaluating on Test Set...")
    model.load_state_dict(torch.load(NEW_MODEL_PATH))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"\n✅ Final Test Accuracy: {test_acc:.2f}%")
    print(f"The new fine-tuned model has been saved as: {NEW_MODEL_PATH}")

if __name__ == "__main__":
    main()
