from datetime import datetime
import time
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import os
import mlflow
import mlflow.pytorch
import random
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from face_binary_net import EfficientNetExpressionNet, EfficientNetGenderNet, MobilenetBinaryNet, MobilenetAgeNet, MobilenetGenderNet, MobilenetExpressionNet
from datasets import SimpleFaceDataset, AgesDataset, GendersDataset, ExpressionsDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
mlflow_dir = os.path.join(script_dir, "mlruns")
os.makedirs(mlflow_dir, exist_ok=True)
mlflow.set_tracking_uri(f"file:{mlflow_dir}")


def get_transforms():
    """Returns train and validation transforms"""
    train_transform = T.Compose([
        T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform


def train_model(model, optimizer, criterion, train_loader, val_loader, model_name, num_epochs=10, lr=1e-4, use_cpu=False):
    """
    Generic training function with MLflow tracking
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model_name: Name for saving model files (e.g., 'face_binary', 'age', 'gender', 'expression')
        num_epochs: Number of training epochs
        lr: Learning rate
        use_cpu: Force CPU training even if GPU is available (default: False)
    """
    # Determine device based on parameter and availability
    training_device = torch.device("cpu") if use_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure we're using the correct tracking URI
    mlflow.set_tracking_uri(f"file:{mlflow_dir}")
    mlflow.set_experiment(f"{model_name}_training")
    
    with mlflow.start_run(run_name=f"{model_name}_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log hyperparameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("device", str(training_device))
        mlflow.log_param("use_cpu", use_cpu)
        
        model = model.to(training_device)
        
        best_val_acc = 0.0
        best_pth_path = None
        
        # Start training time tracking
        training_start_time = time.time()
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images = images.to(training_device)
                labels = labels.to(training_device)
                
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            train_loss = running_loss / total
            train_acc = correct / total
            
            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss_sum = 0.0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(training_device)
                    labels = labels.to(training_device)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    val_loss_sum += loss.item() * images.size(0)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            val_loss = val_loss_sum / val_total if val_total > 0 else 0.0
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            
            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            
            print(
                f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}"
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                pth_path = os.path.join(script_dir, f"models/{model_name}.pth")
                torch.save(model.state_dict(), pth_path)
                print(f"Saved best model to: {pth_path} (Val Acc: {val_acc:.3f})")
                best_pth_path = pth_path
        
        # Calculate and log total training time
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        mlflow.log_metric("total_training_time_seconds", total_training_time)
        mlflow.log_metric("total_training_time_minutes", total_training_time / 60)
        print(f"\nTotal training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
        
        # Compute and save confusion matrices (use best weights if available)
        try:
            model.eval()
            if best_pth_path and os.path.exists(best_pth_path):
                state = torch.load(best_pth_path, map_location=training_device)
                model.load_state_dict(state)
                print(f"Loaded best weights from {best_pth_path} for confusion matrix evaluation.")

            def _compute_cm(loader):
                y_true = []
                y_pred = []
                num_classes = None
                with torch.no_grad():
                    for images, labels in loader:
                        images = images.to(training_device)
                        labels = labels.to(training_device)
                        logits = model(images)
                        if num_classes is None:
                            num_classes = logits.shape[1]
                        preds = logits.argmax(dim=1)
                        y_true.extend(labels.cpu().numpy().tolist())
                        y_pred.extend(preds.cpu().numpy().tolist())
                labels_range = list(range(num_classes if num_classes is not None else max(max(y_true), max(y_pred)) + 1))
                cm = confusion_matrix(y_true, y_pred, labels=labels_range)
                return cm, len(labels_range)

            def _get_class_names(name: str, n: int):
                name = name.lower()
                if "gender" in name:
                    return ["female", "male"][:n]
                if "expression" in name:
                    return ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"][:n]
                if "age" in name:
                    return ["YOUNG", "MIDDLE", "OLD"][:n]
                if "face" in name:
                    return ["not_face", "face"][:n]
                return [str(i) for i in range(n)]

            train_cm, train_classes = _compute_cm(train_loader)
            val_cm, val_classes = _compute_cm(val_loader)
            num_classes = max(train_classes, val_classes)
            class_names = _get_class_names(model_name, num_classes)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, cm, title in [(axes[0], train_cm, "Training Confusion Matrix"), (axes[1], val_cm, "Validation Confusion Matrix")]:
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.set_title(title)
                tick_marks = np.arange(len(class_names))
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(class_names, rotation=45, ha='right')
                ax.set_yticklabels(class_names)
                ax.set_ylabel('True label')
                ax.set_xlabel('Predicted label')
                thresh = cm.max() / 2.0 if cm.size > 0 else 0
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], 'd'),
                                ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()

            models_dir = os.path.join(script_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            cm_path = os.path.join(models_dir, f"{model_name}_confusion.png")
            fig.savefig(cm_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(cm_path)
            print(f"Saved combined confusion matrix image to: {cm_path}")
        except Exception as e:
            print(f"Warning: Failed to compute confusion matrices: {e}")
        
        # Log best validation accuracy
        mlflow.log_metric("best_val_acc", best_val_acc)
        
        # Export to ONNX
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224).to(training_device)
        onnx_path = os.path.join(script_dir, f"models/{model_name}.onnx")
        onnx_data_path = os.path.join(script_dir, f"models/{model_name}.onnx.data")
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Exported model to ONNX: {onnx_path}")
        
        # Log artifacts to MLflow
        mlflow.log_artifact(pth_path)
        mlflow.log_artifact(onnx_path)
        if os.path.exists(onnx_data_path):
            mlflow.log_artifact(onnx_data_path)
            print(f"Logged ONNX data file: {onnx_data_path}")
        mlflow.pytorch.log_model(model, "model")
        
        print(f"Training complete! Best Val Acc: {best_val_acc:.3f}")
        return model, best_val_acc


def train_face_binary():
    """Train face/not_face binary classifier"""
    print("\n" + "="*50)
    print("Training Face Binary Classifier")
    print("="*50)
    
    data_path = os.path.join(script_dir, "data/faces")
    
    # Build and split dataset
    tmp_dataset = SimpleFaceDataset(data_path, sample_ratio=0.2)
    all_samples = tmp_dataset.samples
    random.seed(42)
    random.shuffle(all_samples)
    
    val_ratio = 0.2
    val_size = int(len(all_samples) * val_ratio)
    train_samples = all_samples[val_size:]
    val_samples = all_samples[:val_size]
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = SimpleFaceDataset(samples=train_samples, transform=train_transform)
    val_dataset = SimpleFaceDataset(samples=val_samples, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = MobilenetBinaryNet(pretrained=True)
    return train_model(model, train_loader, val_loader, "face_binary", num_epochs=1)


def train_age_classifier():
    """Train age group classifier (YOUNG, MIDDLE, OLD)"""
    print("\n" + "="*50)
    print("Training Age Classifier")
    print("="*50)
    
    train_path = os.path.join(script_dir, "data/ages2_preprocessed/train")
    test_path = os.path.join(script_dir, "data/ages2_preprocessed/test")
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = AgesDataset(root_dir=train_path, transform=train_transform, sample_ratio=1)
    val_dataset = AgesDataset(root_dir=test_path, transform=val_transform, sample_ratio=1)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = MobilenetAgeNet(pretrained=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    return train_model(model, optimizer, criterion, train_loader, val_loader, "age", num_epochs=15, use_cpu=True)


def train_gender_classifier():
    """Train gender classifier"""
    print("\n" + "="*50)
    print("Training Gender Classifier")
    print("="*50)
    
    train_path = os.path.join(script_dir, "data/gender2_preprocessed/train")
    val_path = os.path.join(script_dir, "data/gender2_preprocessed/test")
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = GendersDataset(root_dir=train_path, transform=train_transform, sample_ratio=1)
    val_dataset = GendersDataset(root_dir=val_path, transform=val_transform, sample_ratio=1)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = MobilenetGenderNet(pretrained=True)
    # model = EfficientNetGenderNet(pretrained=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    # return train_model(model, optimizer, criterion, train_loader, val_loader, "gender_efficient_net", num_epochs=5)
    return train_model(model, optimizer, criterion, train_loader, val_loader, "gender", num_epochs=5, use_cpu=False)


def train_expression_classifier():
    """Train facial expression classifier"""
    print("\n" + "="*50)
    print("Training Expression Classifier")
    print("="*50)
    
    train_path = os.path.join(script_dir, "data/expressions2_preprocessed/train")
    val_path = os.path.join(script_dir, "data/expressions2_preprocessed/test")
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = ExpressionsDataset(root_dir=train_path, transform=train_transform, sample_ratio=1)
    val_dataset = ExpressionsDataset(root_dir=val_path, transform=val_transform, sample_ratio=1)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = MobilenetExpressionNet(pretrained=True)
    # optimizer = optim.Adam(model.parameters(), lr=3e-4)
    # model = MobilenetExpressionNet(freeze_backbone=False)
    # model = EfficientNetExpressionNet(pretrained=True)
    for param in model.backbone.features[:5].parameters():
        param.requires_grad = False

    for param in model.backbone.features[5:].parameters():
        param.requires_grad = True

    optimizer = optim.Adam([
        {"params": model.backbone.features[5:].parameters(), "lr": 5e-5},
        {"params": model.backbone.classifier.parameters(), "lr": 3e-4}
    ])
    criterion = nn.CrossEntropyLoss()
    return train_model(model, optimizer, criterion, train_loader, val_loader, "expression", num_epochs=20)
    # return train_model(model, optimizer, criterion, train_loader, val_loader, "expression_efficient_net", num_epochs=20)


if __name__ == "__main__":
    # MLflow tracking URI is already set at module level
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow artifacts directory: {mlflow_dir}\n")
    
    # Train all models
    print("Starting training for all models...\n")
    
    # Uncomment the models you want to train:
    # train_face_binary()
    # train_age_classifier()
    # train_gender_classifier()
    train_expression_classifier()
    
    print("\n" + "="*50)
    print("All training complete!")
    print("="*50)