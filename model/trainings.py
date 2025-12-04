from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import os
import mlflow
import mlflow.pytorch
import random
import torchvision.transforms as T

from face_binary_net import MobilenetBinaryNet, MobilenetAgeNet, MobilenetGenderNet, MobilenetExpressionNet
from datasets import SimpleFaceDataset, AgesDataset, GendersDataset, ExpressionsDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def train_model(model, train_loader, val_loader, model_name, num_epochs=10, lr=1e-4):
    """
    Generic training function with MLflow tracking
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model_name: Name for saving model files (e.g., 'face_binary', 'age', 'gender', 'expression')
        num_epochs: Number of training epochs
        lr: Learning rate
    """
    # Ensure we're using the correct tracking URI
    mlflow.set_tracking_uri(f"file:{mlflow_dir}")
    mlflow.set_experiment(f"{model_name}_training")
    
    with mlflow.start_run(run_name=f"{model_name}_run"):
        # Log hyperparameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("device", str(device))
        
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                
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
                    images = images.to(device)
                    labels = labels.to(device)
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
                pth_path = os.path.join(script_dir, f"{model_name}.pth")
                torch.save(model.state_dict(), pth_path)
                print(f"Saved best model to: {pth_path} (Val Acc: {val_acc:.3f})")
        
        # Log best validation accuracy
        mlflow.log_metric("best_val_acc", best_val_acc)
        
        # Export to ONNX
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        onnx_path = os.path.join(script_dir, f"{model_name}.onnx")
        onnx_data_path = os.path.join(script_dir, f"{model_name}.onnx.data")
        
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
    tmp_dataset = SimpleFaceDataset(data_path)
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
    return train_model(model, train_loader, val_loader, "face_binary", num_epochs=3)


def train_age_classifier():
    """Train age range classifier"""
    print("\n" + "="*50)
    print("Training Age Classifier")
    print("="*50)
    
    train_path = os.path.join(script_dir, "data/ages/train")
    test_path = os.path.join(script_dir, "data/ages/test")
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = AgesDataset(root_dir=train_path, transform=train_transform)
    val_dataset = AgesDataset(root_dir=test_path, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = MobilenetAgeNet(pretrained=True)
    return train_model(model, train_loader, val_loader, "age", num_epochs=15)


def train_gender_classifier():
    """Train gender classifier"""
    print("\n" + "="*50)
    print("Training Gender Classifier")
    print("="*50)
    
    train_path = os.path.join(script_dir, "data/gender/Training")
    val_path = os.path.join(script_dir, "data/gender/Validation")
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = GendersDataset(root_dir=train_path, transform=train_transform)
    val_dataset = GendersDataset(root_dir=val_path, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = MobilenetGenderNet(pretrained=True)
    return train_model(model, train_loader, val_loader, "gender", num_epochs=10)


def train_expression_classifier():
    """Train facial expression classifier"""
    print("\n" + "="*50)
    print("Training Expression Classifier")
    print("="*50)
    
    train_path = os.path.join(script_dir, "data/expressions/train")
    val_path = os.path.join(script_dir, "data/expressions/validation")
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = ExpressionsDataset(root_dir=train_path, transform=train_transform)
    val_dataset = ExpressionsDataset(root_dir=val_path, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = MobilenetExpressionNet(pretrained=True)
    return train_model(model, train_loader, val_loader, "expression", num_epochs=15)


if __name__ == "__main__":
    # MLflow tracking URI is already set at module level
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow artifacts directory: {mlflow_dir}\n")
    
    # Train all models
    print("Starting training for all models...\n")
    
    # Uncomment the models you want to train:
    train_face_binary()
    # train_age_classifier()
    # train_gender_classifier()
    # train_expression_classifier()
    
    print("\n" + "="*50)
    print("All training complete!")
    print("="*50)