"""
Predictor script for Age, Gender, and Expression recognition using PyTorch models.
Assumes images are already cropped to face regions.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from face_binary_net import (
    MobilenetAgeNet,
    MobilenetGenderNet,
    MobilenetExpressionNet,
)


class ImagePreprocessor:
    """Handles image preprocessing matching the React app preprocessing."""
    
    # ImageNet normalization constants (same as React app)
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    INPUT_SIZE = 224
    
    @staticmethod
    def preprocess(image_data: np.ndarray) -> torch.Tensor:
        """
        Preprocess image data to match React app preprocessing.
        
        Args:
            image_data: RGB image as numpy array (H, W, 3) with values [0, 255]
            
        Returns:
            PyTorch tensor (1, 3, 224, 224) normalized with ImageNet stats
        """
        # Normalize to [0, 1]
        image_data = image_data.astype(np.float32) / 255.0
        
        # Normalize with ImageNet mean/std
        image_data = (image_data - ImagePreprocessor.MEAN) / ImagePreprocessor.STD
        
        # Convert to tensor and add batch dimension (1, 3, H, W)
        tensor = torch.from_numpy(image_data).permute(2, 0, 1).unsqueeze(0).float()
        
        return tensor
    
    @staticmethod
    def denormalize_and_display(
        image_tensor: torch.Tensor, 
        output_path: Path,
        image_name: str
    ) -> None:
        """
        Denormalize and save preprocessed image for visualization.
        
        Args:
            image_tensor: Normalized tensor (1, 3, H, W)
            output_path: Directory to save the image
            image_name: Name of the original image
        """
        try:
            # Remove batch dimension and move to CPU
            image_tensor = image_tensor.squeeze(0).cpu()
            
            # Denormalize: reverse the normalization
            image_denorm = image_tensor.numpy().copy()
            
            # Apply denormalization to each channel
            mean = ImagePreprocessor.MEAN
            std = ImagePreprocessor.STD
            image_denorm[0] = image_denorm[0] * std[0] + mean[0]  # R
            image_denorm[1] = image_denorm[1] * std[1] + mean[1]  # G
            image_denorm[2] = image_denorm[2] * std[2] + mean[2]  # B
            
            # Convert to HWC format
            image_denorm = np.transpose(image_denorm, (1, 2, 0))
            
            # Clip to [0, 1] and convert to uint8
            image_denorm = np.clip(image_denorm, 0, 1)
            image_denorm = (image_denorm * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_denorm, cv2.COLOR_RGB2BGR)
            
            # Create output directory
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output filename
            stem = Path(image_name).stem
            output_file = output_dir / f"{stem}_preprocessed.jpg"
            
            # Save image
            success = cv2.imwrite(str(output_file), image_bgr)
            if success:
                print(f"\n→ Preprocessed image saved: {output_file.name}")
            else:
                print(f"\n✗ Failed to save preprocessed image")
        except Exception as e:
            print(f"\n✗ Error saving preprocessed image: {str(e)}")


class ModelPredictor:
    """Handles model predictions for age, gender, and expression."""
    
    def __init__(self, models_dir: str = "./models", device: str = "cpu"):
        """
        Initialize all models.
        
        Args:
            models_dir: Directory containing .pth model files
            device: Device to load models on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.models_dir = Path(models_dir)
        
        # Load models
        self.age_model = self._load_model(
            MobilenetAgeNet, 
            self.models_dir / "age.pth"
        )
        self.gender_model = self._load_model(
            MobilenetGenderNet, 
            self.models_dir / "gender.pth"
        )
        self.expression_model = self._load_model(
            MobilenetExpressionNet, 
            self.models_dir / "expression.pth"
        )
        
        # Class labels
        self.age_labels = ["Young", "Middle", "Old"]
        self.gender_labels = ["Female", "Male"]
        self.expression_labels = [
            "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"
        ]
    
    def _load_model(self, model_class, model_path: Path):
        """Load a model from checkpoint."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = model_class(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Run predictions on all models.
        
        Args:
            image_tensor: Preprocessed image tensor (1, 3, 224, 224)
            
        Returns:
            Dictionary with predictions for all models
        """
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            results: Dict[str, Any] = {}

            # Age classification
            age_logits = self.age_model(image_tensor)
            age_probs = F.softmax(age_logits, dim=1)
            age_pred_idx = age_probs.argmax(dim=1).item()
            age_confidence = age_probs[0, age_pred_idx].item()

            results["age"] = {
                "label": self.age_labels[age_pred_idx],
                "confidence": age_confidence,
                "probabilities": {
                    "YOUNG": age_probs[0, 0].item(),
                    "MIDDLE": age_probs[0, 1].item(),
                    "OLD": age_probs[0, 2].item(),
                }
            }

            # Gender classification
            gender_logits = self.gender_model(image_tensor)
            gender_probs = F.softmax(gender_logits, dim=1)
            gender_pred_idx = gender_probs.argmax(dim=1).item()
            gender_confidence = gender_probs[0, gender_pred_idx].item()

            results["gender"] = {
                "label": self.gender_labels[gender_pred_idx],
                "confidence": gender_confidence,
                "probabilities": {
                    "female": gender_probs[0, 0].item(),
                    "male": gender_probs[0, 1].item(),
                }
            }

            # Expression classification
            expr_logits = self.expression_model(image_tensor)
            expr_probs = F.softmax(expr_logits, dim=1)
            expr_pred_idx = expr_probs.argmax(dim=1).item()
            expr_confidence = expr_probs[0, expr_pred_idx].item()

            results["expression"] = {
                "label": self.expression_labels[expr_pred_idx],
                "confidence": expr_confidence,
                "probabilities": {
                    "angry": expr_probs[0, 0].item(),
                    "disgust": expr_probs[0, 1].item(),
                    "fear": expr_probs[0, 2].item(),
                    "happy": expr_probs[0, 3].item(),
                    "neutral": expr_probs[0, 4].item(),
                    "sad": expr_probs[0, 5].item(),
                    "surprise": expr_probs[0, 6].item(),
                }
            }

        return results


def print_results(image_path: str, results: Dict[str, Any]):
    """Pretty print prediction results."""
    print(f"\n{'='*70}")
    print(f"Image: {image_path}")
    print(f"{'='*70}")
    
    if "age" in results:
        age_result = results["age"]
        print(f"\nAge Group: {age_result['label']}")
        print(f"  Confidence: {age_result['confidence']:.2%}")
        print(f"  Young: {age_result['probabilities']['YOUNG']:.2%}")
        print(f"  Middle: {age_result['probabilities']['MIDDLE']:.2%}")
        print(f"  Old: {age_result['probabilities']['OLD']:.2%}")
    
    if "gender" in results:
        gender_result = results["gender"]
        print(f"\nGender: {gender_result['label']}")
        print(f"  Confidence: {gender_result['confidence']:.2%}")
        print(f"  Female: {gender_result['probabilities']['female']:.2%}")
        print(f"  Male: {gender_result['probabilities']['male']:.2%}")
    
    if "expression" in results:
        expr_result = results["expression"]
        print(f"\nExpression: {expr_result['label']}")
        print(f"  Confidence: {expr_result['confidence']:.2%}")
        print(f"  Angry: {expr_result['probabilities']['angry']:.2%}")
        print(f"  Disgust: {expr_result['probabilities']['disgust']:.2%}")
        print(f"  Fear: {expr_result['probabilities']['fear']:.2%}")
        print(f"  Happy: {expr_result['probabilities']['happy']:.2%}")
        print(f"  Neutral: {expr_result['probabilities']['neutral']:.2%}")
        print(f"  Sad: {expr_result['probabilities']['sad']:.2%}")
        print(f"  Surprise: {expr_result['probabilities']['surprise']:.2%}")


def main():
    """Main entry point for the predictor script."""
    # Get samples directory
    script_dir = Path(__file__).parent
    samples_dir = script_dir / "samples"
    models_dir = script_dir / "models"
    
    # Check if samples directory exists
    if not samples_dir.exists():
        print(f"Error: Samples directory not found: {samples_dir}")
        return
    
    # Check if models directory exists
    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        return
    
    # Initialize components
    print("Loading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    predictor = ModelPredictor(models_dir=str(models_dir), device=device)
    
    preprocessor = ImagePreprocessor()
    
    # Process all image files in samples folder
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [
        f for f in samples_dir.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No image files found in {samples_dir}")
        return
    
    print(f"\nFound {len(image_files)} image(s) to process\n")
    print("Note: Assuming images are already cropped to face regions.\n")
    
    # Process each image
    for image_path in sorted(image_files):
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Error: Could not load image {image_path}")
                continue
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to target size (224x224)
            image_resized = cv2.resize(image_rgb, (preprocessor.INPUT_SIZE, preprocessor.INPUT_SIZE))
            
            # Preprocess
            image_tensor = preprocessor.preprocess(image_resized)
            
            # Display preprocessed image
            # preprocessor.denormalize_and_display(
            #     image_tensor,
            #     script_dir / "preprocessed_images",
            #     image_path.name
            # )
            
            # Predict
            results = predictor.predict(image_tensor)
            
            # Print results
            print_results(image_path.name, results)
        
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    print(f"\n{'='*70}")
    print("Processing complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
