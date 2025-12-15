"""
Predictor script for Age, Gender, and Expression recognition using PyTorch models.
Uses MediaPipe for face detection and processes images from the samples folder.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import mediapipe as mp
from typing import Optional, Tuple, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from face_binary_net import (
    MobilenetBinaryNet,
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
                print(f"  → Preprocessed image saved: {output_file.name}")
            else:
                print(f"  ✗ Failed to save preprocessed image")
        except Exception as e:
            print(f"  ✗ Error saving preprocessed image: {str(e)}")


class FaceDetector:
    """Wraps MediaPipe FaceDetector for face detection."""
    
    def __init__(self):
        """Initialize MediaPipe FaceDetector."""
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.5
        )
    
    def detect_faces(self, image: np.ndarray) -> list:
        """
        Detect faces in image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            List of detection results
        """
        # MediaPipe expects RGB
        results = self.detector.process(image)
        return results.detections if results.detections else []
    
    def extract_face_region(
        self, 
        image: np.ndarray, 
        detection: Any,
        target_size: int = 224,
        padding_ratio: float = 0.1
    ) -> Optional[np.ndarray]:
        """
        Extract face region from image based on detection.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            detection: MediaPipe detection object
            target_size: Target size for output image (default 224)
            padding_ratio: Padding ratio for bounding box (default 0.1 = 10%)
            
        Returns:
            Cropped and resized face image (target_size, target_size, 3) or None
        """
        h, w, _ = image.shape
        
        # Get bounding box
        bbox = detection.location_data.relative_bounding_box
        
        # Convert to pixel coordinates
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bbox_w = int(bbox.width * w)
        bbox_h = int(bbox.height * h)
        
        # Add padding (10% on all sides)
        padding_x = int(bbox_w * padding_ratio)
        padding_y = int(bbox_h * padding_ratio)
        
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        bbox_w = min(w - x, bbox_w + 2 * padding_x)
        bbox_h = min(h - y, bbox_h + 2 * padding_y)
        
        # Extract face region
        face_region = image[y:y + bbox_h, x:x + bbox_w]
        
        if face_region.size == 0:
            return None
        
        # Resize to target size
        face_resized = cv2.resize(face_region, (target_size, target_size))
        
        return face_resized


class ModelPredictor:
    """Handles model predictions for face classification."""
    
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
        self.face_model = self._load_model(
            MobilenetBinaryNet, 
            self.models_dir / "face_binary.pth"
        )
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
        self.face_labels = ["Not Face", "Face"]
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
            # Face detection
            face_logits = self.face_model(image_tensor)
            face_probs = F.softmax(face_logits, dim=1)
            face_pred_idx = face_probs.argmax(dim=1).item()
            face_confidence = face_probs[0, face_pred_idx].item()
            
            face_result = {
                "label": self.face_labels[face_pred_idx],
                "confidence": face_confidence,
                "probabilities": {
                    "notFace": face_probs[0, 0].item(),
                    "face": face_probs[0, 1].item(),
                }
            }
            
            # Only proceed with other classifications if face detected
            results = {"face": face_result}
            
            if face_pred_idx == 1:  # Face detected
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
    
    # Face detection
    face_result = results["face"]
    print(f"\nFace Detection: {face_result['label']}")
    print(f"  Confidence: {face_result['confidence']:.2%}")
    print(f"  Not Face: {face_result['probabilities']['notFace']:.2%}")
    print(f"  Face: {face_result['probabilities']['face']:.2%}")
    
    # Other classifications (only if face detected)
    if face_result['label'] == "Face":
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
    else:
        print("\n  No face detected - skipping age, gender, and expression classification.")


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
    print("Initializing face detector...")
    face_detector = FaceDetector()
    
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
            
            # Detect faces
            detections = face_detector.detect_faces(image_rgb)
            
            if not detections:
                print(f"\n{'='*70}")
                print(f"Image: {image_path.name}")
                print(f"{'='*70}")
                print("No faces detected in image.")
                continue
            
            # Process each detected face
            for face_idx, detection in enumerate(detections):
                # Extract face region
                face_image = face_detector.extract_face_region(
                    image_rgb, 
                    detection,
                    target_size=preprocessor.INPUT_SIZE
                )
                
                if face_image is None:
                    print(f"Error: Could not extract face region from {image_path.name}")
                    continue
                
                # Preprocess
                image_tensor = preprocessor.preprocess(face_image)
                
                # Display preprocessed image
                preprocessor.denormalize_and_display(
                    image_tensor,
                    script_dir / "preprocessed_images",
                    image_path.name
                )
                
                # Predict
                results = predictor.predict(image_tensor)
                
                # Print results
                if len(detections) > 1:
                    print_results(f"{image_path.name} (Face {face_idx + 1})", results)
                else:
                    print_results(image_path.name, results)
        
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    print(f"\n{'='*70}")
    print("Processing complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
