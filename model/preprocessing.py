import os
import shutil
import pandas as pd
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Optional, Any


class FaceExtractor:
    """Extracts face regions using MediaPipe, matching the inference pipeline."""
    
    def __init__(self):
        """Initialize MediaPipe FaceDetector."""
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range (within 2 meters)
            min_detection_confidence=0.5
        )
    
    def extract_face_region(
        self, 
        image: np.ndarray, 
        detection: Any,
        target_size: int = 224,
        padding_ratio: float = 0.1
    ) -> Optional[np.ndarray]:
        """
        Extract face region from image based on detection.
        Matches the preprocessing logic in preprocess.ts and predictor.py
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            detection: MediaPipe detection object
            target_size: Target size for output image (default 224)
            padding_ratio: Padding ratio for bounding box (default 0.1 = 10%)
            
        Returns:
            Cropped and resized face image (target_size, target_size, 3) or None
        """
        h, w, _ = image.shape
        
        # Get bounding box (relative coordinates)
        bbox = detection.location_data.relative_bounding_box
        
        # Convert to pixel coordinates
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bbox_w = int(bbox.width * w)
        bbox_h = int(bbox.height * h)
        
        # Add padding (10% on all sides, matching inference)
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
        
        # Resize to target size (matching inference)
        face_resized = cv2.resize(face_region, (target_size, target_size))
        
        return face_resized
    
    def detect_and_extract_largest_face(
        self, 
        image_path: str,
        target_size: int = 224
    ) -> Optional[np.ndarray]:
        """
        Detect faces and extract the largest one.
        
        Args:
            image_path: Path to input image
            target_size: Target size for output image (default 224)
            
        Returns:
            Cropped and resized face image or None if no face detected
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert BGR to RGB (MediaPipe expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.detector.process(image_rgb)
        
        if not results.detections or len(results.detections) == 0:
            return None
        
        # Find largest face by area
        largest_detection = results.detections[0]
        if len(results.detections) > 1:
            largest_area = 0
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                area = bbox.width * bbox.height
                if area > largest_area:
                    largest_area = area
                    largest_detection = detection
        
        # Extract face region
        face_image = self.extract_face_region(image_rgb, largest_detection, target_size)
        
        return face_image
    
    def close(self):
        """Clean up resources."""
        self.detector.close()


def organize_ages2_dataset(raw_data_dir, output_dir, train_ratio=0.8, extract_faces=True):
    """
    Organize ages2 raw dataset into train and test folders based on class labels.
    Optionally extract face regions using MediaPipe to match inference preprocessing.
    
    Args:
        raw_data_dir: Path to the raw ages2 directory containing train.csv and Train/
        output_dir: Path where to create the train and test directories
        train_ratio: Ratio for train/test split (default 0.8 for 80/20 split)
        extract_faces: Whether to extract face regions using MediaPipe (default True)
    """
    
    # Define paths
    csv_file = os.path.join(raw_data_dir, 'train.csv')
    images_dir = os.path.join(raw_data_dir, 'Train')
    train_output_dir = os.path.join(output_dir, 'train')
    test_output_dir = os.path.join(output_dir, 'test')
    
    # Verify input files exist
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Ensure base output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    print(f"Reading CSV file from: {csv_file}")
    # Read the CSV file
    df = pd.read_csv(csv_file)
    print(f"Total images in CSV: {len(df)}")
    
    # Get unique classes
    classes = df['Class'].unique()
    print(f"Classes found: {classes}")
    
    # Create output directories for train and test with class subdirectories
    for split_dir in [train_output_dir, test_output_dir]:
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            print(f"Created directory: {class_dir}")
    
    # Split data by stratified sampling to maintain class distribution
    train_df, test_df = train_test_split(
        df,
        test_size=1 - train_ratio,
        stratify=df['Class'],
        random_state=42
    )
    
    print(f"\nTrain set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Initialize face extractor if needed
    face_extractor = None
    if extract_faces:
        print("\nInitializing MediaPipe face detector...")
        face_extractor = FaceExtractor()
        print("Face detector ready")
    
    # Copy train images
    print(f"\nCopying training images{' with face extraction' if extract_faces else ''}...")
    copied_count = 0
    missing_count = 0
    no_face_count = 0
    
    for idx, row in train_df.iterrows():
        image_name = row['ID']
        class_name = row['Class']
        src_path = os.path.join(images_dir, image_name)
        dst_path = os.path.join(train_output_dir, class_name, image_name)
        
        if not os.path.exists(src_path):
            print(f"Warning: Image not found: {src_path}")
            missing_count += 1
            continue
        
        if extract_faces:
            # Extract face region
            face_image = face_extractor.detect_and_extract_largest_face(src_path)
            
            if face_image is not None:
                # Convert RGB back to BGR for OpenCV
                face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                # Save the face-extracted image
                cv2.imwrite(dst_path, face_image_bgr)
                copied_count += 1
            else:
                # No face detected, skip this image
                no_face_count += 1
        else:
            # Just copy the image without face extraction
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} training images...")
    
    print(f"Copied {copied_count} training images")
    if missing_count > 0:
        print(f"  {missing_count} images were missing")
    if extract_faces and no_face_count > 0:
        print(f"  {no_face_count} images skipped (no face detected)")
    
    # Copy test images
    print(f"\nCopying test images{' with face extraction' if extract_faces else ''}...")
    copied_count = 0
    missing_count = 0
    no_face_count = 0
    
    for idx, row in test_df.iterrows():
        image_name = row['ID']
        class_name = row['Class']
        src_path = os.path.join(images_dir, image_name)
        dst_path = os.path.join(test_output_dir, class_name, image_name)
        
        if not os.path.exists(src_path):
            print(f"Warning: Image not found: {src_path}")
            missing_count += 1
            continue
        
        if extract_faces:
            # Extract face region
            face_image = face_extractor.detect_and_extract_largest_face(src_path)
            
            if face_image is not None:
                # Convert RGB back to BGR for OpenCV
                face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                # Save the face-extracted image
                cv2.imwrite(dst_path, face_image_bgr)
                copied_count += 1
            else:
                # No face detected, skip this image
                no_face_count += 1
        else:
            # Just copy the image without face extraction
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} test images...")
    
    print(f"Copied {copied_count} test images")
    if missing_count > 0:
        print(f"  {missing_count} images were missing")
    if extract_faces and no_face_count > 0:
        print(f"  {no_face_count} images skipped (no face detected)")
    
    # Clean up face extractor
    if face_extractor:
        face_extractor.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for split_name, split_dir in [("Train", train_output_dir), ("Test", test_output_dir)]:
        print(f"\n{split_name} set:")
        for class_name in sorted(classes):
            class_dir = os.path.join(split_dir, class_name)
            num_images = len(os.listdir(class_dir))
            print(f"  {class_name}: {num_images} images")
        
        # Print total
        total = sum(len(os.listdir(os.path.join(split_dir, class_name))) for class_name in classes)
        print(f"  Total: {total} images")


def preprocess_classification_dataset_faces(
    input_root: str,
    output_root: str,
    target_size: int = 224,
    padding_ratio: float = 0.1,
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
):
    """
    Apply face extraction to a classification dataset that follows:
      input_root/train/<class>/*.jpg and input_root/test/<class>/*.jpg
    or a flat structure input_root/<class>/*.jpg

    Outputs to mirrored structure under output_root, creating directories as needed.

    Args:
        input_root: Source dataset root directory
        output_root: Destination root for preprocessed dataset
        target_size: Output image size (square)
        padding_ratio: Padding to add around detected face bbox
        valid_exts: Accepted image extensions
    """
    print("\nStarting face preprocessing for classification dataset...")
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")

    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Input root not found: {input_root}")

    os.makedirs(output_root, exist_ok=True)

    # Determine if dataset has explicit splits
    has_splits = all(os.path.isdir(os.path.join(input_root, s)) for s in ["train", "test"]) or \
                 any(os.path.isdir(os.path.join(input_root, s)) for s in ["train", "test"])  # at least one

    splits = ["train", "test"] if has_splits else [None]

    extractor = FaceExtractor()

    total_processed = 0
    total_copied = 0
    total_skipped = 0

    try:
        for split in splits:
            split_input = os.path.join(input_root, split) if split else input_root
            split_output = os.path.join(output_root, split) if split else output_root
            os.makedirs(split_output, exist_ok=True)

            # Classes are immediate subdirectories
            class_names = [d for d in os.listdir(split_input) if os.path.isdir(os.path.join(split_input, d))]
            if not class_names:
                print(f"No class folders found under {split_input}; skipping.")
                continue

            print(f"\nProcessing split: {split or 'all'} (classes: {len(class_names)})")

            for class_name in sorted(class_names):
                class_input_dir = os.path.join(split_input, class_name)
                class_output_dir = os.path.join(split_output, class_name)
                os.makedirs(class_output_dir, exist_ok=True)

                images = [f for f in os.listdir(class_input_dir) if f.lower().endswith(valid_exts)]
                copied = 0
                skipped = 0

                for i, fname in enumerate(images):
                    src = os.path.join(class_input_dir, fname)
                    dst = os.path.join(class_output_dir, fname)

                    face_image = extractor.detect_and_extract_largest_face(src, target_size=target_size)
                    if face_image is None:
                        skipped += 1
                        continue

                    # Convert RGB->BGR before saving with OpenCV
                    face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(dst, face_bgr)
                    copied += 1

                    if (i + 1) % 1000 == 0:
                        print(f"  [{class_name}] Processed {i + 1}/{len(images)} images...")

                print(f"  Class '{class_name}': saved={copied}, skipped(no-face)={skipped}, total={len(images)}")
                total_copied += copied
                total_skipped += skipped
                total_processed += len(images)

        print("\n" + "-"*60)
        print("PREPROCESSING SUMMARY")
        print("-"*60)
        print(f"Total images scanned: {total_processed}")
        print(f"Saved (face-extracted): {total_copied}")
        print(f"Skipped (no-face): {total_skipped}")
    finally:
        extractor.close()


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)

    # 1) Ages2 raw -> organized with face extraction
    raw_data_path = os.path.join(base_dir, 'data', 'ages2_raw')
    ages2_output_path = os.path.join(base_dir, 'data', 'ages2_preprocessed')

    print("Starting ages2 dataset organization with face extraction...")
    print(f"Raw data path: {raw_data_path}")
    print(f"Output path: {ages2_output_path}")
    print("Face extraction: ENABLED (matching inference preprocessing)")
    print()
    organize_ages2_dataset(raw_data_path, ages2_output_path, train_ratio=0.8, extract_faces=True)

    # 2) Expressions dataset face preprocessing (train/test/<class>)
    expressions_input = os.path.join(base_dir, 'data', 'expressions')
    expressions_output = os.path.join(base_dir, 'data', 'expressions_preprocessed')
    print("\nProcessing expressions dataset with face extraction...")
    preprocess_classification_dataset_faces(expressions_input, expressions_output, target_size=224)

    # 3) Gender dataset face preprocessing (train/test/<class>)
    gender_input = os.path.join(base_dir, 'data', 'gender')
    gender_output = os.path.join(base_dir, 'data', 'gender_preprocessed')
    print("\nProcessing gender dataset with face extraction...")
    preprocess_classification_dataset_faces(gender_input, gender_output, target_size=224)

    print("\n" + "="*60)
    print("All dataset preprocessing completed successfully!")
    print("="*60)
