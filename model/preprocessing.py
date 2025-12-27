import os
import shutil
import pandas as pd
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Optional, Any
from collections import defaultdict


def count_images_by_class(root_dir, split_name=None):
    """
    Count images in a directory structure organized by class.
    
    Args:
        root_dir: Root directory containing class subdirectories
        split_name: Optional split name (train/test) to navigate to
        
    Returns:
        Dictionary mapping class names to image counts
    """
    if split_name:
        root_dir = os.path.join(root_dir, split_name)
    
    if not os.path.exists(root_dir):
        return {}
    
    class_counts = defaultdict(int)
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    
    class_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(valid_exts)]
        class_counts[class_name] = len(images)
    
    return dict(class_counts)


def print_comparison_table(dataset_name, before_train, before_test, after_train, after_test):
    """
    Print a comparison table showing before/after preprocessing counts.
    
    Args:
        dataset_name: Name of the dataset
        before_train: Dict of class -> count before preprocessing (train)
        before_test: Dict of class -> count before preprocessing (test)
        after_train: Dict of class -> count after preprocessing (train)
        after_test: Dict of class -> count after preprocessing (test)
    """
    print("\n" + "="*100)
    print(f"PREPROCESSING COMPARISON: {dataset_name.upper()}")
    print("="*100)
    
    # Get all classes from both before and after
    all_classes = sorted(set(list(before_train.keys()) + list(after_train.keys()) + 
                            list(before_test.keys()) + list(after_test.keys())))
    
    # Print header
    print(f"{'Class':<20} {'TRAIN Before':<15} {'TRAIN After':<15} {'Change':<15} {'TEST Before':<15} {'TEST After':<15} {'Change':<15}")
    print("-" * 100)
    
    total_before_train = 0
    total_after_train = 0
    total_before_test = 0
    total_after_test = 0
    
    # Print each class
    for class_name in all_classes:
        before_t = before_train.get(class_name, 0)
        after_t = after_train.get(class_name, 0)
        before_test_c = before_test.get(class_name, 0)
        after_test_c = after_test.get(class_name, 0)
        
        change_train = after_t - before_t
        change_test = after_test_c - before_test_c
        
        change_train_str = f"{change_train:+d}" if change_train != 0 else "0"
        change_test_str = f"{change_test:+d}" if change_test != 0 else "0"
        
        print(f"{class_name:<20} {before_t:<15} {after_t:<15} {change_train_str:<15} {before_test_c:<15} {after_test_c:<15} {change_test_str:<15}")
        
        total_before_train += before_t
        total_after_train += after_t
        total_before_test += before_test_c
        total_after_test += after_test_c
    
    # Print total row
    print("-" * 100)
    total_change_train = total_after_train - total_before_train
    total_change_test = total_after_test - total_before_test
    total_change_train_str = f"{total_change_train:+d}" if total_change_train != 0 else "0"
    total_change_test_str = f"{total_change_test:+d}" if total_change_test != 0 else "0"
    
    print(f"{'TOTAL':<20} {total_before_train:<15} {total_after_train:<15} {total_change_train_str:<15} {total_before_test:<15} {total_after_test:<15} {total_change_test_str:<15}")
    print("="*100)


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


def organize_flat_dataset_with_split(
    input_root: str,
    output_root: str,
    train_ratio: float = 0.8,
    target_size: int = 224,
    extract_faces: bool = True,
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
):
    """
    Organize a flat dataset with class directories (no train/test split) into train/test splits.
    Applies face extraction to match inference preprocessing.
    
    Input structure:
      input_root/<class>/*.jpg
    
    Output structure:
      output_root/train/<class>/*.jpg
      output_root/test/<class>/*.jpg
    
    Args:
        input_root: Source dataset root with class subdirectories
        output_root: Destination root for organized dataset
        train_ratio: Ratio for train/test split (default 0.8 for 80/20 split)
        target_size: Output image size (square)
        extract_faces: Whether to extract face regions using MediaPipe (default True)
        valid_exts: Accepted image extensions
    """
    print(f"\nOrganizing flat dataset with {int(train_ratio*100)}/{int((1-train_ratio)*100)} train/test split...")
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    
    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Input root not found: {input_root}")
    
    os.makedirs(output_root, exist_ok=True)
    
    # Get class directories
    class_names = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    if not class_names:
        raise FileNotFoundError(f"No class directories found in {input_root}")
    
    print(f"Found {len(class_names)} classes: {', '.join(sorted(class_names))}")
    
    # Create output train/test directories
    train_output_dir = os.path.join(output_root, 'train')
    test_output_dir = os.path.join(output_root, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    for class_name in class_names:
        os.makedirs(os.path.join(train_output_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_output_dir, class_name), exist_ok=True)
    
    # Initialize face extractor if needed
    extractor = None
    if extract_faces:
        print("Initializing MediaPipe face detector...")
        extractor = FaceExtractor()
    
    total_processed = 0
    total_copied = 0
    total_skipped = 0
    
    try:
        for class_name in sorted(class_names):
            class_input_dir = os.path.join(input_root, class_name)
            images = [f for f in os.listdir(class_input_dir) if f.lower().endswith(valid_exts)]
            
            print(f"\nProcessing class '{class_name}': {len(images)} images")
            
            # Split images into train/test
            import random
            random.seed(42)
            random.shuffle(images)
            split_idx = int(len(images) * train_ratio)
            train_images = images[:split_idx]
            test_images = images[split_idx:]
            
            print(f"  Train: {len(train_images)}, Test: {len(test_images)}")
            
            # Process train images
            copied = 0
            skipped = 0
            for i, fname in enumerate(train_images):
                src = os.path.join(class_input_dir, fname)
                dst = os.path.join(train_output_dir, class_name, fname)
                
                if extract_faces:
                    face_image = extractor.detect_and_extract_largest_face(src, target_size=target_size)
                    if face_image is None:
                        skipped += 1
                        continue
                    face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(dst, face_bgr)
                else:
                    shutil.copy2(src, dst)
                
                copied += 1
                
                if (i + 1) % 1000 == 0:
                    print(f"    Processed {i + 1}/{len(train_images)} train images...")
            
            print(f"  Train saved: {copied}, skipped: {skipped}")
            total_copied += copied
            total_skipped += skipped
            total_processed += len(train_images)
            
            # Process test images
            copied = 0
            skipped = 0
            for i, fname in enumerate(test_images):
                src = os.path.join(class_input_dir, fname)
                dst = os.path.join(test_output_dir, class_name, fname)
                
                if extract_faces:
                    face_image = extractor.detect_and_extract_largest_face(src, target_size=target_size)
                    if face_image is None:
                        skipped += 1
                        continue
                    face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(dst, face_bgr)
                else:
                    shutil.copy2(src, dst)
                
                copied += 1
                
                if (i + 1) % 1000 == 0:
                    print(f"    Processed {i + 1}/{len(test_images)} test images...")
            
            print(f"  Test saved: {copied}, skipped: {skipped}")
            total_copied += copied
            total_skipped += skipped
            total_processed += len(test_images)
        
        print("\n" + "-"*60)
        print("SPLIT PREPROCESSING SUMMARY")
        print("-"*60)
        print(f"Total images scanned: {total_processed}")
        print(f"Saved (face-extracted): {total_copied}")
        print(f"Skipped (no-face): {total_skipped}")
    finally:
        if extractor:
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
    
    # Collect before counts for ages2
    images_dir = os.path.join(raw_data_path, 'Train')
    csv_file = os.path.join(raw_data_path, 'train.csv')
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Class'], random_state=42)
        ages2_before_train = dict(train_df['Class'].value_counts())
        ages2_before_test = dict(test_df['Class'].value_counts())
    else:
        ages2_before_train = {}
        ages2_before_test = {}
    
    organize_ages2_dataset(raw_data_path, ages2_output_path, train_ratio=0.8, extract_faces=True)
    
    # Collect after counts for ages2
    ages2_after_train = count_images_by_class(ages2_output_path, 'train')
    ages2_after_test = count_images_by_class(ages2_output_path, 'test')
    
    # Print ages2 comparison table
    print_comparison_table("Ages2", ages2_before_train, ages2_before_test, ages2_after_train, ages2_after_test)

    # 2) Expressions dataset face preprocessing (train/test/<class>)
    expressions_input = os.path.join(base_dir, 'data', 'expressions2')
    expressions_output = os.path.join(base_dir, 'data', 'expressions2_preprocessed')
    print("\nProcessing expressions dataset with face extraction...")
    
    # Collect before counts for expressions
    expressions_before_train = count_images_by_class(expressions_input, 'train')
    expressions_before_test = count_images_by_class(expressions_input, 'test')
    
    preprocess_classification_dataset_faces(expressions_input, expressions_output, target_size=224)
    
    # Collect after counts for expressions
    expressions_after_train = count_images_by_class(expressions_output, 'train')
    expressions_after_test = count_images_by_class(expressions_output, 'test')
    
    # Print expressions comparison table
    print_comparison_table("Expressions", expressions_before_train, expressions_before_test, expressions_after_train, expressions_after_test)

    # 3) Gender2 dataset - organize with 80/20 train/test split and face extraction
    gender_input = os.path.join(base_dir, 'data', 'gender2')
    gender_output = os.path.join(base_dir, 'data', 'gender2_preprocessed')
    print("\nProcessing gender2 dataset with 80/20 split and face extraction...")
    
    # Collect before counts for gender (flat structure)
    gender_before_train = {}
    gender_before_test = {}
    class_names = [d for d in os.listdir(gender_input) if os.path.isdir(os.path.join(gender_input, d))]
    for class_name in class_names:
        class_dir = os.path.join(gender_input, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        gender_before_train[class_name] = len(images)  # Will be split in processing
    
    organize_flat_dataset_with_split(gender_input, gender_output, train_ratio=0.8, extract_faces=True)
    
    # Collect after counts for gender
    gender_after_train = count_images_by_class(gender_output, 'train')
    gender_after_test = count_images_by_class(gender_output, 'test')
    
    # For before counts with split applied (approximate)
    gender_before_train_split = {}
    gender_before_test_split = {}
    for class_name in class_names:
        total = gender_before_train.get(class_name, 0)
        gender_before_train_split[class_name] = int(total * 0.8)
        gender_before_test_split[class_name] = int(total * 0.2)
    
    # Print gender comparison table
    print_comparison_table("Gender2", gender_before_train_split, gender_before_test_split, gender_after_train, gender_after_test)

    print("\n" + "="*100)
    print("All dataset preprocessing completed successfully!")
    print("="*100)
