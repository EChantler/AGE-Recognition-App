import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def organize_ages2_dataset(raw_data_dir, output_dir, train_ratio=0.8):
    """
    Organize ages2 raw dataset into train and test folders based on class labels.
    
    Args:
        raw_data_dir: Path to the raw ages2 directory containing train.csv and Train/
        output_dir: Path where to create the train and test directories
        train_ratio: Ratio for train/test split (default 0.8 for 80/20 split)
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
    
    # Copy train images
    print("\nCopying training images...")
    copied_count = 0
    missing_count = 0
    for idx, row in train_df.iterrows():
        image_name = row['ID']
        class_name = row['Class']
        src_path = os.path.join(images_dir, image_name)
        dst_path = os.path.join(train_output_dir, class_name, image_name)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            print(f"Warning: Image not found: {src_path}")
            missing_count += 1
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} training images...")
    
    print(f"Copied {copied_count} training images, {missing_count} missing")
    
    # Copy test images
    print("\nCopying test images...")
    copied_count = 0
    missing_count = 0
    for idx, row in test_df.iterrows():
        image_name = row['ID']
        class_name = row['Class']
        src_path = os.path.join(images_dir, image_name)
        dst_path = os.path.join(test_output_dir, class_name, image_name)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            print(f"Warning: Image not found: {src_path}")
            missing_count += 1
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} test images...")
    
    print(f"Copied {copied_count} test images, {missing_count} missing")
    
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


if __name__ == "__main__":
    # Define paths
    raw_data_path = os.path.join(os.path.dirname(__file__), 'data', 'ages2_raw')
    output_path = os.path.join(os.path.dirname(__file__), 'data', 'ages2')
    
    # Run the organization function
    print("Starting ages2 dataset organization...")
    print(f"Raw data path: {raw_data_path}")
    print(f"Output path: {output_path}")
    print()
    
    organize_ages2_dataset(raw_data_path, output_path, train_ratio=0.8)
    
    print("\n" + "="*60)
    print("Dataset organization completed successfully!")
    print("="*60)
