from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch
import os

class SimpleFaceDataset(Dataset):
    def __init__(self, root_dir=None, samples=None, transform=None):
        """
        If root_dir is provided, it will scan for:
          root_dir/face/*.jpg
          root_dir/not_face/*.jpg
        Alternatively, pass an explicit `samples` list of dicts with keys: {path, label}.
        Provide a `transform` to control augmentation (train vs val).
        """
        if samples is not None:
            self.samples = samples
        else:
            assert root_dir is not None, "Either root_dir or samples must be provided"
            self.samples = []
            for label_name, label_idx in [("face", 1), ("not_face", 0)]:
                folder = os.path.join(root_dir, label_name)
                for fname in os.listdir(folder):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        self.samples.append(
                            {
                                "path": os.path.join(folder, fname),
                                "label": label_idx,
                            }
                        )

        # Default transform (no augmentation). Pass a custom transform for training augmentation.
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(s["label"], dtype=torch.long)
        return img, label

class AgesDataset(Dataset):
    def __init__(self, root_dir=None, samples=None, transform=None):
        """
        Age classification dataset.
        If root_dir is provided, it will scan for subdirectories:
          root_dir/18-20/*.jpg
          root_dir/21-30/*.jpg
          root_dir/31-40/*.jpg
          root_dir/41-50/*.jpg
          root_dir/51-60/*.jpg
        Alternatively, pass an explicit `samples` list of dicts with keys: {path, label}.
        Provide a `transform` to control augmentation (train vs val).
        """
        if samples is not None:
            self.samples = samples
        else:
            assert root_dir is not None, "Either root_dir or samples must be provided"
            self.samples = []
            age_ranges = [
                ("18-20", 0),
                ("21-30", 1),
                ("31-40", 2),
                ("41-50", 3),
                ("51-60", 4),
            ]
            for label_name, label_idx in age_ranges:
                folder = os.path.join(root_dir, label_name)
                if not os.path.exists(folder):
                    continue
                for fname in os.listdir(folder):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        self.samples.append(
                            {
                                "path": os.path.join(folder, fname),
                                "label": label_idx,
                            }
                        )

        # Default transform (no augmentation). Pass a custom transform for training augmentation.
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(s["label"], dtype=torch.long)
        return img, label


class GendersDataset(Dataset):
    def __init__(self, root_dir=None, samples=None, transform=None):
        """
        Gender classification dataset.
        If root_dir is provided, it will scan for subdirectories:
          root_dir/female/*.jpg
          root_dir/male/*.jpg
        Alternatively, pass an explicit `samples` list of dicts with keys: {path, label}.
        Provide a `transform` to control augmentation (train vs val).
        """
        if samples is not None:
            self.samples = samples
        else:
            assert root_dir is not None, "Either root_dir or samples must be provided"
            self.samples = []
            for label_name, label_idx in [("female", 0), ("male", 1)]:
                folder = os.path.join(root_dir, label_name)
                if not os.path.exists(folder):
                    continue
                for fname in os.listdir(folder):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        self.samples.append(
                            {
                                "path": os.path.join(folder, fname),
                                "label": label_idx,
                            }
                        )

        # Default transform (no augmentation). Pass a custom transform for training augmentation.
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(s["label"], dtype=torch.long)
        return img, label


class ExpressionsDataset(Dataset):
    def __init__(self, root_dir=None, samples=None, transform=None):
        """
        Facial expression classification dataset.
        If root_dir is provided, it will scan for subdirectories:
          root_dir/angry/*.jpg
          root_dir/disgust/*.jpg
          root_dir/fear/*.jpg
          root_dir/happy/*.jpg
          root_dir/neutral/*.jpg
          root_dir/sad/*.jpg
          root_dir/surprise/*.jpg
        Alternatively, pass an explicit `samples` list of dicts with keys: {path, label}.
        Provide a `transform` to control augmentation (train vs val).
        """
        if samples is not None:
            self.samples = samples
        else:
            assert root_dir is not None, "Either root_dir or samples must be provided"
            self.samples = []
            expressions = [
                ("angry", 0),
                ("disgust", 1),
                ("fear", 2),
                ("happy", 3),
                ("neutral", 4),
                ("sad", 5),
                ("surprise", 6),
            ]
            for label_name, label_idx in expressions:
                folder = os.path.join(root_dir, label_name)
                if not os.path.exists(folder):
                    continue
                for fname in os.listdir(folder):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        self.samples.append(
                            {
                                "path": os.path.join(folder, fname),
                                "label": label_idx,
                            }
                        )

        # Default transform (no augmentation). Pass a custom transform for training augmentation.
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(s["label"], dtype=torch.long)
        return img, label