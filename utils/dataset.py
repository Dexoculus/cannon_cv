import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob

class AugmentedImageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading augmented images structured by class folders.
    """
    def __init__(self, root_dir, transform=None, image_exts=(".jpg", ".png", ".jpeg")):
        """
        Args:
            root_dir (str): Path to the root directory containing class subfolders
            transform (callable, optional): Optional transform to be applied on a sample.
            image_exts (tuple): Tuple of valid image file extensions.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_exts = image_exts
        self.samples = [] # List to store (image_path, class_index) tuples
        self.classes = [] # List of class names
        self.class_to_idx = {} # Dictionary mapping class name to index

        # Ensure consistent ordering by sorting class names
        class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if not class_names:
            raise FileNotFoundError(f"No class subdirectories found in {root_dir}")

        self.classes = class_names
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        print(f"Found classes: {self.classes}")
        print(f"Class to index mapping: {self.class_to_idx}")

        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(root_dir, class_name)

            # Use glob to find images with specified extensions within the class directory
            for ext in self.image_exts:
                # Using recursive=False (default) as structure is root/class/image.jpg
                image_paths = glob.glob(os.path.join(class_dir, f"*{ext}"))
                for img_path in image_paths:
                    self.samples.append((img_path, class_idx))

        if not self.samples:
             print(f"Warning: No images found in {root_dir} with extensions {image_exts}")

        print(f"Total number of samples found: {len(self.samples)}")


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is the transformed image tensor,
                   and label is the integer class label.
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of bounds for dataset with size {len(self.samples)}")

        # Get the image path and corresponding label index
        img_path, label = self.samples[idx]

        # Load the image using Pillow (recommended for torchvision transforms)
        try:
            # Open in RGB mode to handle grayscale images as well, if needed convert later
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise IOError(f"Could not read image file {img_path}") from e

        # Apply transformations if they exist
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_classes(self):
        """Returns the list of class names."""
        return self.classes

    def get_class_to_idx(self):
        """Returns the dictionary mapping class names to indices."""
        return self.class_to_idx
