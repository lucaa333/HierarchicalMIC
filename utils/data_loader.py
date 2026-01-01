import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy import ndimage
from medmnist import (
    OrganMNIST3D,
    NoduleMNIST3D,
    AdrenalMNIST3D,
    FractureMNIST3D,
    FractureMNIST3D,
    VesselMNIST3D,
    INFO
)

# Import augmentation config
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG, MODEL_CONFIG, AUGMENTATION_CONFIG, DATASET_INFO_KEYS
from config import DEFAULT_MERGED_DATASETS


class Augmentation3D:
    """
    3D Data augmentation for volumetric medical images.
    Applies flipping, rotation, cropping, shearing, noise, and brightness adjustments.
    """
    
    def __init__(self, config=None):
        self.config = config or AUGMENTATION_CONFIG
        self.enabled = self.config.get('enabled', True)
    
    def __call__(self, volume):
        """
        Apply augmentation transforms to a 3D volume.
        
        Args:
            volume: numpy array of shape (C, D, H, W) or (D, H, W)
        
        Returns:
            Augmented volume with same shape
        """
        if not self.enabled:
            return volume
        
        # Handle channel dimension
        has_channel = volume.ndim == 4
        if has_channel:
            # Process each channel, but typically C=1 for medical images
            result = np.stack([self._augment_volume(volume[c]) for c in range(volume.shape[0])])
        else:
            result = self._augment_volume(volume)
        
        return result
    
    def _augment_volume(self, vol):
        """Apply augmentations to a single 3D volume (D, H, W)."""
        vol = vol.copy()
        
        # 1. Random flipping
        # We AVOID axis 2 (width) to preserve Left/Right anatomical laterality
        flip_prob = self.config.get('flip_prob', 0.5)
        if np.random.random() < flip_prob:
            # Random axis flip (0=depth, 1=height)
            axis = np.random.choice([0, 1])
            vol = np.flip(vol, axis=axis).copy()
        
        # 2. Random rotation
        rot_range = self.config.get('rotation_range', (-15, 15))
        angle = np.random.uniform(rot_range[0], rot_range[1])
        # Rotate in the axial plane (height-width)
        vol = ndimage.rotate(vol, angle, axes=(1, 2), reshape=False, order=1, mode='nearest')
        
        # 3. Random cropping and resizing
        crop_scale = self.config.get('crop_scale', (0.85, 1.0))
        crop_ratio = self.config.get('crop_ratio', (0.9, 1.1))
        vol = self._random_resized_crop(vol, crop_scale, crop_ratio)
        
        # 4. Random shearing
        shear_range = self.config.get('shear_range', (-10, 10))
        shear_angle = np.random.uniform(shear_range[0], shear_range[1])
        vol = self._apply_shear(vol, shear_angle)
        
        # 5. Gaussian noise
        noise_std = self.config.get('gaussian_noise_std', 0.02)
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, vol.shape)
            vol = vol + noise
        
        # 6. Brightness adjustment
        brightness_range = self.config.get('brightness_range', (0.9, 1.1))
        brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
        vol = vol * brightness_factor
        
        # Clip values to valid range
        vol = np.clip(vol, 0, 1)
        
        return vol
    
    def _random_resized_crop(self, vol, scale_range, ratio_range):
        """Apply random resized crop to volume."""
        D, H, W = vol.shape
        
        # Calculate crop size
        scale = np.random.uniform(scale_range[0], scale_range[1])
        ratio = np.random.uniform(ratio_range[0], ratio_range[1])
        
        new_h = int(H * scale)
        new_w = int(W * scale * ratio)
        new_d = int(D * scale)
        
        # Ensure minimum size
        new_h = max(new_h, 4)
        new_w = max(new_w, 4)
        new_d = max(new_d, 4)
        
        # Random crop position
        d_start = np.random.randint(0, max(1, D - new_d + 1))
        h_start = np.random.randint(0, max(1, H - new_h + 1))
        w_start = np.random.randint(0, max(1, W - new_w + 1))
        
        # Crop
        cropped = vol[d_start:d_start+new_d, h_start:h_start+new_h, w_start:w_start+new_w]
        
        # Resize back to original shape using zoom
        zoom_factors = (D / cropped.shape[0], H / cropped.shape[1], W / cropped.shape[2])
        resized = ndimage.zoom(cropped, zoom_factors, order=1)
        
        # Ensure exact shape match
        if resized.shape != (D, H, W):
            resized = self._resize_to_shape(resized, (D, H, W))
        
        return resized
    
    def _resize_to_shape(self, vol, target_shape):
        """Resize volume to exact target shape."""
        zoom_factors = tuple(t / s for t, s in zip(target_shape, vol.shape))
        return ndimage.zoom(vol, zoom_factors, order=1)
    
    def _apply_shear(self, vol, angle_degrees):
        """Apply shear transformation along x-axis."""
        # Convert to radians
        shear = np.tan(np.radians(angle_degrees))
        
        # Create affine transformation matrix for shearing
        # Shear in the H-W plane
        D, H, W = vol.shape
        
        # Transformation matrix for shearing
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, shear, 0],
            [0, 0, 1, 0]
        ])
        
        # Apply affine transform
        # Center the shear around image center
        offset = np.array([0, -shear * W / 2, 0])
        
        result = ndimage.affine_transform(vol, matrix[:, :3], offset=offset, order=1, mode='nearest')
        
        return result


# Dataset mapping for anatomical regions
REGION_DATASET_MAPPING = {
    'organ': OrganMNIST3D,      # Multi-organ (11 classes)
    'nodule': NoduleMNIST3D,    # Chest (2 classes: benign/malignant)
    'adrenal': AdrenalMNIST3D,  # Abdomen (2 classes)
    'fracture': FractureMNIST3D, # Bone (3 classes)
    'vessel': VesselMNIST3D     # Brain vessels (2 classes)
}

# ============================================================================
# Fine-to-Coarse Label Mapping (per paper specification)
# ============================================================================
# Maps each fine-grained label to its coarse anatomical region

# OrganMNIST3D: 11 organ classes -> regions
ORGAN_FINE_TO_COARSE = {
    0: 'abdomen',   # liver
    1: 'abdomen',   # kidney-right
    2: 'abdomen',   # kidney-left
    3: 'abdomen',   # femur-right
    4: 'abdomen',   # femur-left
    5: 'abdomen',   # bladder
    6: 'chest',     # heart
    7: 'chest',     # lung-right
    8: 'chest',     # lung-left
    9: 'abdomen',   # spleen
    10: 'abdomen',  # pancreas
}

# NoduleMNIST3D: 2 nodule classes -> chest
NODULE_FINE_TO_COARSE = {
    0: 'chest',     # benign
    1: 'chest',     # malignant
}

# AdrenalMNIST3D: 2 adrenal classes -> abdomen
ADRENAL_FINE_TO_COARSE = {
    0: 'abdomen',   # normal
    1: 'abdomen',   # hyperplasia
}

# FractureMNIST3D: 3 fracture classes -> chest (rib fractures)
FRACTURE_FINE_TO_COARSE = {
    0: 'chest',     # buckle rib fracture
    1: 'chest',     # nondisplaced rib fracture
    2: 'chest',     # displaced rib fracture
}

# VesselMNIST3D: 2 vessel classes -> brain
VESSEL_FINE_TO_COARSE = {
    0: 'brain',     # vessel
    1: 'brain',     # aneurysm
}

# Master mapping: dataset_name -> fine_label -> coarse_region
FINE_TO_COARSE_MAPPING = {
    'organ': ORGAN_FINE_TO_COARSE,
    'nodule': NODULE_FINE_TO_COARSE,
    'adrenal': ADRENAL_FINE_TO_COARSE,
    'fracture': FRACTURE_FINE_TO_COARSE,
    'vessel': VESSEL_FINE_TO_COARSE
}

# ============================================================================
# Region Fine Class Configuration (per paper Figure 2)
# ============================================================================
# Total fine classes per region for Stage 2 classifiers
REGION_FINE_CLASS_COUNTS = {
    'brain': 2,     # VesselMNIST: aneurysm, vessel
    'chest': 8,     # Organs (heart, lung-right, lung-left) = 3
                    # + NoduleMNIST (benign, malignant) = 2  
                    # + FractureMNIST (buckle, nondisplaced, displaced) = 3
    'abdomen': 10,  # Organs (liver, kidney-R/L, femur-R/L, bladder, pancreas, spleen) = 8
                    # + AdrenalMNIST (normal, hyperplasia) = 2
}


def get_medmnist_dataloaders(
    dataset_name='organ',
    batch_size=32,
    num_workers=4,
    download=True
):
    if dataset_name not in REGION_DATASET_MAPPING:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_class = REGION_DATASET_MAPPING[dataset_name]

    # Load datasets
    train_dataset = dataset_class(split='train', download=download)
    val_dataset = dataset_class(split='val', download=download)
    test_dataset = dataset_class(split='test', download=download)

    # Determine number of classes
    labels = train_dataset.labels
    if labels.ndim > 1 and labels.shape[1] > 1:
        num_classes = labels.shape[1]
    else:
        num_classes = int(labels.max()) + 1

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, num_classes


class HierarchicalMedMNISTDataset(Dataset):
    """
    Hierarchical MedMNIST 3D Dataset with optional data augmentation.
    
    Uses fine-to-coarse label mapping to assign each sample's coarse region
    based on its fine-grained label (e.g., 'heart' -> 'chest', 'liver' -> 'abdomen').
    
    Fine labels are remapped to be region-local (0-indexed within each region)
    to match the expected output size of each region's fine classifier.
    
    Args:
        datasets_config: Dict specifying which datasets to include
        split: 'train', 'val', or 'test'
        augment: Whether to apply data augmentation (default: True for train/val)
        augmentation_config: Optional custom augmentation config
        return_global_labels: If True, returns (img, coarse_label, fine_label, global_fine_label).
                            If False, returns (img, coarse_label, fine_label). Default: False.
    """
    def __init__(self, datasets_config, split='train', augment=None, augmentation_config=None, return_global_labels=False):
        self.split = split
        self.return_global_labels = return_global_labels
        self.samples = []
        self.coarse_labels = []
        self.fine_labels = []  # Region-local fine labels (0-indexed per region)
        self.global_fine_labels = []  # Global fine labels (0-indexed across all regions, for flat classification)
        self.original_fine_labels = []  # Original dataset labels (for reference)
        
        # Set up augmentation (apply to train and val by default, not test)
        if augment is None:
            augment = split in ['train', 'val']
        self.augment = augment
        
        if self.augment:
            self.augmenter = Augmentation3D(augmentation_config)
        else:
            self.augmenter = None

        # Build region index mapping from unique regions across all datasets
        # Using only 3 regions: abdomen, chest, brain (as per paper)
        all_regions = ['abdomen', 'chest', 'brain']
        self.region_to_idx = {region: idx for idx, region in enumerate(all_regions)}
        
        # First pass: Collect all (dataset_name, original_fine_label) pairs per region
        # to build the global-to-local fine label mapping
        region_label_sets = {region: set() for region in all_regions}
        
        for dataset_name in datasets_config:
            if dataset_name not in REGION_DATASET_MAPPING:
                continue
            fine_to_coarse_map = FINE_TO_COARSE_MAPPING[dataset_name]
            for orig_label, coarse_region in fine_to_coarse_map.items():
                # Use tuple (dataset_name, original_label) as unique identifier
                region_label_sets[coarse_region].add((dataset_name, orig_label))
        
        # Build region-local fine label mapping: {region: {(dataset, orig_label): local_idx}}
        self.region_fine_label_map = {}
        self.region_num_classes = {}
        for region in all_regions:
            sorted_labels = sorted(region_label_sets[region])  # Deterministic ordering
            self.region_fine_label_map[region] = {lbl: idx for idx, lbl in enumerate(sorted_labels)}
            self.region_num_classes[region] = len(sorted_labels)
        
        # Build region offsets for global fine label computation
        # Global label = region_offset + local_fine_label
        self.region_offsets = {}
        offset = 0
        for region in all_regions:
            self.region_offsets[region] = offset
            offset += self.region_num_classes[region]

        # Load and combine datasets with proper fine-to-coarse mapping
        for dataset_name in datasets_config:
            if dataset_name not in REGION_DATASET_MAPPING:
                continue
                
            dataset_class = REGION_DATASET_MAPPING[dataset_name]
            dataset = dataset_class(split=split, download=True)
            fine_to_coarse_map = FINE_TO_COARSE_MAPPING[dataset_name]

            for i in range(len(dataset)):
                img, label = dataset[i]
                # Get fine label (handle both scalar and array labels)
                orig_fine_label = int(label.squeeze()) if hasattr(label, 'squeeze') else int(label)
                
                # Map fine label to coarse region using the mapping
                coarse_region = fine_to_coarse_map.get(orig_fine_label, 'abdomen')
                coarse_idx = self.region_to_idx[coarse_region]
                
                # Get region-local fine label (0-indexed within region)
                local_fine_label = self.region_fine_label_map[coarse_region][(dataset_name, orig_fine_label)]
                
                # Compute global fine label (0-indexed across all regions)
                global_fine_label = self.region_offsets[coarse_region] + local_fine_label
                
                self.samples.append(img)
                self.coarse_labels.append(coarse_idx)
                self.fine_labels.append(local_fine_label)
                self.global_fine_labels.append(global_fine_label)
                self.original_fine_labels.append(orig_fine_label)

        self.samples = np.array(self.samples)
        self.coarse_labels = np.array(self.coarse_labels)
        self.fine_labels = np.array(self.fine_labels)
        self.global_fine_labels = np.array(self.global_fine_labels)
        self.original_fine_labels = np.array(self.original_fine_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx].copy()
        
        # Normalize to [0, 1] range first (before augmentation)
        if img.max() > 1:
            img = img / 255.0
        
        # Apply augmentation if enabled
        if self.augmenter is not None:
            img = self.augmenter(img)
        
        img = torch.from_numpy(img).float()
        coarse_label = torch.tensor(self.coarse_labels[idx]).long()
        fine_label = torch.tensor(self.fine_labels[idx]).long().squeeze()
        
        if self.return_global_labels:
            global_fine_label = torch.tensor(self.global_fine_labels[idx]).long().squeeze()
            return img, coarse_label, fine_label, global_fine_label
            
        return img, coarse_label, fine_label


def     create_hierarchical_dataset(
    datasets_to_include=None,
    batch_size=32,
    num_workers=4,
    return_global_labels=False
):
    """
    Create merged hierarchical dataset from multiple 3D MedMNIST datasets.
    
    Args:
        datasets_to_include: List of dataset names. Default: all 5 datasets
                            ['organ', 'nodule', 'adrenal', 'fracture', 'vessel']
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for data loading
        return_global_labels: Whether to include global fine labels in returned batches.
        
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        dataset_info: Dict with metadata about the merged dataset
    """
    if datasets_to_include is None:
        # Default: Use all 5 3D MedMNIST datasets for hierarchical training
        datasets_to_include = ['organ', 'nodule', 'adrenal', 'fracture', 'vessel']

    datasets_config = {name: True for name in datasets_to_include}

    train_dataset = HierarchicalMedMNISTDataset(datasets_config, split='train', return_global_labels=return_global_labels)
    val_dataset = HierarchicalMedMNISTDataset(datasets_config, split='val', return_global_labels=return_global_labels)
    test_dataset = HierarchicalMedMNISTDataset(datasets_config, split='test', return_global_labels=return_global_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Compute dataset info for convenience
    num_coarse_classes = len(train_dataset.region_to_idx)
    # Total fine classes = sum of all region-specific fine classes
    num_fine_classes = sum(train_dataset.region_num_classes.values())
    
    dataset_info = {
        'num_coarse_classes': num_coarse_classes,
        'num_fine_classes': num_fine_classes,
        'region_to_idx': train_dataset.region_to_idx,
        'idx_to_region': {v: k for k, v in train_dataset.region_to_idx.items()},
        'datasets_included': datasets_to_include,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        # Region-specific fine class counts for model initialization
        'region_num_classes': train_dataset.region_num_classes,
        'region_fine_label_map': train_dataset.region_fine_label_map,
        'global_idx_to_name': _get_global_label_mapping(train_dataset)
    }

    return train_loader, val_loader, test_loader, dataset_info


def _get_global_label_mapping(dataset):
    """Helper to build global index -> label name mapping."""
    
    mapping = {}
    
    # Iterate through regions in the same order as dataset initialization
    for region, offset in dataset.region_offsets.items():
        fine_map = dataset.region_fine_label_map[region]
        
        # Iterate over local items
        for (dataset_name, orig_idx), local_idx in fine_map.items():
            global_idx = offset + local_idx
            
            # Retrieve name
            info_key = DATASET_INFO_KEYS.get(dataset_name)
            if info_key and str(orig_idx) in INFO[info_key]['label']:
                label_name = INFO[info_key]['label'][str(orig_idx)]
            else:
                label_name = f"{dataset_name}_{orig_idx}" # Fallback
                
            mapping[global_idx] = {
                "name": label_name,
                "region": region,
                "dataset": dataset_name,
                "original_label": orig_idx
            }
            
    return mapping