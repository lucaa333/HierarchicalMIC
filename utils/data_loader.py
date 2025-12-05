import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from medmnist import (
    OrganMNIST3D,
    NoduleMNIST3D,
    AdrenalMNIST3D,
    FractureMNIST3D,
    VesselMNIST3D,
    SynapseMNIST3D
)


# Dataset mapping for anatomical regions
REGION_DATASET_MAPPING = {
    'organ': OrganMNIST3D,      # Multi-organ (11 classes)
    'nodule': NoduleMNIST3D,    # Chest (2 classes: benign/malignant)
    'adrenal': AdrenalMNIST3D,  # Abdomen (2 classes)
    'fracture': FractureMNIST3D, # Bone (3 classes)
    'vessel': VesselMNIST3D,    # Brain vessels (2 classes)
    'synapse': SynapseMNIST3D   # Brain synapses (2 classes)
}

# Anatomical region mapping for hierarchical classification
DATASET_TO_REGION = {
    'organ': 'multi',  # Contains multiple regions
    'nodule': 'chest',
    'adrenal': 'abdomen',
    'fracture': 'chest',  # Rib fractures from chest CT
    'vessel': 'brain',
    'synapse': 'brain'
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
    def __init__(self, datasets_config, split='train'):
        self.split = split
        self.samples = []
        self.coarse_labels = []
        self.fine_labels = []

        # Region to index mapping
        unique_regions = list(set(DATASET_TO_REGION.values()))
        self.region_to_idx = {region: idx for idx, region in enumerate(unique_regions)}

        # Load and combine datasets
        for dataset_name, region in DATASET_TO_REGION.items():
            if dataset_name in datasets_config:
                dataset_class = REGION_DATASET_MAPPING[dataset_name]
                dataset = dataset_class(split=split, download=True)

                region_idx = self.region_to_idx[region]

                for i in range(len(dataset)):
                    img, label = dataset[i]
                    self.samples.append(img)
                    self.coarse_labels.append(region_idx)
                    self.fine_labels.append(label)

        self.samples = np.array(self.samples)
        self.coarse_labels = np.array(self.coarse_labels)
        self.fine_labels = np.array(self.fine_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.samples[idx]).float()
        coarse_label = torch.tensor(self.coarse_labels[idx]).long()
        fine_label = torch.tensor(self.fine_labels[idx]).long()

        # Normalize if needed
        if img.max() > 1:
            img = img / 255.0

        return img, coarse_label, fine_label


def create_hierarchical_dataset(
    datasets_to_include=None,
    batch_size=32,
    num_workers=4
):
    if datasets_to_include is None:
        datasets_to_include = ['nodule', 'adrenal', 'vessel']

    datasets_config = {name: True for name in datasets_to_include}

    train_dataset = HierarchicalMedMNISTDataset(datasets_config, split='train')
    val_dataset = HierarchicalMedMNISTDataset(datasets_config, split='val')
    test_dataset = HierarchicalMedMNISTDataset(datasets_config, split='test')

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

    return train_loader, val