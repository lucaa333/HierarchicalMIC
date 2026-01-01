
import sys
import os
import torch
import numpy as np

# Mocking MedMNIST to avoid actual downloads if possible, or just relying on existing files.
# But assuming the user has the data since they were running notebooks.
# We will just import the dataloader and test it.

sys.path.append(os.path.abspath('c:/Users/BS-06/Desktop/HMIC/HierarchicalMIC'))

from utils.data_loader import create_hierarchical_dataset, HierarchicalMedMNISTDataset

def verify_dataloader():
    print("Starting verification...")
    
    # Test 1: Default behavior (Should be 3 values)
    print("\n--- Test 1: Default (return_global_labels=False) ---")
    try:
        # We invoke the dataset directly to be fast, avoiding the full loader creation overhead if possible
        # But create_hierarchical_dataset is what we changed too.
        
        # We use a small subset or just one dataset to be fast. 'organ' is smallish.
        config = {'organ': True}
        ds = HierarchicalMedMNISTDataset(config, split='test', augment=False)
        
        item = ds[0]
        print(f"Returned {len(item)} items.")
        if len(item) == 3:
            print("PASS: Got 3 items (img, coarse, fine).")
        else:
            print(f"FAIL: Expected 3 items, got {len(item)}.")
            
    except Exception as e:
        print(f"FAIL: Exception in Test 1: {e}")

    # Test 2: Opt-in behavior (Should be 4 values)
    print("\n--- Test 2: Opt-in (return_global_labels=True) ---")
    try:
        config = {'organ': True}
        ds_global = HierarchicalMedMNISTDataset(config, split='test', augment=False, return_global_labels=True)
        
        item = ds_global[0]
        print(f"Returned {len(item)} items.")
        if len(item) == 4:
            print("PASS: Got 4 items (img, coarse, fine, global_fine).")
        else:
            print(f"FAIL: Expected 4 items, got {len(item)}.")
            
    except Exception as e:
        print(f"FAIL: Exception in Test 2: {e}")

if __name__ == "__main__":
    verify_dataloader()
