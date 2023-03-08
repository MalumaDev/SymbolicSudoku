# Run main.py with different arguments
import os

datasets_key = ["mnist4", "mnist9"]#, "emnist4", "emnist9", "fmnist4", "fmnist9", "kmnist4", "kmnist9"]

for dataset in datasets_key:
    os.system(f"python main.py --dataset {dataset} --generate_dataset")
