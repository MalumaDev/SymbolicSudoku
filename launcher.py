# Run main.py with different arguments
import os

from tqdm import tqdm, trange

datasets_key = ["mnist"]#, "emnist", "fmnist", "kmnist", ]
sizes = [4]#, 9]
only_dataset = False
algorithm = 1

for dataset in tqdm(datasets_key):
    for size in sizes:
        d = f"{dataset}{size}"
        for split in trange(11):
            os.system(f"python SymbolicSudoku-main/main.py --dataset {d} --split {split} --algorithm {algorithm}" + (" --generate_dataset" if only_dataset else ""))
