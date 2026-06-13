import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def process_single_file(pt_file, displacement_threshold=1.0):
	sample = torch.load(pt_file)	
	radii = sample['radii'].numpy() if isinstance(sample['radii'], torch.Tensor) else sample['radii']
	displacement = sample['displacement'].numpy() if isinstance(sample['displacement'], torch.Tensor) else sample['displacement']
	perturbed = sample['perturbed'].numpy() if isinstance(sample['perturbed'], torch.Tensor) else sample['perturbed']

	if perturbed.any():
		radius_label = float(np.quantile(radii[perturbed.astype(bool)], 0.95))
	else:
		radius_label = 0.0
	max_disp = float(displacement.max()) if displacement.size else 0.0

	if max_disp <= displacement_threshold:
		class_label = 0
	elif radius_label <= 8.0:
		class_label = 1
	else:
		class_label = 2

	sample['radius_label'] = torch.tensor([radius_label], dtype=torch.float32)
	sample['class_label'] = torch.tensor([class_label], dtype=torch.long)

	# torch.save(sample, str(pt_file).replace('samples_subset_c1000_raw', 'samples_subset_c1000_raw_r95'))
	return class_label

def update_labels(sample_dir, max_workers=16):
	sample_dir = Path(sample_dir)
	pt_files = list(sample_dir.glob("*.pt"))
	class_counts = {0: 0, 1: 0, 2: 0}
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		futures = [executor.submit(process_single_file, pt_file) for pt_file in pt_files]
		for future in tqdm(as_completed(futures), total=len(futures), desc="Updating labels (Parallel)"):
			cls_label = future.result()
			class_counts[cls_label] += 1
	for cls, count in sorted(class_counts.items()):
		print(f"Class {cls}: {count}")

if __name__ == "__main__":
	update_labels("data/processed/samples_subset_c1000_raw/")
