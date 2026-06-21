import matplotlib.pyplot as plt
from typing import List


def show_from_dataset(dataset, idx: int | List = 0):
	if isinstance(idx, int):
		plt.imshow(dataset[idx][0].permute(1, 2, 0).numpy())
		plt.title(f"Label: {dataset[idx][1].item()}")
		plt.axis('off')
	else:
		if len(idx) >= 5:
			fig, axs = plt.subplots(len(idx) // 5, 5)
			for i, subaxs in enumerate(axs):
				for j, ax in enumerate(subaxs):
					img_idx = i*5+j
					ax.imshow(dataset[img_idx][0].permute(1, 2, 0).numpy())
					ax.set_title(f"Label: {dataset[img_idx][1].item()}")
					ax.axis('off')
		else:
			fig, axs = plt.subplots(1, len(idx))
			for i, ax in enumerate(axs):
				ax.imshow(dataset[i][0].permute(1, 2, 0).numpy())
				ax.set_title(f"Label: {dataset[i][1].item()}")
				ax.axis('off')
	plt.show()