import torch
from torch.utils import data
from torchvision.transforms import v2
from config import COLORS



class ColoredMNISTDataset(data.Dataset):
	"""Wrapper class that applies coloring based on the target class ID."""
	def __init__(self, dataset_list, transform=None, colored_proportions=0.95, colors_list=COLORS):
		"""
			Initialize 
		"""
		super().__init__()
		self.colors_list = colors_list
		self.pil_to_tensor = v2.Compose(
			[
				v2.ToImage(),
				v2.ToDtype(torch.float32, scale=True)
			]
		)

		self.x = []
		for x, y in dataset_list:
			if torch.rand(1).item() < colored_proportions:
				colored_image = self._gray_to_colored(self.pil_to_tensor(x), y)
				self.x.append(colored_image)
			else:
				w, h = x.size
				self.x.append(self.pil_to_tensor(x).expand(1, 3, w, h))

		self.x = torch.cat(self.x, dim=0)
		self.y = torch.tensor([y for x, y in dataset_list])
		self.transform = transform

	def __len__(self):
		return len(self.x)
	
	def __getitem__(self, index):
		if self.transform is not None:
			return self.transform(self.x[index]), self.y[index]
		return self.x[index], self.y[index]
	
	def _gray_to_colored(self, x, y):
		assert len(self.colors_list) > y, f"Color index {y} is out of range for COLORS list."

		color = self.colors_list[y]
		colored_image = torch.zeros(3, x.shape[1], x.shape[2])
		colored_image[0] = x * int(color[1:3], 16) / 255.0
		colored_image[1] = x * int(color[3:5], 16) / 255.0
		colored_image[2] = x * int(color[5:7], 16) / 255.0
		return colored_image.unsqueeze(0)