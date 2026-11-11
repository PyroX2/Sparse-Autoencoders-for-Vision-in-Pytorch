import torch
from sklearn import model_selection
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import v2


class DummyDataset(data.Dataset):
    def __init__(self, x, y, transform=None):
        super().__init__()
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.transform is not None:
            return self.transform(self.x[index]), self.y[index]
        return self.x[index], self.y[index]


class CelebADataset(data.Dataset):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("CelebA dataset is not implemented.")


class ColoredMNISTDataset:
    """Wrapper class that applies coloring based on the target class ID."""

    COLORS = (
        "#FF0000",  # Red
        "#00FF00",  # Green
        "#0000FF",  # Blue
        "#FFFF00",  # Yellow
        "#FF00FF",  # Magenta
        "#00FFFF",  # Cyan
        "#FFA500",  # Orange
        "#B1636F",  # Purple
        "#BED944",
        "#9792D4FF",
    )

    def __init__(
        self,
        val_split: float = 0.2,
        transform: v2.Transform | None = None,
        colored_proportions: float = 0.95,
        colors_list: list | tuple = COLORS,
        seed: int = 0,
    ):
        """
        Initialize
        """
        self.colored_proportions = colored_proportions

        train_dataset = datasets.MNIST(root="./data", download=True)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True)

        train_labels = torch.tensor([y for x, y in train_dataset])
        train_split, val_split = model_selection.train_test_split(
            train_dataset, test_size=val_split, stratify=train_labels, random_state=seed
        )

        self.colors_list = colors_list
        self.pil_to_tensor = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )
        self.transform = transform

        train_x, train_y = self.color_dataset(train_split)
        val_x, val_y = self.color_dataset(val_split)
        test_x, test_y = self.color_dataset(test_dataset)

        self.train = DummyDataset(train_x, train_y)
        self.val = DummyDataset(val_x, val_y)
        self.test = DummyDataset(test_x, test_y)

    def color_dataset(self, dataset):
        x_list = []
        for x, y in dataset:
            if torch.rand(1).item() < self.colored_proportions:
                colored_image = self._gray_to_colored(self.pil_to_tensor(x), y)
                x_list.append(colored_image)
            else:
                w, h = x.size
                x_list.append(self.pil_to_tensor(x).expand(1, 3, w, h))

        x_tensor = torch.cat(x_list, dim=0)
        y_tensor = torch.tensor([y for x, y in dataset])
        return x_tensor, y_tensor

    def _gray_to_colored(self, x, y):
        assert len(self.colors_list) > y, (
            f"Color index {y} is out of range for COLORS list."
        )

        color = self.colors_list[y]
        colored_image = torch.zeros(3, x.shape[1], x.shape[2])
        colored_image[0] = x * int(color[1:3], 16) / 255.0
        colored_image[1] = x * int(color[3:5], 16) / 255.0
        colored_image[2] = x * int(color[5:7], 16) / 255.0
        return colored_image.unsqueeze(0)


class WaterBirdsDataset(data.Dataset):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("WaterBirds dataset is not implemented.")
