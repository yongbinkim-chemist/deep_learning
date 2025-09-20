from typing import Tuple, List
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

def create_dataloaders(image_path: str,
                       transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int = 1,
                       seed: int = None) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, List[str]]:
  """Creates training and testing DataLoaders

  Takes in image path and turns it into PyTorch DataLoader.

  Args:
    image_path: Path to images.
    transform: torchvision transforms to perform on images.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.
    seed: Random seed for train_test_split.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).

  Example usage:
    train_dataloader, test_dataloader, class_names = create_dataloaders(image_path=path/to/images
                                                                        transform=some_transforms,
                                                                        batch_size=32,
                                                                        num_workers=1,
                                                                        seed=42)
  """

  dataset = datasets.ImageFolder(root=image_path,
                                 transform=transform)
  class_names = dataset.classes

  targets = [label for _, label in dataset.samples]
  indices = list(range(len(targets)))
  train_indices, test_indices = train_test_split(indices,
                                                test_size=0.25,
                                                stratify=targets,
                                                random_state=seed)

  train_dataset = Subset(dataset, train_indices)
  test_dataset = Subset(dataset, test_indices)
  
  train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True)

  test_dataloader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers)
  
  return train_dataloader, test_dataloader, class_names

if __name__ == "__main__":
  import os

  image_path = "data/food-101-mini"
  BATCH_SIZE = 32
  NUM_WORKERS = os.cpu_count()
  SEED = 42
  data_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor()])
  
  train_dataloader, test_dataloader, class_names = create_dataloaders(image_path=image_path,
                                                                      transform=data_transforms,
                                                                      batch_size=BATCH_SIZE,
                                                                      num_workers=NUM_WORKERS,
                                                                      seed=SEED)
  print(train_dataloader, test_dataloader, class_names)
