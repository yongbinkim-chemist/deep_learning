import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from timeit import default_timer as timer
from typing import Dict, List, Tuple

def train_step(model: nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: nn.Module = nn.CrossEntropyLoss(),
               device: str = "cpu") -> Tuple[float, float]:
  
  train_loss, train_acc = 0, 0  
  model.train()
  
  for batch, (X, y) in enumerate(train_dataloader):
    X, y = X.to(device), y.to(device)
    # Forward pass
    y_logits = model(X)
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)
    # Calculate the loss
    loss = loss_fn(y_logits, y)
    train_loss += loss.item()
    train_acc += (y_preds == y).sum().item()/len(y_preds)
    # Optimizer zero grad
    optimizer.zero_grad()
    # Loss backward (backpropagation)
    loss.backward()
    # Optimizer step
    optimizer.step()

  train_loss /= len(train_dataloader)
  train_acc /= len(train_dataloader)
  return train_loss, train_acc

def test_step(model: nn.Module,
              test_dataloader: torch.utils.data.DataLoader,
              loss_fn: nn.Module = nn.CrossEntropyLoss(),
              device: str = "cpu") -> Tuple[float, float]:
  
  test_loss, test_acc = 0, 0
  model.eval()

  with torch.inference_mode():
    for batch, (X, y) in enumerate(test_dataloader):
      X, y = X.to(device), y.to(device)
      # Forward pass
      y_logits = model(X)
      # Calculate the loss
      loss = loss_fn(y_logits, y)
      test_loss += loss.item()
      test_acc += (y_logits.argmax(dim=1) == y).sum().item()/len(y_logits)
  
  test_loss /= len(test_dataloader)
  test_acc /= len(test_dataloader)
  return test_loss, test_acc

def train(model: nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          device: str = "cpu") -> Dict[str, List]:

  results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []}
  
  start_time = timer()
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model=model,
                                       train_dataloader=train_dataloader,
                                       optimizer=optimizer,
                                       loss_fn=loss_fn,
                                       device=device)
    test_loss, test_acc = test_step(model=model,
                                    test_dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device=device)
    print(f"Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  return results

if __name__ == "__main__":
  import os
  import torch
  from torch import nn
  from torchvision import transforms
  import data_setup, models

  device = "cuda" if torch.cuda.is_available() else "cpu"
  image_path = "data/food-101-mini"
  BATCH_SIZE = 32
  NUM_WORKERS = os.cpu_count()
  SEED = 42
  EPOCHS = 5
  LR = 0.001

  data_transforms = transforms.Compose([transforms.Resize((64, 64)),
                                        transforms.ToTensor()])
  
  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(image_path=image_path,
                                                                                 transform=data_transforms,
                                                                                 batch_size=BATCH_SIZE,
                                                                                 num_workers=NUM_WORKERS,
                                                                                 seed=SEED)
  model = models.VGGNet(num_classes=len(class_names),
                        drop_p=0.5,
                        batch_norm=True,
                        init_weights=True).to(device)

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(params=model.parameters(),
                               lr=LR)
  results = train(model=model,
                  train_dataloader=train_dataloader,
                  test_dataloader=test_dataloader,
                  optimizer=optimizer,
                  loss_fn=loss_fn,
                  epochs=EPOCHS,
                  device=device)
  print(results)
