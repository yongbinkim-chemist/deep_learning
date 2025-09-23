
import os
import argparse

import torch
from torch import nn
from torchvision import transforms

import data_setup, models, engine, utils

# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

# Get an arg for num_epochs
parser.add_argument("--num_epochs",
                    default=5,
                    type=int,
                    help="the number of epochs to train for")

# Get an arg for batch_size
parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="number of samples per batch")

# Get an arg for hidden_units
parser.add_argument("--hidden_units",
                    default=10,
                    type=int,
                    help="number of hidden units in hidden layers")

# Get an arg for learning_rate
parser.add_argument("--learning_rate",
                    default=0.001,
                    type=float,
                    help="learning rate to use for model")

# Get an arg for image directory
parser.add_argument("--image_path",
                    default="data/food-101-mini",
                    type=str,
                    help="path to folder with images")


# Get an arg for seed
parser.add_argument("--random_seed",
                    default=None,
                    type=int,
                    help="random seed value")

# Get argments from the parser
args = parser.parse_args()

# Device agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
RANDOM_SEED = args.random_seed
IMAGE_PATH = args.image_path
NUM_WORKERS = os.cpu_count()

MODEL = "VGGNet" # hyperparameter later
print(f"[INFO] Chosen model: {MODEL}")
print(f"[INFO] Training a model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE} using a learning rate of {LEARNING_RATE}")


data_transform = transforms.Compose([transforms.Resize((64, 64)),
                                     transforms.ToTensor()])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(image_path=IMAGE_PATH,
                                                                               transform=data_transform,
                                                                               batch_size=BATCH_SIZE,
                                                                               num_workers=NUM_WORKERS,
                                                                               seed=RANDOM_SEED)
model = models.VGGNet(hidden_units=HIDDEN_UNITS,
                      num_classes=len(class_names),
                      drop_p=0.5,
                      batch_norm=True,
                      init_weights=True).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=LEARNING_RATE)

results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=NUM_EPOCHS,
                       device=device)

utils.save_model(model=model,
                 target_dir="models",
                 model_name=f"computer_vision_{MODEL}.pth")
