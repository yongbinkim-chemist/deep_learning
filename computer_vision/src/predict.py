from typing import List
import torch
import torchvision
import argparse
import models

def load_model(filepath: str,
               hidden_units: int,
               num_classes: int):
  
  model = models.VGGNet(hidden_units=10,
                        num_classes=3,
                        drop_p=0.5,
                        batch_norm=True,
                        init_weights=True)
  
  model.load_state_dict(torch.load(filepath))
  return model

def predict_on_image(model_path: str,
                     image_path: str,
                     hidden_units: int,
                     class_names: List,
                     device: str):
  
  model = load_model(filepath=model_path,
                     hidden_units=hidden_units,
                     num_classes=len(class_names))
  
  image = torchvision.io.read_image(str(image_path)).type(torch.float32)

  image = image / 255.
  transform = torchvision.transforms.Resize((64, 64))
  image = transform(image)

  model.eval()
  with torch.inference_mode():
    image = image.to(device)
    image = torch.unsqueeze(image, dim=0)
    
    pred_logits = model(image)
    pred_probs = torch.softmax(pred_logits, dim=1)
    pred_label = torch.argmax(pred_probs, dim=1)
    pred_class = class_names[pred_label]

  print(f"[INFO] Pred class: {pred_class}, Pred prob: {pred_probs.max():.3f}")

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument("--image",
                      help="target image filepath to predict on")

  parser.add_argument("--model_path",
                      default="models/computer_vision_modular.pth",
                      type=str,
                      help="target model to use for prediction filepath")

  parser.add_argument("--hidden_units",
                      default=10,
                      type=int,
                      help="number of hidden units in hidden layers")

  args = parser.parse_args()

  class_names = ["bibimbap", "hamburger", "sashimi"]

  device = "cuda" if torch.cuda.is_available() else "cpu"

  IMG_PATH = args.image
  print(f"[INFO] Predicting on {IMG_PATH}")

  predict_on_image(model_path=args.model_path,
                   image_path=args.image,
                   hidden_units=args.hidden_units,
                   class_names=class_names,
                   device=device)
