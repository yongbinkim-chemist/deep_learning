from typing import List, Tuple
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from torchvision import transforms

def predict_and_plot(model: torch.nn.Module,
                     class_names: List[str],
                     image_path: str,
                     image_size: Tuple[int, int],
                     transform: torchvision.transforms = None,
                     device: torch.device = "cpu"):

  img = Image.open(image_path)  
  if transform:
    transformed_img = transform(img)
  else:
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor()])
    transformed_img = transform(img)

  ### Make prediction ###
  model.to(device)
  model.eval()
  with torch.inference_mode():
    pred_logit = model(transformed_img.unsqueeze(dim=0).to(device))
    pred_prob = torch.softmax(pred_logit, dim=1)
    pred_label = torch.argmax(pred_prob, dim=1)
    pred_class = class_names[pred_label]
  
    ### Plot image ###
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {pred_class} | Prob: {pred_prob.max():.3f}")
    plt.axis(False)
