
import torch
from torch import nn
from pathlib import Path

def save_model(model: nn.Module,
               target_dir: str,
               model_name: str) -> None:

  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model (.pth or .pt).
  
  Example usage:
    save_model(model=model,
               target_dir="models",
               model_name="computer_vision_modular.pth")
  """

  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)
  
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
