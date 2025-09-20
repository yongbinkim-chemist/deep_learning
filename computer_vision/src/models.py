
import torch
from torch import nn

class VGGNet(nn.Module):
  # cfgs = { "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
  #         "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
  #         "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
  #         "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"] }

  def __init__(self,
               num_classes: int,
               drop_p: float = 0.5,
               batch_norm: bool = False,
               init_weights: bool = True):
    
    super().__init__()
    self.cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    self.features = self.create_conv_layers(batch_norm=batch_norm)
    self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
    self.classifier = nn.Sequential(nn.Flatten(),
                                    nn.Linear(in_features=512 * 7 * 7,
                                              out_features=4096,
                                              bias=True),
                                    nn.ReLU(inplace=True), # makes nn.ReLU() overwirte the input tensor directly instead of creating a new one
                                    nn.Dropout(p=drop_p),
                                    nn.Linear(in_features=4096,
                                              out_features=4096,
                                              bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=drop_p),
                                    nn.Linear(in_features=4096,
                                              out_features=num_classes,
                                              bias=True))

    if init_weights:
      # `self.modules()` returns the model and all submodules (layers, activations, etc..) recursively, while self.children() only returns the direct child modules.
      # It's commonly used to loop over layers for tasks like weight initialization.
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
          if m.bias is not None:
            nn.init.constant_(m.bias, val=0)
        elif isinstance(m, nn.Linear):
          nn.init.normal_(m.weight, mean=0, std=0.01)
          nn.init.constant_(m.bias, val=0)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.classifier(self.avgpool(self.features(x)))

  def create_conv_layers(self, batch_norm: bool):
    layers = []
    in_channels = 3 # color channels

    for x in self.cfg:
      if type(x) == int:
        out_channels = x

        if batch_norm:
          layers += [nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False),
                     nn.BatchNorm2d(num_features=out_channels),
                     nn.ReLU()]
        else:
          layers += [nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1),
                     nn.ReLU()]
        in_channels = x
      else:
        layers += [nn.MaxPool2d(kernel_size=2,
                                stride=2,
                                padding=0)]

    return nn.Sequential(*layers)

if __name__ == "__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"

  model = VGGNet(num_classes=3,
                 drop_p=0.5,
                 batch_norm=True,
                 init_weights=True).to(device)
  print(model)
