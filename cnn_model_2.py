import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # input_channel, output_channel, kernel_size, stride, padding， 下面的意思就是省略了stride，默认是1
    self.conv1 = nn.Conv2d(1, 128, 5, padding=2)
    self.conv2 = nn.Conv2d(128, 128, 5, padding=2)
    self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
    self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.bn_conv1 = nn.BatchNorm2d(128)
    self.bn_conv2 = nn.BatchNorm2d(128)
    self.bn_conv3 = nn.BatchNorm2d(256)
    self.bn_conv4 = nn.BatchNorm2d(256)
    self.bn_dense1 = nn.BatchNorm1d(1024)
    self.bn_dense2 = nn.BatchNorm1d(512)
    self.dropout_conv = nn.Dropout2d(p=0.25)
    self.dropout = nn.Dropout(p=0.5)
    self.fc1 = nn.Linear(256 * 7 * 7 , 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 10)

    # (bt, 1, 28 , 28)
  def conv_layers(self, x):
      # (bt, 1, 28, 28)
    out = F.relu(self.bn_conv1(self.conv1(x)))
      # (bt, 128, 28, 28)
    out = F.relu(self.bn_conv2(self.conv2(out)))
      # (bt, 128, 28, 28)
    out = self.pool(out)
      # (bt, 128, 14, 14)
    out = self.dropout_conv(out)
      # (bt, 128, 14, 14)
    out = F.relu(self.bn_conv3(self.conv3(out)))
      # (bt, 256, 14, 14)
    out = F.relu(self.bn_conv4(self.conv4(out)))
      # (bt, 256, 14, 14)
    out = self.pool(out)
      # (bt, 256, 7, 7)
    out = self.dropout_conv(out)
    return out
  # (bt, 256, 7, 7)

  def dense_layers(self, x):
      # (bt, 256*7*7)
    out = F.relu(self.bn_dense1(self.fc1(x)))
      # (bt, 1024)
    out = self.dropout(out)
    out = F.relu(self.bn_dense2(self.fc2(out)))
      # (bt, 512)
    out = self.dropout(out)
    out = self.fc3(out)
      # (bt, 10)
    return out
  # (bt, 10)

  def forward(self, x):
      # (bt, 1, 28, 28 )
    out = self.conv_layers(x)
      # (bt, 256 , 7 ,7)

      # what does this mean?
    out = out.view(-1, 256 * 7 * 7)
      # (bt, 256*7*7)
    out = self.dense_layers(out)
      # (bt, 10)
    return out

# net = Net()
# print(net)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('Device:', device)
# net.to(device)
#
# num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
# print("Number of trainable parameters:", num_params)