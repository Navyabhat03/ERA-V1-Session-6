# ERA V1 Session 5

## Create and Train a Neural Network in Python

An implementation to create and train a simple neural network in python - just to learn the basics of how neural networks work.

## Usage
### model.py

- First we import the important libraries and packages. 

```ruby
import torch.nn as nn
import torch.nn.functional as F
```
- Next we build a simple Neural Network model. Here, we use the nn package to implement our model. 
For this, we define a **class Net()** and pass **nn.Module** as the parameter.

```ruby
class Net(nn.Module):
```

- Create two functions inside the class to get our model ready. First is the **init()** and the second is the **forward()**.
- Within the init() function, we call a super() function and define different layers.
- We need to instantiate the class to use for training on the dataset. When we instantiate the class, the forward() function is executed.

```ruby
def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
    self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
    self.fc1 = nn.Linear(4096, 50)
    self.fc2 = nn.Linear(50, 10)

def forward(self, x):
    x = F.relu(self.conv1(x), 2)
    x = F.relu(F.max_pool2d(self.conv2(x), 2)) 
    x = F.relu(self.conv3(x), 2)
    x = F.relu(F.max_pool2d(self.conv4(x), 2)) 
    x = x.view(-1, 4096)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)
 ```
 
### utils.py
- Created two functions **train()** and **test()**
- In train() funtion compute the  prediction, traininng accuracy and loss
- Reset the gradient value to 0
- Perform Back propogation
```ruby
def train(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))
```
- And in test() function calculated the loss and accuracy of the model
```ruby
def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```

### S5.ipynb
- First Loaded MNIST dataset
```ruby
from torchvision import datasets
```

- Defining device
```ruby
device = torch.device("cuda" if cuda else "cpu")
```

- Then reating train data and test data
```ruby
train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
```

- Plotting the dataset of **train_loader**

![image](https://github.com/Navyabhat03/ERA-V1/assets/60884505/a79a40ab-1603-49ea-81ea-424f145e4a6c)

- **Training and Testing trigger**
```ruby
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
# New Line
criterion = F.nll_loss
num_epochs = 20

for epoch in range(1, num_epochs+1):
  print(f'Epoch {epoch}')
  utils.train(model, device, train_loader, optimizer, criterion)
  utils.test(model, device, test_loader, criterion)
  scheduler.step()
```

We used total 20 epoch (The number of passes that needs to be done on entire dataset)
```
Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 20
Train: Loss=0.0613 Batch_id=117 Accuracy=99.09: 100%|██████████| 118/118 [03:16<00:00,  1.67s/it]
Test set: Average loss: 0.0214, Accuracy: 9927/10000 (99.27%)

Adjusting learning rate of group 0 to 1.0000e-03.
```
We can see the accuracy is above 95%

## Model Summary
- End we printed the summary of model

```ruby
from torchsummary import summary
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
```
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94
```

### Thank You
