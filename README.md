# ERA V1 Session 6

# Part 1 - Backpropagation
## Forward propagation
First, we compute the output of the neural network by propagating the input data through the network's layers. Each layer has a set of weights to the input and passes the result through an activation function and then calculate the loss.

```ruby
h1 = w1*i1 + w2*i2		
h2 = w3*i1 + w4*i2		
a_h1 = σ(h1) = 1/(1 + exp(-h1))		
a_h2 = σ(h2)		
o1 = w5*a_h1 + w6*a_h2		
o2 = w7*a_h1 + w8*a_h2		
a_o1 = σ(o1)		
a_o2 = σ(o2)	
E_total = E1 + E2		
E1 = ½ * (t1 - a_o1)²		
E2 = ½ * (t2 - a_o2)²	
```
## Backward propagation
```ruby
∂E_total/∂w5 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h1					
∂E_total/∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h2					
∂E_total/∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h1					
∂E_total/∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2					

∂E_total/∂w1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1
∂E_total/∂w2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2												
∂E_total/∂w3 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1												
∂E_total/∂w4 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2		
```
## Loss curve with change in learning rate
## learning rate - 0.1

![image](https://github.com/Navyabhat03/ERA-V1-Session-6/assets/60884505/d9837669-eb78-4e29-8234-1b1e3cb19a95)

## learning rate - 0.2
![image](https://github.com/Navyabhat03/ERA-V1-Session-6/assets/60884505/e40d68ee-733c-4090-93ff-735afb9286ab)

## learning rate - 0.5
![image](https://github.com/Navyabhat03/ERA-V1-Session-6/assets/60884505/4b73a78e-2d12-4f05-a6d2-a14624d7af26)

## learning rate - 0.8
![image](https://github.com/Navyabhat03/ERA-V1-Session-6/assets/60884505/861e1b96-7640-4f5f-a0c4-9bc237d40ae1)


## learning rate - 1.0
![image](https://github.com/Navyabhat03/ERA-V1-Session-6/assets/60884505/46d40ba3-1f12-4f16-b584-dff0c8e1ae53)

## learning rate - 2.0
![image](https://github.com/Navyabhat03/ERA-V1-Session-6/assets/60884505/0aba6405-3f23-454d-8895-48039e0f5b57)

# Part 2

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
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) 


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
 ```
 
### utils.py
- Created two functions **train()** and **test()**
- In train() funtion compute the  prediction, traininng accuracy and loss
- Reset the gradient value to 0
- Perform Back propogation
```ruby
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = F.nll_loss(y_pred, target)
        train_losses.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))
```

### S6.ipynb
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
model =  Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=6, gamma=0.1)


EPOCHS = 20
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    utils.train(model, device, train_loader, optimizer, epoch)
    scheduler.step()
    utils.test(model, device, test_loader)
```

We used total 20 epoch (The number of passes that needs to be done on entire dataset)
```
EPOCH: 0
Loss=0.07002481818199158 Batch_id=937 Accuracy=90.91: 100%|██████████████████████████| 938/938 [04:00<00:00,  3.90it/s]

Test set: Average loss: 0.0461, Accuracy: 9870/10000 (98.70%)

EPOCH: 1
Loss=0.03088083304464817 Batch_id=937 Accuracy=97.71: 100%|██████████████████████████| 938/938 [06:52<00:00,  2.27it/s]

Test set: Average loss: 0.0304, Accuracy: 9909/10000 (99.09%)

EPOCH: 2
Loss=0.025257356464862823 Batch_id=937 Accuracy=98.13: 100%|█████████████████████████| 938/938 [03:49<00:00,  4.09it/s]

Test set: Average loss: 0.0287, Accuracy: 9909/10000 (99.09%)

EPOCH: 3
Loss=0.05391354486346245 Batch_id=937 Accuracy=98.41: 100%|██████████████████████████| 938/938 [03:05<00:00,  5.06it/s]

Test set: Average loss: 0.0254, Accuracy: 9919/10000 (99.19%)

EPOCH: 4
Loss=0.08542244881391525 Batch_id=937 Accuracy=98.53: 100%|██████████████████████████| 938/938 [03:24<00:00,  4.58it/s]

Test set: Average loss: 0.0258, Accuracy: 9913/10000 (99.13%)

EPOCH: 5
Loss=0.050177741795778275 Batch_id=937 Accuracy=98.69: 100%|█████████████████████████| 938/938 [03:27<00:00,  4.53it/s]

Test set: Average loss: 0.0226, Accuracy: 9935/10000 (99.35%)

EPOCH: 6
Loss=0.026430033147335052 Batch_id=937 Accuracy=98.94: 100%|█████████████████████████| 938/938 [02:05<00:00,  7.48it/s]

Test set: Average loss: 0.0187, Accuracy: 9938/10000 (99.38%)

EPOCH: 7
Loss=0.04906224459409714 Batch_id=937 Accuracy=98.99: 100%|██████████████████████████| 938/938 [01:58<00:00,  7.91it/s]

Test set: Average loss: 0.0185, Accuracy: 9940/10000 (99.40%)

EPOCH: 8
Loss=0.018612515181303024 Batch_id=937 Accuracy=99.00: 100%|█████████████████████████| 938/938 [02:23<00:00,  6.53it/s]

Test set: Average loss: 0.0179, Accuracy: 9942/10000 (99.42%)

EPOCH: 9
Loss=0.0925775095820427 Batch_id=937 Accuracy=99.03: 100%|███████████████████████████| 938/938 [03:56<00:00,  3.97it/s]

Test set: Average loss: 0.0194, Accuracy: 9941/10000 (99.41%)

EPOCH: 10
Loss=0.0022130135912448168 Batch_id=937 Accuracy=99.08: 100%|████████████████████████| 938/938 [02:55<00:00,  5.35it/s]

Test set: Average loss: 0.0177, Accuracy: 9944/10000 (99.44%)

EPOCH: 11
Loss=0.0562167763710022 Batch_id=937 Accuracy=99.10: 100%|███████████████████████████| 938/938 [03:49<00:00,  4.10it/s]

Test set: Average loss: 0.0186, Accuracy: 9941/10000 (99.41%)

EPOCH: 12
Loss=0.21695517003536224 Batch_id=937 Accuracy=99.16: 100%|██████████████████████████| 938/938 [03:15<00:00,  4.80it/s]

Test set: Average loss: 0.0190, Accuracy: 9942/10000 (99.42%)

EPOCH: 13
Loss=0.1727476269006729 Batch_id=937 Accuracy=99.14: 100%|███████████████████████████| 938/938 [01:55<00:00,  8.14it/s]

Test set: Average loss: 0.0183, Accuracy: 9943/10000 (99.43%)

EPOCH: 14
Loss=0.03677593171596527 Batch_id=937 Accuracy=99.13: 100%|██████████████████████████| 938/938 [02:34<00:00,  6.07it/s]

Test set: Average loss: 0.0179, Accuracy: 9943/10000 (99.43%)

EPOCH: 15
Loss=0.005717517342418432 Batch_id=937 Accuracy=99.17: 100%|█████████████████████████| 938/938 [02:44<00:00,  5.71it/s]

Test set: Average loss: 0.0178, Accuracy: 9939/10000 (99.39%)

EPOCH: 16
Loss=0.009821132756769657 Batch_id=937 Accuracy=99.16: 100%|█████████████████████████| 938/938 [05:11<00:00,  3.01it/s]

Test set: Average loss: 0.0190, Accuracy: 9940/10000 (99.40%)

EPOCH: 17
Loss=0.0018008594634011388 Batch_id=937 Accuracy=99.11: 100%|████████████████████████| 938/938 [02:27<00:00,  6.37it/s]

Test set: Average loss: 0.0183, Accuracy: 9940/10000 (99.40%)

EPOCH: 18
Loss=0.0014565926976501942 Batch_id=937 Accuracy=99.11: 100%|████████████████████████| 938/938 [02:26<00:00,  6.39it/s]

Test set: Average loss: 0.0175, Accuracy: 9944/10000 (99.44%)

EPOCH: 19
Loss=0.07155724614858627 Batch_id=937 Accuracy=99.17: 100%|██████████████████████████| 938/938 [01:56<00:00,  8.08it/s]

Test set: Average loss: 0.0177, Accuracy: 9944/10000 (99.44%)

```
We can see the train accuracy is 99.17% and test accuracy is 99.44%

## Model Summary
- End we printed the summary of model

```ruby
from torchsummary import summary
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
```
```
cpu
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
           Dropout-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 32, 24, 24]           4,608
              ReLU-6           [-1, 32, 24, 24]               0
       BatchNorm2d-7           [-1, 32, 24, 24]              64
           Dropout-8           [-1, 32, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             320
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 16, 10, 10]           1,440
             ReLU-12           [-1, 16, 10, 10]               0
      BatchNorm2d-13           [-1, 16, 10, 10]              32
          Dropout-14           [-1, 16, 10, 10]               0
           Conv2d-15             [-1, 16, 8, 8]           2,304
             ReLU-16             [-1, 16, 8, 8]               0
      BatchNorm2d-17             [-1, 16, 8, 8]              32
          Dropout-18             [-1, 16, 8, 8]               0
           Conv2d-19             [-1, 16, 6, 6]           2,304
             ReLU-20             [-1, 16, 6, 6]               0
      BatchNorm2d-21             [-1, 16, 6, 6]              32
          Dropout-22             [-1, 16, 6, 6]               0
           Conv2d-23             [-1, 16, 6, 6]           2,304
             ReLU-24             [-1, 16, 6, 6]               0
      BatchNorm2d-25             [-1, 16, 6, 6]              32
          Dropout-26             [-1, 16, 6, 6]               0
        AvgPool2d-27             [-1, 16, 1, 1]               0
           Conv2d-28             [-1, 10, 1, 1]             160
================================================================
Total params: 13,808
Trainable params: 13,808
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.06
Params size (MB): 0.05
Estimated Total Size (MB): 1.12
----------------------------------------------------------------
```

### Thank You
