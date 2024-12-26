# Details of the 1st experementation's results

## Gradcam explanation
![exp_01_result.png](./exp_01_result.png)

## Model architecture
```py
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, ReLU, Linear, Dropout, Softmax

# input: 28X28 images
# output: 3 possible outputs (0: out of problem label, 1: number 0, 2: number 1)

model = Sequential(
    # 1st group
    Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),
    # 2nd group
    Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),
    # TODO: maybe 3rd group
    # Flattening into 1d array
    Flatten(),
    # Dense layers (aka Fully-connected layers)
    Linear(in_features=32 * 7 * 7, out_features=128), # maxpooling reduces dimensionality by half, 7 = 28 (image_size) / (2 * 2)
    ReLU(),
    Linear(in_features=128, out_features=num_classes),
    Softmax(dim=1)
)
model.eval()
```

## Training
- SGD with momentum:
```py
lr = 0.001
sgd_opt = SGD(model.parameters(), lr=lr, momentum=0.9)
```
- Loss function:
```py
loss = nn.CrossEntropyLoss()
```
- Training output:
```
Epoch 1
--------------------
loss: 0.693945 [   64/11774]
loss: 0.684892 [ 6464/11774]
Test Error: 
 Accuracy: 93.3%, Avg loss: 0.675611 

Done!
```


## Model weights
[model weights](./exp_01weights.pth)
