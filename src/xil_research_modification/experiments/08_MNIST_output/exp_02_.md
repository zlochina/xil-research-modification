
# Details of the 02th experimentation's results

## Gradcam explanation
![exp_02_result.png](./exp_02_result.png)

## Model architecture

```python
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
```


## Training

- Training data: subset length of 100
- ADAM optimizer:
```python
lr = 0.001
adam_opt = Adam(model.parameters(), lr=lr)
```

- Loss function:
```python
loss = nn.CrossEntropyLoss()
```

- Training output:

```
Epoch 1
--------------------
loss: 0.687422 [   64/  100]
Test Error: 
 Accuracy: 50.2%, Avg loss: 0.673224 

Epoch 2
--------------------
loss: 0.658346 [   64/  100]
Test Error: 
 Accuracy: 56.4%, Avg loss: 0.641278 

Epoch 3
--------------------
loss: 0.622856 [   64/  100]
Test Error: 
 Accuracy: 84.5%, Avg loss: 0.595633 

Epoch 4
--------------------
loss: 0.549103 [   64/  100]
Test Error: 
 Accuracy: 90.2%, Avg loss: 0.551854 

Epoch 5
--------------------
loss: 0.503039 [   64/  100]
Test Error: 
 Accuracy: 92.1%, Avg loss: 0.505418 

Epoch 6
--------------------
loss: 0.481600 [   64/  100]
Test Error: 
 Accuracy: 92.8%, Avg loss: 0.467716 

Epoch 7
--------------------
loss: 0.424880 [   64/  100]
Test Error: 
 Accuracy: 90.5%, Avg loss: 0.456843 

Epoch 8
--------------------
loss: 0.408036 [   64/  100]
Test Error: 
 Accuracy: 93.1%, Avg loss: 0.420653 

Epoch 9
--------------------
loss: 0.374069 [   64/  100]
Test Error: 
 Accuracy: 93.6%, Avg loss: 0.395361 

Epoch 10
--------------------
loss: 0.372876 [   64/  100]
Test Error: 
 Accuracy: 94.1%, Avg loss: 0.383304 

Epoch 11
--------------------
loss: 0.373261 [   64/  100]
Test Error: 
 Accuracy: 94.1%, Avg loss: 0.383514 

Epoch 12
--------------------
loss: 0.337864 [   64/  100]
Test Error: 
 Accuracy: 94.2%, Avg loss: 0.377352 

Epoch 13
--------------------
loss: 0.339471 [   64/  100]
Test Error: 
 Accuracy: 95.6%, Avg loss: 0.363040 

Epoch 14
--------------------
loss: 0.345438 [   64/  100]
Test Error: 
 Accuracy: 95.9%, Avg loss: 0.356655 

Epoch 15
--------------------
loss: 0.326928 [   64/  100]
Test Error: 
 Accuracy: 96.3%, Avg loss: 0.353133 

Epoch 16
--------------------
loss: 0.333543 [   64/  100]
Test Error: 
 Accuracy: 96.2%, Avg loss: 0.354929 

Epoch 17
--------------------
loss: 0.317823 [   64/  100]
Test Error: 
 Accuracy: 95.5%, Avg loss: 0.358456 

Epoch 18
--------------------
loss: 0.326388 [   64/  100]
Test Error: 
 Accuracy: 95.7%, Avg loss: 0.355547 

Epoch 19
--------------------
loss: 0.319027 [   64/  100]
Test Error: 
 Accuracy: 96.3%, Avg loss: 0.349122 

Epoch 20
--------------------
loss: 0.315832 [   64/  100]
Test Error: 
 Accuracy: 96.6%, Avg loss: 0.346879 

Done!
```


## Model weights
[model weights](./exp_02_weights.pth)
