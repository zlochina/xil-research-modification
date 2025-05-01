from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Linear, Softmax
from src.masked_dropout import MaskedDropout, ScaleLessDropout


class CNNTwoConv(Module):
    def __init__(self, num_classes=10, device="cpu"):
        super(CNNTwoConv, self).__init__()
        self.features = Sequential(
            # 1st group
            Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, device=device),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, return_indices=True),
            # 2nd group
            Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, device=device),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )
        self.classifier = Sequential(
            Flatten(),
            Linear(in_features=32 * 7 * 7, out_features=128, device=device),
            # maxpooling reduces dimensionality by half, 7 = 28 (image_size) / (2 * 2)
            ReLU(),
            Linear(in_features=128, out_features=num_classes, device=device),
            Softmax(dim=1)
        )

    def forward(self, x):
        indices_list = []
        for layer in self.features:
            if isinstance(layer, MaxPool2d):
                x, indices = layer(x)
                indices_list.append(indices)
            else:
                x = layer(x)
        x = self.classifier(x)
        return x

    def __getitem__(self, idx):
        features_size = len(self.features)
        classifier_size = len(self.classifier)
        if idx < 0 or idx >= (features_size + classifier_size):
            raise RuntimeError(f"Index [{idx}] out of bounds")
        if idx < features_size:
            return self.features[idx]
        idx -= features_size
        return self.classifier[idx]


class CNNTwoConvWithDropout(CNNTwoConv):
    def __init__(self, num_classes=10, dropout_prob=0.5, device="cpu"):
        super(CNNTwoConvWithDropout, self).__init__()
        self.dropout = MaskedDropout(torch_dropout=ScaleLessDropout(p=dropout_prob))
        self.features = Sequential(
            # 1st group
            Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, device=device),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, return_indices=True),
            # 2nd group
            Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, device=device),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )
        self.classifier = Sequential(
            Flatten(),
            Linear(in_features=32 * 7 * 7, out_features=128, device=device),
            # maxpooling reduces dimensionality by half, 7 = 28 (image_size) / (2 * 2)
            ReLU(),
            Linear(in_features=128, out_features=num_classes, device=device),
            Softmax(dim=1)
        )

    def forward(self, x, explanation_mask=None):
        if explanation_mask is None:
            raise RuntimeError(f"explanation_mask must be provided for {self.__class__.__name__}")

        x = self.dropout(x, explanation_mask)
        for layer in self.features:
            if isinstance(layer, MaxPool2d):
                x, indices = layer(x)
            else:
                x = layer(x)
        x = self.classifier(x)
        return x