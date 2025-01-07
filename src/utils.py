import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy

# TODO: remove this class or replace variables with such as `model`, `optimizer`, `loss_fn` etc.
class ModelConfig:
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 5

    def __init__(self, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=EPOCHS):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

class XILUtils:
    @staticmethod
    def define_device() -> str:
        """
        Determines the appropriate device for PyTorch computations.

        This function checks the availability of CUDA and MPS (Metal Performance Shaders)
        and returns the most suitable device for computation. If neither CUDA nor MPS
        is available, it defaults to using the CPU.

        Returns:
            str: A string representing the device to be used for computations.
                 Possible values are "cuda", "mps", or "cpu".

        """
        device: str  = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        return device

    @staticmethod
    def rrr_train_loop(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, optimizer: nn.Module,
                       batch_size: int, device: str):
        size = len(dataloader.dataset)
        interval = size // 10

        model.train()
        for batch, (X, y, A) in enumerate(dataloader):
            # move X and y to device
            X, y, A = X.to(device), y.to(device), A.to(device)
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y, A)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (batch * batch_size) % interval < batch_size:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    @staticmethod
    def rrr_test_loop(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, num_classes: int, device: str):
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        # with torch.no_grad(): prevents calculating gradients, which does not suit our rrr loss function
        for X, y, A in dataloader:
            # move X, y to device
            X, y, A = X.to(device), y.to(device), A.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y, A).item()
            correct += (F.one_hot(pred.argmax(1), num_classes=num_classes) == y).type(
                torch.float).sum().item() / num_classes

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    @staticmethod
    def train_loop(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, loss_fn, optimizer, model_config: ModelConfig, device: str):

        size = len(dataloader.dataset)

        model.train()
        for batch, (X, y) in enumerate(dataloader):
            # move X and y to device
            X, y = X.to(device), y.to(device)
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * model_config.batch_size + len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    # TODO: review how it is used in `pytorch.ipynb` and why I was forced to rewrite it in `08_MNIST.ipynb`
    @staticmethod
    def test_loop(dataloader, model, loss_fn, device):
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                # move X, y to device
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    # Explainers

    @staticmethod
    def apply_gradcam(model, target_layers, n_examples, dataset, num_classes, device, shuffle_ds=False, batch_num=0, batch_size=5, guided_backprop=False):
        # Get batch
        batch = XILUtils._get_batch_from_dataset(dataset, batch_num, batch_size, shuffle_ds)

        # batch data
        examples = batch[0]
        targets = batch[1]

        # Process all examples
        images = []
        cam_images = []
        grayscale_maps = []
        predictions = []
        certainties = []
        is_correct = []

        gb_model = None
        if guided_backprop:
            gb_model = GuidedBackpropReLUModel(model=model, device=device)

        model.eval()
        for index in range(n_examples):
            example = examples[index]
            target = targets[index]

            # Get prediction
            prediction_probs = model(example.unsqueeze(0).to(device))
            prediction = torch.zeros(num_classes)
            prediction[prediction_probs.argmax()] = 1
            certainty = prediction_probs.max().item() * 100

            predictions.append(prediction_probs.argmax().item())
            certainties.append(certainty)
            is_correct.append(all(prediction == target))

            print(f"\nExample {index}:")
            print(f"Shape of example: {example.shape}")
            print(f"Target of example: {target}")
            print(f"Predicted target: {prediction} with {certainty:.3f}% certainty. Correct? {all(prediction == target)}")

            # Prepare image
            img: numpy.ndarray = example.reshape((28, 28, 1)).repeat(1, 1, 3).numpy()
            images.append(img)

            # Generate GradCAM
            with GradCAM(model=model, target_layers=target_layers) as cam:
                input_tensor = example.unsqueeze(0)
                grayscale_cam = cam(input_tensor=input_tensor, targets=None, aug_smooth=False, eigen_smooth=False)
                grayscale_cam = numpy.squeeze(grayscale_cam, axis=0) # squeeze for example from (1, 28, 28) to (28, 28)
                if guided_backprop:
                    grayscale_cam *= numpy.squeeze(gb_model(input_tensor, target_category=None), axis=-1)

                grayscale_maps.append(grayscale_cam)

            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=False)
            cam_images.append(cam_image)

        return dict(images=images, cam_images=cam_images, grayscale_maps=grayscale_maps, predictions=predictions,
                    certainties=certainties, is_correct=is_correct)


    @staticmethod
    def plot_grad_cam(gradcam_aggregated_dict, labels, n_examples, plt):
        predictions = gradcam_aggregated_dict["predictions"]
        certainties = gradcam_aggregated_dict["certainties"]
        is_correct = gradcam_aggregated_dict["is_correct"]
        cam_images = gradcam_aggregated_dict["cam_images"]
        images = gradcam_aggregated_dict["images"]
        grayscale_maps = gradcam_aggregated_dict["grayscale_maps"]
        # Plot all examples
        fig, axes = plt.subplots(n_examples, 3, figsize=(18, 6 * n_examples))
        if n_examples == 1:
            axes = axes.reshape(1, -1)

        for idx in range(n_examples):
            # Create title with prediction and certainty
            pred_title = f'Predicted: {labels[predictions[idx]]} ({certainties[idx]:.1f}%). Prediction is {"" if is_correct[idx] else "NOT "}correct'

            # First subplot: CAM Overlay
            axes[idx, 0].imshow(cam_images[idx], cmap='viridis')
            axes[idx, 0].set_title(f'CAM Overlay\n{pred_title}')
            axes[idx, 0].axis('off')

            # Second subplot: Original Image
            axes[idx, 1].imshow(images[idx], cmap='viridis')
            axes[idx, 1].set_title(f'Original\n{pred_title}')
            axes[idx, 1].axis('off')

            # Third subplot: Attention Map
            axes[idx, 2].imshow(grayscale_maps[idx])
            axes[idx, 2].set_title(f'Attention Map\n{pred_title}')
            axes[idx, 2].axis('off')

        plt.tight_layout()
        plt.show()


    @staticmethod
    def apply_and_show_gradcam(model, target_layers, n_examples, dataset, num_classes, labels, plt, device, shuffle_ds=False, batch_num=0, batch_size=5, guided_backprop=False):
        # Calculate GradCAM
        aggregated_gradcam_dict = XILUtils.apply_gradcam(model, target_layers, n_examples, dataset, num_classes, device=device, batch_num=batch_num,
                               batch_size=batch_size, shuffle_ds=shuffle_ds, guided_backprop=guided_backprop)

        XILUtils.plot_grad_cam(
            aggregated_gradcam_dict,
            labels,
            n_examples,
            plt=plt
        )

    # Method helpers
    @staticmethod
    def _get_batch_from_dataset(dataset, batch_num, batch_size, shuffle_ds):
        # dataloader
        gradcam_dataloader = DataLoader(dataset, batch_size, shuffle=shuffle_ds)
        dataloader_iterator = iter(gradcam_dataloader)
        batch = next(dataloader_iterator)
        for i in range(batch_num):
            if i == batch_num - 1:
                batch = next(dataloader_iterator)
            else:
                next(dataloader_iterator)

        return batch
