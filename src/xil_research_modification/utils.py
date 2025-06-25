import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel as nativeGuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from . import guided_backprop

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
    def train_loop(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, loss_fn, optimizer, device: str):

        size = len(dataloader.dataset)
        batch_size = dataloader.batch_size

        model.train()
        for batch_i, (X, y) in enumerate(dataloader):
            # move X and y to device
            X, y = X.to(device), y.to(device)
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_i % 100 == 0:
                loss, current = loss.item(), batch_i * batch_size + len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    standard_aggregation = lambda input_tensor: torch.argmax(input_tensor, dim=-1)
    @staticmethod
    def test_loop(dataloader, model, loss_fn, device, prediction_agg=standard_aggregation, target_agg=standard_aggregation):
        # Prediction and Target aggregations should be such that there is a single number for correctness of the item
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
                correct += (prediction_agg(pred) == target_agg(y)).type(torch.float).sum().item()

            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    # Explainers

    @staticmethod
    def apply_gradcam(model, target_layers, dataset, device, shuffle_ds=False, batch_num=0, batch_size=5, guided_gradcam=False, **kwargs):
        model.eval()
        # Get batch
        batch = XILUtils._get_batch_from_dataset(dataset, batch_num, batch_size, shuffle_ds)

        # batch data
        examples = batch[0]
        targets = batch[1]

        # Process all examples
        images = []
        cam_images = []
        predictions = []
        certainties = []
        is_correct = []

        if guided_gradcam:
            grayscale_maps = XILUtils.guided_gradcam_explain(
                examples, targets, model, device, target_layers,
                target_classification_criterium=kwargs.get(
                    "target_classification_criterium"),
                aug_smooth=kwargs.get("aug_smooth", False),
                eigen_smooth=kwargs.get("eigen_smooth", False),
                normalize=kwargs.get("normalize", True)
            )
        else:
            grayscale_maps = XILUtils.gradcam_explain(
                examples, model, device, target_layers,
                target_classification_criterium=kwargs.get(
                    "target_classification_criterium"),
                aug_smooth=kwargs.get("aug_smooth", False),
                eigen_smooth=kwargs.get("eigen_smooth", False),
                cam_class=kwargs.get("cam_class", GradCAM)
            )

        grayscale_maps = np.transpose(grayscale_maps.detach().cpu().numpy(), (0, 2, 3, 1))
        grayscale_maps = list(grayscale_maps)

        for index in range(batch_size):
            example = examples[index]
            target = targets[index]

            # Get prediction
            prediction_probs = F.softmax(model(example.unsqueeze(0).to(device)), dim=-1)
            prediction = torch.nn.functional.one_hot(prediction_probs.argmax(),
                                                     num_classes=prediction_probs.shape[-1])
            certainty = prediction_probs.max().item() * 100

            predictions.append(prediction_probs.argmax().item())
            certainties.append(certainty)
            is_correct.append(prediction.equal(target.to(device)))

            print(f"\nExample {index}:")
            print(f"Shape of example: {example.shape}")
            print(f"Target of example: {target}")
            print(f"Predicted target: {prediction} with {certainty:.3f}% certainty. Correct? {all(prediction == target)}")

            # Prepare image
            img_np = example.cpu().numpy()

            if img_np.ndim == 3:
                img_np = np.transpose(img_np, (1, 2, 0))  # CxHxW → HxWxC

            if img_np.shape[2] == 1:
                img_np = np.repeat(img_np, 3, axis=2)  # Grayscale → RGB

            images.append(img_np)

            cam_image = show_cam_on_image(img_np, grayscale_maps[index], use_rgb=False)
            cam_images.append(cam_image)

        return dict(images=images, cam_images=cam_images, grayscale_maps=grayscale_maps, predictions=predictions,
                    certainties=certainties, is_correct=is_correct)

    @staticmethod
    def plot_three_columns(plt, columns, n_examples, titles=None, cmap='viridis'):
        fig, axes = plt.subplots(n_examples, 3, figsize=(18, 6 * n_examples))
        if n_examples == 1:
            axes = axes.reshape(1, -1)

        column_1, column_2, column_3 = columns

        for idx in range(n_examples):
            # First subplot: CAM Overlay
            axes[idx, 0].imshow(column_1[idx], cmap=cmap)
            if titles:
                axes[idx, 0].set_title(titles[0][idx])
            axes[idx, 0].axis('off')

            # Second subplot: Original Image
            axes[idx, 1].imshow(column_2[idx], cmap=cmap)
            if titles:
                axes[idx, 1].set_title(titles[1][idx])
            axes[idx, 1].axis('off')

            # Third subplot: Attention Map
            axes[idx, 2].imshow(column_3[idx])
            if titles:
                axes[idx, 2].set_title(titles[2][idx])
            axes[idx, 2].axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_grad_cam(gradcam_aggregated_dict, labels, n_examples, plt):
        predictions = gradcam_aggregated_dict["predictions"]
        certainties = gradcam_aggregated_dict["certainties"]
        is_correct = gradcam_aggregated_dict["is_correct"]
        cam_images = gradcam_aggregated_dict["cam_images"]
        images = gradcam_aggregated_dict["images"]
        grayscale_maps = gradcam_aggregated_dict["grayscale_maps"]
        # Plot all examples
        titles_1 = []
        titles_2 = []
        titles_3 = []
        columns = [cam_images, images, grayscale_maps]
        for i in range(n_examples):
            # Create title with prediction and certainty
            pred_title = f'Predicted: {labels[predictions[i]]} ({certainties[i]:.1f}%). Prediction is {"" if is_correct[i] else "NOT "}correct'

            titles_1.append(f'CAM Overlay\n{pred_title}')
            titles_2.append(f'Original\n{pred_title}')
            titles_3.append(f'Attention Map\n{pred_title}')
        XILUtils.plot_three_columns(plt, columns, n_examples, (titles_1, titles_2, titles_3))


    @staticmethod
    def apply_and_show_gradcam(model, target_layers, dataset, labels, plt, device=torch.device('cpu'),
                               shuffle_ds=False, batch_num=0, batch_size=5, guided_gradcam=False):
        # Calculate GradCAM
        aggregated_gradcam_dict = XILUtils.apply_gradcam(model, target_layers, dataset, device=device, batch_num=batch_num,
                               batch_size=batch_size, shuffle_ds=shuffle_ds, guided_gradcam=guided_gradcam)

        XILUtils.plot_grad_cam(
            aggregated_gradcam_dict,
            labels,
            batch_size,
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

    @staticmethod
    def minmax_normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize a tensor using min-max normalization per batch element.

        Args:
            tensor (torch.Tensor): The input tensor of shape (B, ...).

        Returns:
            torch.Tensor: The normalized tensor with each batch element scaled individually.
        """
        # Compute min and max over dimensions 1,2,...,n (all except the batch dimension)
        dims = tuple(range(1, tensor.ndim))
        min_val = tensor.amin(dim=dims, keepdim=True)
        max_val = tensor.amax(dim=dims, keepdim=True)

        # Compute the denominator and avoid division by zero
        denom = max_val - min_val
        # Replace zeros in denom with ones to avoid division by zero
        denom[denom == 0] = 1.0

        return (tensor - min_val) / denom

    @staticmethod
    def subtract_mean_per_batch(tensor: torch.Tensor) -> torch.Tensor:
        """
        Subtract the mean from each batch element individually.

        Args:
            tensor (torch.Tensor): The input tensor of shape (B, ...).

        Returns:
            torch.Tensor: The tensor with the mean subtracted per batch element.
        """
        # Compute mean over dimensions 1,2,...,n (all dimensions except the batch dimension)
        dims = tuple(range(1, tensor.ndim))
        mean_val = tensor.mean(dim=dims, keepdim=True)

        # Subtract the mean for each batch element
        return tensor - mean_val

    @staticmethod
    def subtract_batch_mean_diff(tensor: torch.Tensor) -> torch.Tensor:
        """
        For a tensor of shape (B, ...), compute the mean for each batch element.
        Then, take the minimum mean from these batch means, compute the difference
        between each batch mean and this minimum, and subtract that difference from
        the corresponding batch element in the original tensor.

        Args:
            tensor (torch.Tensor): The input tensor of shape (B, ...).

        Returns:
            torch.Tensor: The resulting tensor after subtracting the per-batch differences.
        """
        # Compute the mean for each batch element over all other dimensions
        dims = tuple(range(1, tensor.ndim))
        means = tensor.mean(dim=dims, keepdim=True)  # shape: (B, 1, 1, ..., 1)

        # Flatten the batch means to compute the overall minimum mean
        batch_means = means.view(tensor.shape[0])
        min_mean = batch_means.min()

        # Compute the difference for each batch element: mean - min_mean
        diff = means - min_mean  # shape is (B, 1, 1, ..., 1)

        # Subtract the computed difference from the original tensor (broadcasting over the remaining dimensions)
        return tensor - diff

    @staticmethod
    def gradcam_explain(
            x_batch, model, device, target_layers,
            target_classification_criterium=None,
            aug_smooth=False,
            eigen_smooth=False,
            cam_class=GradCAM):
        model.eval()

        with cam_class(model=model, target_layers=target_layers) as cam:
            grayscale_cam = torch.tensor(
                cam(
                    input_tensor=x_batch, targets=target_classification_criterium, aug_smooth=aug_smooth,
                    eigen_smooth=eigen_smooth)
                ,
                device=device
            )

        return grayscale_cam.unsqueeze(1)

    @staticmethod
    def guided_backprop_explain(x_batch, target_batch, model, device, normalize=True):
        model.eval()
        gb_model = guided_backprop.GuidedBackpropagation(model=model, device=device)
        gb_model_out = gb_model(x_batch, target_batch)
        if normalize:
            gb_model_out = XILUtils.minmax_normalize_tensor(gb_model_out)

        return gb_model_out

    @staticmethod
    def guided_gradcam_explain(x_batch, target_batch, model, device, target_layers,
                               aug_smooth=False, eigen_smooth=False,
                               target_classification_criterium=None,
                               normalize=True):
        model.eval()

        grayscale_cam = XILUtils.gradcam_explain(x_batch, model, device, target_layers,
                                                 aug_smooth=aug_smooth,
                                                 eigen_smooth=eigen_smooth,
                                                 target_classification_criterium=target_classification_criterium)
        gb_model_out = XILUtils.guided_backprop_explain(x_batch, target_batch, model, device, normalize=normalize)
        # !!! Normalizing output of guided backpropagation is having a positive effect on visualising
        grayscale_cam *= gb_model_out

        return XILUtils.minmax_normalize_tensor(grayscale_cam)

    @staticmethod
    def create_explanation(input_tensor: torch.Tensor, binary_mask_tensor: torch.Tensor, target_tensor, explanation_type: str="guided_gradcam", **kwargs):
        # get explanation specific parameters

        if explanation_type == "guided_gradcam":
            # Guided GradCAM
            target_layers = kwargs.get("target_layers")
            device = kwargs.get("device")
            model = kwargs.get("model")
            threshold = kwargs.get("threshold", 0.5)

            if target_layers is None:
                raise ValueError("Target layers must be provided for Guided GradCAM.")
            if device is None:
                raise ValueError("Device must be provided for Guided GradCAM.")
            if model is None:
                raise ValueError("Model must be provided for Guided GradCAM.")

            guided_gradcam_explanation = XILUtils.guided_gradcam_explain(input_tensor, target_tensor, model, device, target_layers).unsqueeze(1)
            result = guided_gradcam_explanation * binary_mask_tensor > threshold
        else:
            raise NotImplementedError(f"Explanation type '{explanation_type}' is not implemented.")
        return result