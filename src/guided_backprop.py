import pytorch_grad_cam.guided_backprop as orig_gb
from pytorch_grad_cam.utils.find_layers import replace_all_layer_type_recursive
import torch

class GuidedBackpropReLUModel(orig_gb.GuidedBackpropReLUModel):
    def __init__(self, model, device):
        super().__init__(model, device)

    def __call__(self, input_img, target_category=None):
        replace_all_layer_type_recursive(self.model,
                                         torch.nn.ReLU,
                                         orig_gb.GuidedBackpropReLUasModule())


        input_img = input_img.to(self.device)

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category is None:
            target_category = lambda i: torch.argmax(i, dim=1)

        loss = torch.sum(output[target_category(output)]).cpu()
        loss.backward(retain_graph=True)

        output: torch.Tensor = input_img.grad
        # output = output.permute(0, 2, 3, 1) numpy adaptation

        replace_all_layer_type_recursive(self.model,
                                         orig_gb.GuidedBackpropReLUasModule,
                                         torch.nn.ReLU())

        return output
