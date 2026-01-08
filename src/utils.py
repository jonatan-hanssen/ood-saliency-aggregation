from typing import Callable

import math
import captum
import numpy as np
import torch
import os
import gdown
from sklearn.linear_model import LinearRegression
import shutil

from .resnet18_32_patched import ResNet18_32x32

from openood.networks import (
    ResNet18_224x224,
    ResNet50,
)

URLS = {
    'cifar10': '1byGeYxM_PlLjT72wZsMQvP6popJeWBgt',
    'cifar100': '1s-1oNrRtmA0pGefxXJOUVRYpaoAML0C-',
    'imagenet200': '1ddVmwc8zmzSjdLUO84EuV4Gz1c7vhIAs',
    'imagenet': '15PdDMNRfnJ7f2oxW6lI-Ge4QJJH3Z0Fy',
}

checkpoint_paths = {
    'cifar10': 'cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt',
    'cifar100': 'cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt',
    'imagenet200': 'imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt',
    'imagenet': '15PdDMNRfnJ7f2oxW6lI-Ge4QJJH3Z0Fy',
}

STORE_PATH = './results/checkpoints'


def get_network(id_name: str, device_str: str):
    device = torch.device(device_str)
    match id_name:
        case 'cifar10':
            net = ResNet18_32x32(num_classes=10)
        case 'imagenet':
            net = ResNet50(num_classes=1000)
        case 'imagenet200':
            net = ResNet18_224x224(num_classes=200)
        case 'cifar100':
            net = ResNet18_32x32(num_classes=100)
        case _:
            raise ValueError('No such dataset')

    if not os.path.exists(os.path.join(STORE_PATH, checkpoint_paths[id_name])):
        print('Downloading pretrained weights...')
        os.makedirs(STORE_PATH, exist_ok=True)
        file_path = os.path.join(STORE_PATH, 'temp.zip')
        gdown.download(id=URLS[id_name], output=file_path)
        shutil.unpack_archive(file_path, STORE_PATH)
        os.remove(file_path)

    net.load_state_dict(
        torch.load(
            os.path.join(STORE_PATH, checkpoint_paths[id_name]), map_location=device
        )
    )

    net = net.to(device)
    net.eval()
    return net


def get_aggregate_function(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get an aggregation function over the batch dimension based on the provided name.

    Args:
        name (str): The name of the aggregation function. Supported names are:
            - 'mean': Computes the sum of the data.
            - 'median': Computes the median of the data.
            - 'norm': Computes the vector norm of the data.
            - 'range': Computes the range (max - min) of the data.
            - 'max': Computes the maximum value of the data.
            - 'q3': Computes the third quartile (75th percentile) of the data.
            - 'cv': Computes the coefficient of variation (std / mean) of the data.
            - 'rmd': Computes the relative mean deviation of the data.
            - 'qcd': Computes the quartile coefficient of dispersion of the data.

    Returns:
        Callable: A function that takes a tensor as input and returns the aggregated result.

    Raises:
        ValueError: If the provided name is not supported.
    """
    match name:
        case 'mean':
            return lambda data: torch.mean(data, dim=-1)
        case 'median':
            return lambda data: torch.median(data, dim=-1)[0]
        case 'norm':
            return lambda data: torch.linalg.vector_norm(data, dim=-1)
        case 'range':
            return lambda data: torch.max(data, dim=-1)[0] - torch.min(data, dim=-1)[0]
        case 'max':
            return lambda data: torch.max(data, dim=-1)[0]
        case 'q3':
            return lambda data: np.quantile(data, 0.75, axis=-1)
        case 'cv':
            return lambda data: torch.std(data, dim=-1) / torch.where(
                torch.mean(data, dim=-1) != 0, torch.mean(data, dim=-1), 1e-10
            )
        case 'rmd':
            return rmd
        case 'qcd':
            return qcd
        case _:
            raise ValueError(f'No such aggregate function: {name}')


def get_saliency_generator(
    name: str,
    net: torch.nn.Module,
) -> Callable:
    if name == 'gradcam':
        cam_wrapper = GradCAMWrapper(
            model=net, target_layer=net.layer4[-1], normalize=False
        )
        generator_func = cam_wrapper

    elif name == 'lime':
        generator_func = lambda data: lime_explanation(
            net, data, repeats=4, do_relu=False
        )

    elif name == 'integratedgradients':

        def generator_func(data):
            targets = torch.argmax(net(data), dim=-1)
            lrp = captum.attr.IntegratedGradients(net)

            attributions = lrp.attribute(data, target=targets)
            attributions = attributions.detach().cpu()

            return attributions

    elif name == 'gbp':

        def generator_func(data):
            targets = torch.argmax(net(data), dim=-1)
            gbp = captum.attr.GuidedBackprop(net)

            attributions = gbp.attribute(data, target=targets)

            attributions = attributions.detach().cpu()

            return attributions

    elif name == 'occlusion':

        def generator_func(data):
            targets = torch.argmax(net(data), dim=-1)
            lrp = captum.attr.Occlusion(net)

            block_size = data.shape[-1] // 4

            attributions = lrp.attribute(
                data,
                target=targets,
                sliding_window_shapes=(3, block_size, block_size),
                strides=(3, block_size, block_size),
            )

            attributions = attributions.sum(dim=1)
            attributions = (
                torch.nn.MaxPool2d(1, block_size)(attributions).detach().cpu()
            )

            return attributions

    else:
        raise TypeError('No such generator')

    return generator_func


class GradCAMWrapper(torch.nn.Module):
    def __init__(
        self,
        model=None,
        target_layer=None,
        do_relu=False,
        subtype=None,
        normalize=False,
    ):
        super().__init__()
        self.model = model
        self.target_layer = target_layer
        self.do_relu = do_relu
        self.normalize = normalize

        self.subtype = subtype

        self.grads = None
        self.acts = None

        self.outputs = None

        self.handles = list()

        self.handles.append(
            self.target_layer.register_full_backward_hook(self.grad_hook)
        )
        self.handles.append(self.target_layer.register_forward_hook(self.act_hook))

    def grad_hook(self, module, grad_input, grad_output):
        self.grads = grad_output[0]

    def act_hook(self, module, input, output):
        self.acts = output

    def forward(self, x, return_feature=False, class_to_backprog=None):
        batch_size = x.shape[0]

        if return_feature:
            preds, feature = self.model(x, return_feature=True)

        else:
            preds = self.model(x)

        self.outputs = preds

        self.model.zero_grad(set_to_none=True)

        if class_to_backprog is None:
            idxs = torch.argmax(preds, dim=1)
        else:
            idxs = torch.ones(batch_size, dtype=int) * class_to_backprog

        # backward pass, this gets gradients for each prediction
        torch.sum(preds[torch.arange(batch_size), idxs]).backward()

        average_gradients = self.grads.mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1)
        saliency = self.acts * average_gradients

        saliency = torch.sum(saliency, dim=1)
        if self.do_relu:
            saliency = torch.nn.functional.relu(saliency)

        if self.normalize:
            mins = saliency.min(dim=-1)[0].min(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
            saliency -= mins
            maxes = saliency.max(dim=-1)[0].max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
            saliency /= maxes

        if return_feature:
            return saliency.cpu().detach(), feature

        else:
            return saliency.cpu().detach()

    def __del__(self):
        for handle in self.handles:
            handle.remove()


def rmd(saliencies):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    ginies = list()

    for row in saliencies:
        array = row.numpy()
        array = array.flatten()  # all values are treated equally, arrays must be 1d
        if np.amin(array) < 0:
            array -= np.amin(array)  # values cannot be negative
        array += 0.0000001  # values cannot be 0
        array = np.sort(array)  # values must be sorted
        index = np.arange(1, array.shape[0] + 1)  # index per array element
        n = array.shape[0]  # number of array elements
        gini = (np.sum((2 * index - n - 1) * array)) / (
            n * np.sum(array)
        )  # Gini coefficient
        ginies.append(gini)

    return torch.tensor(ginies) * 2


def lime_explanation(
    net,
    batch,
    perturbations=100,
    mask_prob=0.5,
    repeats=8,
    kernel_width=0.25,
    do_relu=False,
):
    preds = net(batch)
    max_pred_ind = torch.argmax(preds, dim=1)

    kernel = lambda distances: torch.sqrt(torch.exp(-(distances**2) / kernel_width**2))

    all_betas = list()

    for image, pred_label in zip(batch, max_pred_ind):
        images = image.unsqueeze(0).expand(perturbations, -1, -1, -1)

        masked_images, masks = mask_image(images, repeats=repeats, mask_prob=mask_prob)

        with torch.no_grad():
            network_preds = net(masked_images)[:, pred_label]

        original = torch.ones((1, masks.shape[-1]))

        cos = torch.nn.CosineSimilarity(dim=1)
        distances = 1 - cos(masks.float(), original.float())

        weights = kernel(distances)

        regressor = LinearRegression()

        regressor.fit(
            numpify(masks), numpify(network_preds), sample_weight=numpify(weights)
        )

        betas = torch.tensor(regressor.coef_).unsqueeze(0)

        all_betas.append(betas)

    all_betas = torch.cat(all_betas, dim=0)
    if do_relu:
        all_betas = torch.nn.functional.relu(all_betas)

    return all_betas.reshape(-1, repeats, repeats)


def numpify(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def create_batch_masks(segmentation, batch_segment_values):
    # segmentation is of shape (H, W)
    # batch_segment_values is of shape (N, num_segments)

    # Get batch size and number of segments
    N, num_segments = batch_segment_values.shape
    H, W = segmentation.shape

    # Initialize a mask tensor with ones
    batch_masks = torch.ones((N, H, W), dtype=torch.float32)

    # Create an expanded segmentation for comparison
    segmentation_expanded = segmentation.unsqueeze(0).expand(N, -1, -1)

    for segment_id in range(1, num_segments + 1):
        # Create a mask to determine where the zeroing should occur
        segment_mask = batch_segment_values[:, segment_id - 1].unsqueeze(1).unsqueeze(2)
        zero_mask = (segmentation_expanded == segment_id) * (segment_mask == 0)
        batch_masks[zero_mask] = 0

    return batch_masks


def occlusion(net, batch, repeats=8, do_relu=False):
    preds = net(batch)
    preds = torch.nn.functional.softmax(preds, dim=1)

    max_pred, max_pred_ind = torch.max(preds, dim=1)

    block_size = batch.shape[-1] // repeats

    saliencies = list()

    for image, pred_value, pred_label in zip(batch, max_pred, max_pred_ind):
        images = image.unsqueeze(0).expand(repeats**2, -1, -1, -1)

        masked_images = occlude_images(images, block_size=block_size)

        with torch.no_grad():
            network_preds = net(masked_images)
            network_preds = torch.nn.functional.softmax(network_preds, dim=1)[
                :, pred_label
            ]
        saliencies.append((pred_value.detach() - network_preds.detach()).unsqueeze(0))

    saliencies = torch.cat(saliencies, dim=0)

    if do_relu:
        saliencies = torch.nn.functional.relu(saliencies)

    return saliencies.reshape(-1, repeats, repeats)


def mask_image(batch, repeats=8, mask_prob=0.5):
    batch_size = batch.shape[0]
    h = batch.shape[-1]

    block_size = math.ceil(h / repeats)

    masks = (torch.rand(batch_size, repeats**2) > mask_prob).to(int)

    # Prepare the mask array
    mask_array = (
        masks.reshape((batch_size, repeats, repeats))
        .repeat_interleave(block_size, dim=1)
        .repeat_interleave(block_size, dim=2)
    )

    mask_array = mask_array[:, :h, :h]

    mask_array = mask_array.unsqueeze(1)

    masked_images = batch * mask_array.to(batch.device)

    return masked_images, masks


def occlude_images(batch, block_size=4):
    batch_size = batch.shape[0]
    h = batch.shape[-1]

    repeats = h // block_size

    masks = torch.where(torch.eye(repeats**2) == 1, 0, 1)

    # Prepare the mask array
    mask_array = (
        masks.reshape((batch_size, repeats, repeats))
        .repeat_interleave(block_size, dim=1)
        .repeat_interleave(block_size, dim=2)
    )

    mask_array = mask_array.unsqueeze(1)

    masked_images = batch * mask_array.to(batch.device)

    return masked_images


def qcd(data):
    q1 = np.quantile(data, 0.25, axis=-1)
    q3 = np.quantile(data, 0.75, axis=-1)

    numerator = q3 - q1
    denominator = q3 + q1

    new_denominator = denominator[:]
    new_denominator[np.where(denominator == 0)] = 1
    numerator[np.where(denominator == 0)] = float('inf')

    return numerator / new_denominator
