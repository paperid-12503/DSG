import torch
from PIL import Image
import numpy as np
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import warnings
import copy
warnings.filterwarnings("ignore")

def crop_foreground(image, mask, bias=10):
    image = np.array(image)
    mask = np.array(mask)
    mask = mask.astype(bool)

    foreground_pixels = np.column_stack(np.where(mask))

    if foreground_pixels.size == 0:
        raise ValueError("no foreground pixels found in mask")

    (y_min, x_min) = (np.min(foreground_pixels, axis=0))
    (y_max, x_max) = (np.max(foreground_pixels, axis=0))

    y_min = max(0, y_min - bias)
    x_min = max(0, x_min - bias)
    y_max = min(image.shape[0], y_max + bias)
    x_max = min(image.shape[1], x_max + bias)

    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    foreground_background = np.zeros_like(cropped_image)
    foreground_background[cropped_mask] = cropped_image[cropped_mask]

    cropped_image_pil = Image.fromarray(cropped_image)
    cropped_mask_pil = Image.fromarray(cropped_mask.astype(np.uint8) * 255)
    foreground_background_pil = Image.fromarray(foreground_background)

    return cropped_image_pil, cropped_mask_pil, foreground_background_pil


def transform_matrix(matrix, max=2, sigma=100, if_CDF=False):
    matrix = copy.deepcopy(matrix)
    matrix = matrix.flatten()
    # mu = torch.max(matrix[matrix>0]) / 2
    # mu = matrix.mean()

    if if_CDF:
        R = torch.exp((matrix) ** 2 / (2 * sigma ** 2))
        result = (R - R.min() + 1e-5) / (R.max() - R.min() + 1e-5)
    else:
        result = (matrix - matrix.min()) / (matrix.max() - matrix.min())
    
    matrix = max * result
    # matrix[matrix>0] = cdf_result
    matrix = torch.cat([torch.ones(1), matrix]).unsqueeze(dim=0)

    add_tensor = torch.zeros(size=(196, 197))
    matrix_1 = torch.cat([matrix, add_tensor], dim=0)
    matrix_2 = torch.cat([torch.zeros(size=(1, 197)), (matrix / max -0.5).expand(196, -1)])

    return matrix_1, matrix_2


class VisionEncoder_attn(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        visual = clip_model.visual
        self.ln_pre = visual.ln_pre
        self.transformer = visual.transformer.resblocks
        self.ln_post = visual.ln_post
        self.proj = visual.proj
        self.class_embedding = visual.class_embedding
        self.dtype = clip_model.dtype
        self.conv1 = clip_model.visual.conv1
        self.positional_embedding = clip_model.visual.positional_embedding
        # self.cs_layers = range(6, 12)
        self.post_layers = range(6, 12)
        self.pre_layers = range(0, 6)
        self.ratios = 0.1

    def forward(self, x: torch.Tensor, matrix_1, matrix_2):
        x = self.conv1(x.half())  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # <tokens, bs, dim>
    
        # Again combine the inputs, so nn.sequential can work
        attns = []
        for layer_idx, layer in enumerate(self.transformer):
            if layer_idx in self.post_layers:
                x, attn = layer(x, matrix_1)
                attns.append(attn.squeeze(dim=0))
            elif layer_idx in self.pre_layers:
                x, attn = layer(x, matrix_2)
                attns.append(attn.squeeze(dim=0))
            else:
                x, attn = layer(x, None)
                attns.append(attn.squeeze(dim=0))
    
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x, torch.stack(attns, dim=0)