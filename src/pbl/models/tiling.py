from typing import Callable, List
import numpy as np
import scipy.signal
import torch
from torch.nn import functional as func
from tqdm import tqdm


WINDOW_CACHE = dict()


def _spline_window(window_size: int, power: int = 2) -> np.ndarray:
    """Generates a 1-dimensional spline of order 'power' (typically 2), in the designated
    window.
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2

    Args:
        window_size (int): size of the interested window
        power (int, optional): Order of the spline. Defaults to 2.

    Returns:
        np.ndarray: 1D spline
    """
    intersection = int(window_size / 4)
    wind_outer = (abs(2 * (scipy.signal.triang(window_size))) ** power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2 * (scipy.signal.triang(window_size) - 1)) ** power) / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


def _spline_2d(window_size: int, power: int = 2) -> torch.Tensor:
    """Makes a 1D window spline function, then combines it to return a 2D window function.
    The 2D window is useful to smoothly interpolate between patches.

    Args:
        window_size (int): size of the window (patch)
        power (int, optional): Which order for the spline. Defaults to 2.

    Returns:
        np.ndarray: numpy array containing a 2D spline function
    """
    # Memorization to avoid remaking it for every call
    # since the same window is needed multiple times
    global WINDOW_CACHE
    key = f"{window_size}_{power}"
    if key in WINDOW_CACHE:
        wind = WINDOW_CACHE[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 1), 1)  # SREENI: Changed from 3, 3, to 1, 1
        wind = torch.from_numpy(wind * wind.transpose(1, 0, 2))
        WINDOW_CACHE[key] = wind
    return wind


def pad_image(image: torch.Tensor, tile_size: int, subdivisions: int) -> torch.Tensor:
    """Add borders to the given image for a "valid" border pattern according to "window_size" and "subdivisions".
    Image is expected as a numpy array with shape (width, height, channels).

    Args:
        image (torch.Tensor): input image, 3D channels-last tensor
        tile_size (int): size of a single patch, useful to compute padding
        subdivisions (int): amount of overlap, useful for padding

    Returns:
        torch.Tensor: same image, padded specularly by a certain amount in every direction
    """
    # compute the pad as (window - window/subdivisions)
    pad = int(round(tile_size * (1 - 1.0 / subdivisions)))
    batch = image.permute(2, 0, 1).unsqueeze(0)
    # pad the image with the computed amount
    pad_left = pad
    pad_top = pad
    pad_right = pad
    pad_bottom = pad 
    # Adjust padding to ensure the shape is divisible by pad
    pad_right += pad_right - ((image.shape[0] + pad_left + pad_right) % pad)
    pad_bottom += pad_bottom - ((image.shape[1] + pad_top + pad_bottom) % pad)
    # add pixels to pad_right and pad_bottom until the shape is divisible by tile_size/subdivisions
    padded_width = image.shape[0] + pad_left + pad_right
    padded_height = image.shape[1] + pad_top + pad_bottom
    if padded_width < padded_height:
        # add the difference to the right
        diff = padded_height - padded_width
        pad_right += diff
    elif padded_height < padded_width:
        # add the difference to the bottom
        diff = padded_width - padded_height
        pad_bottom += diff
    # pad the image with the computed amount
    padding = (pad_top, pad_bottom, pad_left, pad_right)
    batch = func.pad(batch, padding, mode="reflect")
    return batch.squeeze(0).permute(1, 2, 0), padding


def unpad_image(padded_image: torch.Tensor, tile_size: int, subdivisions: int, pads) -> torch.Tensor:
    """Reverts changes made by 'pad_image'. The same padding is removed, so tile_size and subdivisions
    must be coherent.

    Args:
        padded_image (torch.Tensor): image with padding still applied
        tile_size (int): size of a single patch
        subdivisions (int): subdivisions to compute overlap

    Returns:
        torch.Tensor: image without padding, 2D channels-last tensor
    """
    pad_top, pad_bottom, pad_left, pad_right = pads
    # crop the image left, right, top and bottom
    # get number of dimensions of padded_image
    n_dims = len(padded_image.shape)
    # if padded_image is 2d
    if n_dims == 2:
        result = padded_image[pad_left:-pad_right, pad_top:-pad_bottom]
    # if padded_image is 3d
    elif n_dims == 3:
        result = padded_image[:, pad_left:-pad_right, pad_top:-pad_bottom]
    else:
        raise ValueError(f"padded_image has {n_dims} dimensions, expected 2 or 3.")
    return result


def rotate_and_mirror(image: torch.Tensor) -> List[torch.Tensor]:
    """Duplicates an image with shape (h, w, channels) 8 times, in order
    to have all the possible rotations and mirrors of that image that fits the
    possible 90 degrees rotations. https://en.wikipedia.org/wiki/Dihedral_group

    Args:
        image (torch.Tensor): input image, already padded.

    Returns:
        List[torch.Tensor]: list of images, rotated and mirrored.
    """
    variants = []
    variants.append(image)
    variants.append(torch.rot90(image, k=1, dims=(0, 1)))
    variants.append(torch.rot90(image, k=2, dims=(0, 1)))
    variants.append(torch.rot90(image, k=3, dims=(0, 1)))
    image = torch.flip(image, dims=(0, 1))
    variants.append(image)
    variants.append(torch.rot90(image, k=1, dims=(0, 1)))
    variants.append(torch.rot90(image, k=2, dims=(0, 1)))
    variants.append(torch.rot90(image, k=3, dims=(0, 1)))
    return variants


def undo_rotate_and_mirror(variants: List[torch.Tensor]) -> torch.Tensor:
    """Reverts the 8 duplications provided by rotate and mirror.
    This restores the transformed inputs to the original position, then averages them.

    Args:
        variants (List[torch.Tensor]): D4 dihedral group of the same image

    Returns:
        torch.Tensor: averaged result over the given input.
    """
    origs = []
    origs.append(variants[0])
    origs.append(torch.rot90(variants[1], k=3, dims=(0, 1)))
    origs.append(torch.rot90(variants[2], k=2, dims=(0, 1)))
    origs.append(torch.rot90(variants[3], k=1, dims=(0, 1)))
    origs.append(torch.flip(variants[4], dims=(0, 1)))
    origs.append(torch.flip(torch.rot90(variants[5], k=3, dims=(0, 1)), dims=(0, 1)))
    origs.append(torch.flip(torch.rot90(variants[6], k=2, dims=(0, 1)), dims=(0, 1)))
    origs.append(torch.flip(torch.rot90(variants[7], k=1, dims=(0, 1)), dims=(0, 1)))
    return torch.mean(torch.stack(origs), axis=0)


def windowed_generator(padded_image: torch.Tensor, window_size: int, subdivisions: int, batch_size: int = None):
    """Generator that yield tiles grouped by batch size.
    Args:
        padded_image (np.ndarray): input image to be processed (already padded), supposed channels-first
        window_size (int): size of a single patch
        subdivisions (int): subdivision count on each patch to compute the step
        batch_size (int, optional): amount of patches in each batch. Defaults to None.

    Yields:
        Tuple[List[tuple], np.ndarray]: list of coordinates and respective patches as single batch array
    """
    step = window_size // subdivisions
    width, height, _ = padded_image.shape
    batch_size = batch_size or 1
    batch = []
    coords = []
    # step with fixed window on the image to build up the arrays
    for x in range(0, width - window_size + 1, step):
        for y in range(0, height - window_size + 1, step):
            coords.append((x, y))
            # extract the tile, place channels first for batch
            tile = padded_image[x : x + window_size, y : y + window_size]
            batch.append(tile.permute(2, 0, 1))
            # yield the batch once full and restore lists right after
            if len(batch) == batch_size:
                yield coords, torch.stack(batch)
                coords.clear()
                batch.clear()
    # handle last (possibly unfinished) batch
    if len(batch) > 0:
        yield coords, torch.stack(batch)


def reconstruct(canvas: torch.Tensor, tile_size: int, coords: List[tuple], predictions: torch.Tensor) -> torch.Tensor:
    """Helper function that iterates the result batch onto the given canvas to reconstruct
    the final result batch after batch.
    Args:
        canvas (torch.Tensor): container for the final image.
        tile_size (int): size of a single patch.
        coords (List[tuple]): list of pixel coordinates corresponding to the batch items
        predictions (torch.Tensor): array containing patch predictions, shape (batch, tile_size, tile_size, num_classes)

    Returns:
        torch.Tensor: the updated canvas, shape (padded_w, padded_h, num_classes)
    """
    for (x, y), patch in zip(coords, predictions):
        # get canvas number of dimensions
        n_dims = len(canvas.shape)
        # if canvas is 2d
        if n_dims == 2:
            canvas[x : x + tile_size, y : y + tile_size] += patch
        # if canvas is 3d
        elif n_dims == 3:
            canvas[:, x : x + tile_size, y : y + tile_size] += patch
        else:
            raise ValueError(f"Canvas has {n_dims} dimensions, expected 2 or 3.")
    return canvas


def predict_smooth_windowing(
    image: torch.Tensor,
    tile_size: int,
    subdivisions: int,
    prediction_fn: Callable,
    batch_size: int = None,
    channels_first: bool = False,
    mirrored: bool = False,
) -> np.ndarray:
    """Allows to predict a large image in one go, dividing it in squared, fixed-size tiles and smoothly
    interpolating over them to produce a single, coherent output with the same dimensions.
    Args:
        image (np.ndarray): input image, expected a 3D vector
        tile_size (int): size of each squared tile
        subdivisions (int): number of subdivisions over the single tile for overlaps
        prediction_fn (Callable): callback that takes the input batch and returns an output tensor
        batch_size (int, optional): size of each batch. Defaults to None.
        channels_first (int, optional): whether the input image is channels-first or not
        mirrored (bool, optional): whether to use dihedral predictions (every simmetry). Defaults to False.

    Returns:
        np.ndarray: numpy array with dimensions (w, h), containing smooth predictions
    """
    if channels_first:
        image = image.permute(1, 2, 0)
    width, height, _ = image.shape
    padded, pads = pad_image(image=image, tile_size=tile_size, subdivisions=subdivisions)
    padded_width, padded_height, _ = padded.shape
    padded_variants = rotate_and_mirror(padded) if mirrored else [padded]
    spline = _spline_2d(window_size=tile_size, power=2).to(image.device).squeeze(-1)
    results = []
    for img in padded_variants:
        canvas = torch.zeros((padded_width, padded_height), device=image.device)
        for coords, batch in windowed_generator(
            padded_image=img, window_size=tile_size, subdivisions=subdivisions, batch_size=batch_size
        ):
            # returns batch of channels-first, return to channels-last
            pred_batch = prediction_fn(batch)  # .permute(0, 2, 3, 1)
            # must be 3d for reconstruction to work
            # if it is 4d, check if the channel dimension is 1, then remove it
            if len(pred_batch.shape) == 4 and pred_batch.shape[1] == 1:
                pred_batch = pred_batch[:, 0]
            pred_batch = [tile * spline for tile in pred_batch]
            canvas = reconstruct(canvas, tile_size=tile_size, coords=coords, predictions=pred_batch)
        canvas /= subdivisions**2
        results.append(canvas)
    padded_result = undo_rotate_and_mirror(results) if mirrored else results[0]
    prediction = unpad_image(padded_result, tile_size=tile_size, subdivisions=subdivisions, pads=pads)
    return prediction[:width, :height]
import torch.nn.functional as F


def overlapping_predictions_torch(input, prediction_fn, size=256, stride=128, batch_size=2, val_in_cpu=False, verbose=False, out_dim=1):
    # check if stride is divisible by 1, if not raise an error, then cast to int
    if stride % 1 != 0:
        raise ValueError("stride must be divisible by 1")
    if input.dim() == 4:
        if input.shape[0] != 1:
            raise ValueError("batch size must be 1 in inference mode")
        input = input.squeeze(0)
    stride = int(stride)
    # store original input and target shape
    device = input.device
    tile_list = []
    coords_list = []
    # create a cosine window
    han = torch.from_numpy(np.hanning(size))
    window = torch.outer(han, han) + 1e-6  # to avoid division by zero
    # repeat the window for each channel
    window = window.repeat(out_dim, 1, 1)
    # compute the padding so that it is both bigger than stride and the padded image is divisible by stride
    pad_left = 0
    pad_right = 0
    # iterate from pad_right, adding 1 until (input.shape[2] + pad_left + pad_right) % stride == 0
    while (input.shape[2] + pad_left + pad_right) % stride != 0:
        pad_right += 1
    pad_top = 0
    pad_bottom = 0
    # iterate from pad_bottom, adding 1 until (input.shape[1] + pad_top + pad_bottom) % stride == 0
    while (input.shape[1] + pad_top + pad_bottom) % stride != 0:
        pad_bottom += 1
    # pad the image and label, the image has 3 channels
    input = F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
    # iterate over the image
    for i in range(0, input.shape[1]-size+1, stride):
        for j in range(0, input.shape[2]-size+1, stride):
            tile = input[:, i:i+size, j:j+size]
            # to cpu
            tile = tile.cpu()
            # append the tile to the list
            tile_list.append(tile)
            coords_list.append((i, j))
    # remove input from gpu to avoid memory issues
    input = input.cpu()
    # initiate pre_canvas as zero tensor with dimension dim_out x input.shape[1] x input.shape[2]
    canvas = torch.zeros(out_dim, input.shape[1], input.shape[2])
    # initiate weight_canvas as zero tensor with dimension input.shape[1] x input.shape[2]
    weight_canvas = torch.zeros(out_dim, input.shape[1], input.shape[2])
    if verbose:
        loop = tqdm(range(0, len(tile_list), batch_size))
    else:
        loop = range(0, len(tile_list), batch_size)
    for k in loop:
        # get the tiles
        tiles = tile_list[k:k+batch_size]
        # stack the tiles
        tiles = torch.stack(tiles)
        # to device
        tiles = tiles.to(device)
        # compute the predictions
        pred = prediction_fn(tiles)
        # move preds to cpu
        pred = pred.cpu()
        # for each prediction, multiply by the window and sum to the pred_canvas
        for l, _ in enumerate(pred):
            i, j = coords_list[k+l]
            canvas[:, i:i+size, j:j+size] += pred[l] * window
            weight_canvas[:, i:i+size, j:j+size] += window
    # divide the pred_canvas by the weight_canvas
    canvas = canvas / weight_canvas
    # # crop the pred_canvas to the original size
    bottom = canvas.shape[1] - pad_bottom
    right = canvas.shape[2] - pad_right
    canvas = canvas[:, pad_top:bottom, pad_left:right]
    # expand the pred_canvas to 4 dimensions
    canvas = canvas.unsqueeze(0)
    if not val_in_cpu:
        canvas = canvas.to(device)
    return canvas

def overlapping_predictions_aug(input, prediction_fn, size=256, stride=128, batch_size=2, val_in_cpu=False, verbose=False, out_dim=1):
    # gets input, rotates it 4 times, flips each of them vertically and horizontally, compute the predictions, then rotate and flip back
    rot_90 = torch.rot90(input, 1, [1, 2])
    rot_180 = torch.rot90(input, 2, [1, 2])
    rot_270 = torch.rot90(input, 3, [1, 2])
    rot_90_flip_v = torch.flip(rot_90, [1])
    rot_90_flip_h = torch.flip(rot_90, [2])
    rot_180_flip_v = torch.flip(rot_180, [1])
    rot_180_flip_h = torch.flip(rot_180, [2])
    rot_270_flip_v = torch.flip(rot_270, [1])
    rot_270_flip_h = torch.flip(rot_270, [2])
    flip_v = torch.flip(input, [1])
    flip_h = torch.flip(input, [2])
    # get the predictions
    pred = overlapping_predictions_torch(input, prediction_fn, size, stride, batch_size, val_in_cpu, verbose, out_dim)
    pred_rot_90 = overlapping_predictions_torch(rot_90, prediction_fn, size, stride, batch_size, val_in_cpu, verbose, out_dim)
    pred_rot_180 = overlapping_predictions_torch(rot_180, prediction_fn, size, stride, batch_size, val_in_cpu, verbose, out_dim)
    pred_rot_270 = overlapping_predictions_torch(rot_270, prediction_fn, size, stride, batch_size, val_in_cpu, verbose, out_dim)
    pred_rot_90_flip_v = overlapping_predictions_torch(rot_90_flip_v, prediction_fn, size, stride, batch_size, val_in_cpu, verbose, out_dim)
    pred_rot_90_flip_h = overlapping_predictions_torch(rot_90_flip_h, prediction_fn, size, stride, batch_size, val_in_cpu, verbose, out_dim)
    pred_rot_180_flip_v = overlapping_predictions_torch(rot_180_flip_v, prediction_fn, size, stride, batch_size, val_in_cpu, verbose, out_dim)
    pred_rot_180_flip_h = overlapping_predictions_torch(rot_180_flip_h, prediction_fn, size, stride, batch_size, val_in_cpu, verbose, out_dim)
    pred_rot_270_flip_v = overlapping_predictions_torch(rot_270_flip_v, prediction_fn, size, stride, batch_size, val_in_cpu, verbose, out_dim)
    pred_rot_270_flip_h = overlapping_predictions_torch(rot_270_flip_h, prediction_fn, size, stride, batch_size, val_in_cpu, verbose, out_dim)
    pred_flip_v = overlapping_predictions_torch(flip_v, prediction_fn, size, stride, batch_size, val_in_cpu, verbose, out_dim)
    pred_flip_h = overlapping_predictions_torch(flip_h, prediction_fn, size, stride, batch_size, val_in_cpu, verbose, out_dim)
    # get the predictions back to the original shape
    pred = pred.squeeze(0)
    pred_rot_90 = torch.rot90(pred_rot_90.squeeze(0), 3, [1, 2])
    pred_rot_180 = torch.rot90(pred_rot_180.squeeze(0), 2, [1, 2])
    pred_rot_270 = torch.rot90(pred_rot_270.squeeze(0), 1, [1, 2])
    pred_rot_90_flip_v = torch.flip(torch.rot90(pred_rot_90_flip_v.squeeze(0), 1, [1, 2]), [1]) 
    pred_rot_90_flip_h = torch.flip(torch.rot90(pred_rot_90_flip_h.squeeze(0), 1, [1, 2]), [2]) 
    pred_rot_180_flip_v = torch.flip(torch.rot90(pred_rot_180_flip_v.squeeze(0), 2, [1, 2]), [1])
    pred_rot_180_flip_h = torch.flip(torch.rot90(pred_rot_180_flip_h.squeeze(0), 2, [1, 2]), [2])
    pred_rot_270_flip_v = torch.flip(torch.rot90(pred_rot_270_flip_v.squeeze(0), 3, [1, 2]), [1]) 
    pred_rot_270_flip_h = torch.flip(torch.rot90(pred_rot_270_flip_h.squeeze(0), 3, [1, 2]), [2]) 
    pred_flip_v = torch.flip(pred_flip_v.squeeze(0), [1])
    pred_flip_h = torch.flip(pred_flip_h.squeeze(0), [2])
    pred = (pred + pred_rot_90 + pred_rot_180 + pred_rot_270 + pred_rot_90_flip_v + pred_rot_90_flip_h + pred_rot_180_flip_v + pred_rot_180_flip_h + pred_rot_270_flip_v + pred_rot_270_flip_h + pred_flip_v + pred_flip_h) / 12
    pred = pred.unsqueeze(0)
    return pred
    