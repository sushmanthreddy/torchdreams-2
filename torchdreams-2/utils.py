from __future__ import absolute_import, division, print_function


def _make_arg_str(arg):
    """
    Convert an argument to a string for logging.
    args:
        arg: The argument to be converted.
    Returns:
        str: The converted argument.
    """
    arg = str(arg)
    too_big = len(arg) > 15 or "\n" in arg
    return "..." if too_big else arg


def _extract_act_pos(acts, h=None, w=None):
    """
    Extract the activation at a specified position.

    Args:
       acts (torch.Tensor): The input tensor.
       h (int, optional): The height position. Defaults to None.
       w (int, optional): The width position. Defaults to None.

    Returns:
        torch.Tensor: The extracted activation.
    """
    shape = acts.shape
    h = shape[3] // 2 if h is None else h
    w = shape[4] // 2 if w is None else w
    return acts[:, :, :, h : h + 1, w : w + 1]


def get_layer_names(model):
    """
    Extracts layer names from a PyTorch model and replaces dots with underscores.

    Parameters:
    model (torch.nn.Module): The PyTorch model.

    Returns:
    list: A list of modified layer names with dots replaced by underscores.
    """
    return [name.replace(".", "_") for name, _ in model.named_modules()]


def torch_to_numpy(tensor):
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): The input tensor to be converted.

    Returns:
        np.ndarray: The converted NumPy array.
    """
    try:
        array = tensor.detach().cpu().numpy()
    except:
        array = np.array(tensor)

    if len(array.shape) == 3:
        if array.shape[0] == 3:
            array = np.moveaxis(array, 0, -1)
    elif len(array.shape) == 4:
        if array.shape[1] == 3:
            array = np.moveaxis(array, 1, -1)

    return array.astype(np.float32)


def clip_percentile(array, percentile=0.1):
    """
    Clip the values of an array at a specified percentile.

    Args:
        array (np.ndarray): The input array to be clipped.
        percentile (float, optional): The percentile value for clipping. Defaults to 0.1.

    Returns:
        np.ndarray: The clipped array.
    """
    return np.clip(array, None, np.percentile(array, 100 - percentile))
