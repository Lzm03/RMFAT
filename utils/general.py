import datetime
import os
from collections import OrderedDict

import torch


def get_time(path):
    """Get the time of the file

    Args:
        path (string): path of the file to get the date / time of

    Returns:
        datetime: return time of the file
    """
    tstring = path.split("_")[-1]
    return datetime.datetime.strptime(tstring, "%m-%d-%Y-%H-%M-%S")


def find_latest_checkpoint(all_log_dir, run_name):
    """find last chpt

    Args:
        all_log_dir (string): path of the log directory
        run_name (string): name of the run

    Returns:
        string: path of the latest checkpoint
    """
    run_name_list = [v for v in os.listdir(all_log_dir) if v.startswith(run_name)]
    run_name_list.sort(reverse=True, key=get_time)
    for p in run_name_list:
        ckpt_path = os.path.join(all_log_dir, p, "checkpoints", "latest.pth")
        if os.path.exists(ckpt_path):
            return ckpt_path
    return None


def create_log_folder(dir_name):
    """Create the log folder and return subdirectory paths

    Args:
        dir_name (string): directory name log folder

    Returns:
        string: path to the img folder within the log folder
        string: path to the checkpoint folder within the log folder
    """
    os.makedirs(dir_name)
    path_imgs = os.path.join(dir_name, "imgs")
    os.mkdir(path_imgs)
    path_ckpt = os.path.join(dir_name, "checkpoints")
    os.mkdir(path_ckpt)
    return path_imgs, path_ckpt


def load_checkpoint(model, weights):
    """Load weights into a model

    Args:
        model: model to load the weights of
        weights: the weights to load into the model

    Returns:
        checkpoint: checkpoint of model
        model: model loaded with weights
    """
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    return checkpoint, model


def get_cuda_info(logger):
    """Get the CUDA information and output to logger

    Args:
        logger: Information to dump the CUDA information to
    """
    cudnn_version = torch.backends.cudnn.version()
    count = torch.cuda.device_count()

    logger.info(f"__CUDNN VERSION: {cudnn_version}\n" f"__Number CUDA Devices: {count}")

    for device_id in range(count):
        device_name = torch.cuda.get_device_name(device_id)
        memory = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)

        logger.info(
            f"__CUDA Device {device_id} Name: {device_name}\n"
            f"__CUDA Device {device_id} Total Memory [GB]: {memory}"
        )
