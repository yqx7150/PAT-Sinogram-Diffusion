import torch
import tensorflow as tf
import os
import logging


def restore_checkpoint(ckpt_dir, state, device):
    if not tf.io.gfile.exists(ckpt_dir):
        tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
        logging.warning(
            f"No checkpoint found at {ckpt_dir}. " f"Returned the same state as input"
        )
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state["optimizer"].load_state_dict(loaded_state["optimizer"])
        state["model"].load_state_dict(loaded_state["model"], strict=False)
        state["ema"].load_state_dict(loaded_state["ema"])
        state["step"] = loaded_state["step"]
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
    }
    torch.save(saved_state, ckpt_dir)


if __name__ == "__main__":
    state = {}
    state["optimizer"] = 0
    state["model"] = 0
    state["ema"] = 0
    state["step"] = 0
    state = restore_checkpoint(
        ckpt_dir="/home/lqg/LJB/work_after_24_1_28/diffu1/exp_100BW_2000_128_ljb_2024_2_6/checkpoint_11.pth",
        state=state,
        device="cuda:0",
    )
    print(state["step"])
