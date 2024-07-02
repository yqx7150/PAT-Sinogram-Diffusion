import os
from models import utils as mutils
import torch
import numpy as np
from sampling import (
    NoneCorrector,
    NonePredictor,
    shared_corrector_update_fn,
    shared_predictor_update_fn,
)
import functools
import cv2
import math
from scipy.io import savemat

# from skimage.measure import compare_psnr,compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import scipy.io as io


# from lmafit_mc_adp_v2_numpy import lmafit_mc_adp


def write_Data(filedir, model_num, psnr, ssim):
    # filedir="result.txt"
    with open(os.path.join("./results", filedir), "a+") as f:  # a+
        f.writelines(
            str(model_num)
            + " "
            + "["
            + str(round(psnr, 2))
            + " "
            + str(round(ssim, 4))
            + "]"
        )
        f.write("\n")


def save_img(img, img_path):

    img = np.clip(img * 255, 0, 255)

    cv2.imwrite(img_path, img)


def write_Data(filedir, num, psnr, ssim):
    # filedir="result.txt"
    with open(os.path.join(filedir, str(num) + ".txt"), "a+") as f:  # a+
        f.writelines(str(round(psnr, 4)) + "  " + str(round(ssim, 4)))
        f.write("\n")


def get_pc_inpainter(
    sde,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-5,
):
    """Create an image inpainting function that uses PC samplers.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
      corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for the corrector.
      n_steps: An integer. The number of corrector steps per update of the corrector.
      probability_flow: If `True`, predictor solves the probability flow ODE for sampling.
      continuous: `True` indicates that the score-based model was trained with continuous time.
      denoise: If `True`, add one-step denoising to final samples.
      eps: A `float` number. The reverse-time SDE/ODE is integrated to `eps` for numerical stability.

    Returns:
      An inpainting function.
    """
    # Define predictor & corrector
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    def pc_inpainter(model, dimg, data, good_input, unet_input):
        """Predictor-Corrector (PC) sampler for image inpainting.

        Args:
          model: A score model.
          data: A PyTorch tensor that represents a mini-batch of images to inpaint.
          mask: A 0-1 tensor with the same shape of `data`. Value `1` marks known pixels,
            and value `0` marks pixels that require inpainting.

        Returns:
          Inpainted (complete) images.
        """
        with torch.no_grad():
            timesteps = torch.linspace(sde.T, eps, sde.N)
            x_input = dimg
            x_input = (
                torch.from_numpy(x_input)
                .to(torch.float32)
                .cuda()
                .unsqueeze(0)
                .unsqueeze(0)
            )

            data = (
                torch.from_numpy(data)
                .to(torch.float32)
                .cuda()
                .unsqueeze(0)
                .unsqueeze(0)
            )
            x_mean = x_input
            x1 = x_mean
            x2 = x_mean
            x3 = x_mean
            ##########置换保真
            mask = np.zeros((512, 512))
            y0 = np.zeros((512, 512))
            data1 = data.squeeze(0).squeeze(0)
            data2 = data1.cpu()
            data2 = np.array(data2)
            for i in range(32):
                mask[16 * i, :] = 1
                y0[16 * i, :] = data2[i, :]
            cuda_device = torch.device("cuda")
            mask = torch.from_numpy(mask)
            mask = mask.to(cuda_device)
            mask = mask.to(torch.float32)
            y0 = torch.from_numpy(y0)
            y0 = y0.to(cuda_device)
            y0 = y0.to(torch.float32)

            for i in range(sde.N):
                print("===================", i, "===================")
                t = timesteps[i].cuda()
                vec_t = torch.ones(x_input.shape[0], device=t.device) * t
                x, x_mean = predictor_update_fn(
                    x_mean, vec_t, good=unet_input, model=model
                )
                ####保真
                x_mean = x_mean * (1 - mask) + y0
                x1, x2, x3, x_mean = corrector_update_fn(
                    x1, x2, x3, x_mean, vec_t, good=unet_input, model=model
                )
                x_mean = x_mean.to(torch.float32).cuda()
                # 保真
                x_mean = x_mean * (1 - mask) + y0
                x_show = np.array(x_mean.squeeze(0).squeeze(0).cpu())
                # x_show = np.array(
                #    ((x_show - x_show.min()) / (x_show.max() - x_show.min())).to("cpu")
                # )
                # cv2.imwrite(
                #    "./temp_outcomes_mat/testpadsecond_" + str(i) + ".png", x_show * 255.0
                # )
                mat_dict = {"sensor_data": x_show}
                savemat(
                    "/home/lqg/LJB/work_after_24_1_28/ncns+unt/retest/xiaoshu_matmat/32/testpadsecond_"
                    + str(i)
                    + ".mat",
                    mat_dict,
                )
                # psnr = compare_psnr(x_show, good_input / 255.0, data_range=1)
                # print(np.max(x_show), np.max(good_input / 255.0))
                # ssim = compare_ssim(x_show, good_input / 255.0, data_range=1)
                # print(" PSNR:", psnr, " SSIM:", ssim)
                psnr, ssim = None, None
                if i == (sde.N - 2):
                    return x_show * 255.0, psnr, ssim

    return pc_inpainter
