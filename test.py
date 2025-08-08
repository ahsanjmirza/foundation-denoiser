import imageio.v2 as iio
import numpy as np
import torch
import extract_denoiser
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
denoiser = extract_denoiser.load_Denoiser().to(device)

def run_inference(noisy, block_size, stride):
    denoiser.eval()
    with torch.no_grad():
        noisy = np.float32(noisy)
        hb = int(block_size/2)
        hs = int(stride/2)
        noisy_padded0 = np.pad(noisy, ((hb, hb), (hb, hb), (0, 0)), 'reflect')
        noisy_padded = np.zeros((1, 3, noisy_padded0.shape[0], noisy_padded0.shape[1]), np.float32)
        noisy_padded[0, 0, :, :] = noisy_padded0[:, :, 0]
        noisy_padded[0, 1, :, :] = noisy_padded0[:, :, 1]
        noisy_padded[0, 2, :, :] = noisy_padded0[:, :, 2]
        denoised_padded = np.pad(np.zeros(noisy.shape, dtype=np.float32), ((hs, hs), (hs, hs), (0, 0)), 'reflect')
        for i in range(hb, noisy_padded.shape[2]-hb, hs):
            for j in range(hb, noisy_padded.shape[3]-hb, hs):
                temp = noisy_padded[:, :, i-hb:i+hb, j-hb:j+hb]
                noisy_tensor = torch.from_numpy(temp).to(device)
                denoised_tensor = denoiser(noisy_tensor)[1].detach().cpu().numpy()[0]
                denoised_padded[i-hb:i-hb+stride, j-hb:j-hb+stride, 0] = denoised_tensor[0, hb-hs:hb+hs, hb-hs:hb+hs]
                denoised_padded[i-hb:i-hb+stride, j-hb:j-hb+stride, 1] = denoised_tensor[1, hb-hs:hb+hs, hb-hs:hb+hs]
                denoised_padded[i-hb:i-hb+stride, j-hb:j-hb+stride, 2] = denoised_tensor[2, hb-hs:hb+hs, hb-hs:hb+hs]
        denoised = denoised_padded[hs:-hs, hs:-hs]
        denoised = np.uint8(np.clip(denoised, 0, 255))
    return denoised


datasets_awgn = ["kodak", "McMaster", "CBSD68", "Set5"]



def process_dataset(dataset):

    dataset_path = os.path.join("./test/denoise/awgn", dataset)
    clean_images = os.path.join(dataset_path, "clean")

    # Sigma 15
    noisy_psnrs, denoised_psnrs = [], []
    noisy_ssim, denoised_ssim = [], []

    noisy_sigma_15_images = os.path.join(dataset_path, "noisy_15")
    denoised_sigma_15_images = os.path.join(dataset_path, "denoised_15")

    if not os.path.exists(denoised_sigma_15_images):
        os.makedirs(denoised_sigma_15_images)

    for img in os.listdir(noisy_sigma_15_images):
        noisy = iio.imread(os.path.join(noisy_sigma_15_images, img))
        clean = iio.imread(os.path.join(clean_images, img))
        denoised = run_inference(noisy, 256, 230)
        noisy_psnrs.append(psnr(clean, noisy, data_range = 255))
        noisy_ssim.append(ssim(clean, noisy, win_size = 15, channel_axis = -1))
        denoised_psnrs.append(psnr(clean, denoised, data_range = 255))
        denoised_ssim.append(ssim(clean, denoised, win_size = 15, channel_axis = -1))
        iio.imwrite(os.path.join(denoised_sigma_15_images, img), denoised)

    print(dataset, "-----> Sigma 15" )
    print("PSNR: ", np.mean(noisy_psnrs), "----->", np.mean(denoised_psnrs))
    print("SSIM: ", np.mean(noisy_ssim), "----->", np.mean(denoised_ssim))
    print()
    del noisy_psnrs, denoised_psnrs, noisy_ssim, denoised_ssim


    # Sigma 25
    noisy_psnrs, denoised_psnrs = [], []
    noisy_ssim, denoised_ssim = [], []

    noisy_sigma_25_images = os.path.join(dataset_path, "noisy_25")
    denoised_sigma_25_images = os.path.join(dataset_path, "denoised_25")

    if not os.path.exists(denoised_sigma_25_images):
        os.makedirs(denoised_sigma_25_images)

    for img in os.listdir(noisy_sigma_25_images):
        noisy = iio.imread(os.path.join(noisy_sigma_25_images, img))
        clean = iio.imread(os.path.join(clean_images, img))
        denoised = run_inference(noisy, 256, 230)
        noisy_psnrs.append(psnr(clean, noisy, data_range = 255))
        noisy_ssim.append(ssim(clean, noisy, win_size = 15, channel_axis = -1))
        denoised_psnrs.append(psnr(clean, denoised, data_range = 255))
        denoised_ssim.append(ssim(clean, denoised, win_size = 15, channel_axis = -1))
        iio.imwrite(os.path.join(denoised_sigma_25_images, img), denoised)

    print(dataset, "-----> Sigma 25" )
    print("PSNR: ", np.mean(noisy_psnrs), "----->", np.mean(denoised_psnrs))
    print("SSIM: ", np.mean(noisy_ssim), "----->", np.mean(denoised_ssim))
    print()
    del noisy_psnrs, denoised_psnrs, noisy_ssim, denoised_ssim


    # Sigma 50
    noisy_psnrs, denoised_psnrs = [], []
    noisy_ssim, denoised_ssim = [], []

    noisy_sigma_50_images = os.path.join(dataset_path, "noisy_50")
    denoised_sigma_50_images = os.path.join(dataset_path, "denoised_50")

    if not os.path.exists(denoised_sigma_50_images):
        os.makedirs(denoised_sigma_50_images)

    for img in os.listdir(noisy_sigma_50_images):
        noisy = iio.imread(os.path.join(noisy_sigma_50_images, img))
        clean = iio.imread(os.path.join(clean_images, img))
        denoised = run_inference(noisy, 256, 230)
        noisy_psnrs.append(psnr(clean, noisy, data_range = 255))
        noisy_ssim.append(ssim(clean, noisy, win_size = 15, channel_axis = -1))
        denoised_psnrs.append(psnr(clean, denoised, data_range = 255))
        denoised_ssim.append(ssim(clean, denoised, win_size = 15, channel_axis = -1))
        iio.imwrite(os.path.join(denoised_sigma_50_images, img), denoised)

    print(dataset, "-----> Sigma 50" )
    print("PSNR: ", np.mean(noisy_psnrs), "----->", np.mean(denoised_psnrs))
    print("SSIM: ", np.mean(noisy_ssim), "----->", np.mean(denoised_ssim))
    print()
    del noisy_psnrs, denoised_psnrs, noisy_ssim, denoised_ssim

for d in datasets_awgn:
    process_dataset(d)