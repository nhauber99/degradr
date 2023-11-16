# quick and dirty python script for calculating a few statistics using a folder with several bias frames
#   for analysing a new camera, capture ideally more than 100 bias frames (short exposure + no light)
#   then convert them to raw tif files (done with PixInsight in my case) and put them into a "raw" folder
#   set the root_dir to the folder containing the "raw" folder and run the script

import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from Operations import Noise, Convolve
from Analysis.Utilities import FileIO, Plotter
from Analysis.Utilities.TorchUtil import np_to_torch

max_count = -1  # set to some number > 0 for faster debugging
bit_depth = 14
root_dir = "data/CanonR6_ISO1600"


def analyse_read_noise(path: str):
    raw_dir = os.path.join(path, "raw")
    hist_path = os.path.join(path, "histogram.txt")
    mean_image_path = os.path.join(path, "mean.tif")
    std_image_path = os.path.join(path, "std.tif")

    count = 0
    mean_image = None
    histogram = np.zeros(2 ** bit_depth, np.int64)

    if not os.path.exists(hist_path) or not os.path.exists(mean_image_path) or not os.path.exists(std_image_path):
        for image_name in tqdm(os.listdir(raw_dir), 'calculating mean'):
            image = FileIO.read_image(os.path.join(raw_dir, image_name), convert_to_float=False)
            hist, bins = np.histogram(image, bins=2 ** bit_depth, range=(0, 2 ** bit_depth))
            histogram += hist
            if mean_image is None:
                mean_image = image.astype(np.float64)
            else:
                mean_image += image
            count += 1
            if 0 < max_count <= count:
                break

        mean_image /= count
        count = 0

        col_std = 0
        row_std = 0
        lowpass_kernel = Convolve.gaussian_kernel(65, 5).cuda()
        std_image = np.zeros_like(mean_image)
        for image_name in tqdm(os.listdir(raw_dir), 'calculating per pixel variance'):
            image = FileIO.read_image(os.path.join(raw_dir, image_name), convert_to_float=False)
            image_tensor = np_to_torch(image).cuda()
            image_tensor -= Convolve.apply_kernel(image_tensor.unsqueeze(0), lowpass_kernel).squeeze(0)
            col_std += float(image_tensor.mean(dim=2).std())
            row_std += float(image_tensor.mean(dim=1).std())
            std_image += (image - mean_image) ** 2
            count += 1
            if 0 < max_count <= count:
                break

        col_std /= count
        row_std /= count
        std_image = np.sqrt(std_image / count)

        theoretical_col_std = std_image.mean() / np.sqrt(std_image.shape[0])
        theoretical_row_std = std_image.mean() / np.sqrt(std_image.shape[1])

        col_std = np.sqrt(col_std ** 2 - theoretical_col_std ** 2)
        row_std = np.sqrt(row_std ** 2 - theoretical_row_std ** 2)

        print(f"row std: {row_std}")
        print(f"col std: {col_std}")

        FileIO.write_image(mean_image_path, mean_image / (2 ** 16), np.float32)
        FileIO.write_image(std_image_path, std_image / (2 ** 16), np.float32)
        histogram = histogram.astype(np.double) / (count * mean_image.shape[0] * mean_image.shape[1])
        np.savetxt(hist_path, histogram)
    else:
        histogram = np.loadtxt(hist_path)
        mean_image = FileIO.read_image(mean_image_path, True) * (2 ** 16)
        std_image = FileIO.read_image(std_image_path, True) * (2 ** 16)

    mean = (histogram * np.arange(histogram.shape[0])).sum()
    std = np.sqrt((histogram * (np.arange(histogram.shape[0]) - mean) ** 2).sum())
    std_std = std_image.std()
    gaussian1 = Noise.GaussianParams(mean - 0.2 * std, 0.7 * std, 0.79965)
    gaussian2 = Noise.GaussianParams(mean + 0.5 * std, 1.5 * std, 0.2)
    gaussian3 = Noise.GaussianParams(mean + 2 * std, 9 * std, 0.00035)
    outliers = 1 - (histogram[int(mean - 10 * std):int(mean + 10 * std)].sum() / histogram.sum())

    print(f"mean: {mean}")
    print(f"std: {std}")
    print(f"std of pixel std: {std_std}")
    print(f"gaussian fit 1: {gaussian1.amplitude}*gaussian({gaussian1.mean},{gaussian1.std})")
    print(f"gaussian fit 2: {gaussian2.amplitude}*gaussian({gaussian2.mean},{gaussian2.std})")
    print(f"gaussian fit 3: {gaussian3.amplitude}*gaussian({gaussian3.mean},{gaussian3.std})")
    print(f"amount of pixels outside +-10*std: {outliers}")

    # plot result
    plt.figure(figsize=(8, 6), dpi=200)
    Plotter.histogram_plot(histogram, True)
    x = np.arange(histogram.shape[0], dtype=np.float32)
    plt.plot(x, Noise.gaussian_dist_combination(x, (gaussian1, gaussian2, gaussian3)), 'r--')
    plt.ylim(1e-10, 1)
    # plt.xlim(mean - 40 * std, mean + 40 * std)
    plt.xlim(1900, 2600)
    plt.savefig(os.path.join(path, "histogram2.png"), dpi=200)
    plt.show()


if __name__ == "__main__":
    analyse_read_noise(root_dir)
