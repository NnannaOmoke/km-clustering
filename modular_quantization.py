import numpy as np
import matplotlib.pyplot as plotter
import matplotlib.image as imager
from sklearn.cluster import KMeans
from PIL import Image as pil_imager


def read_process(path):
    original_rgb = imager.imread(path)
    original_val = original_rgb.shape
    packed_val = original_rgb.reshape(original_val[0] * original_val[1], original_val[2])
    return packed_val, original_val

def quantizer(colors, packed_val):
    learner = KMeans(n_clusters = colors, n_init = "auto")
    n_colors = colors
    label = learner.fit_predict(packed_val[0])
    centers = learner.cluster_centers_.round(0).astype(int)
    quantized_img = np.reshape(centers[label], packed_val[1])
    return quantized_img

def main():
    packed_val = read_process(input("Please input the file location of the image"))
    pil_imager.fromarray((quantizer(int(input("Please input how many colors you want to quantize the image with")), packed_val) * -255).astype(np.uint8)).save("quantized_image.jpg")
main()