from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#image = data.hubble_deep_field()[0:500, 0:500]
from PIL import Image
import os

file_dir = "/home/mll/v_mll3/OCR_data/CRAFT-pytorch-master/data/test_gaussian"
file_list = os.listdir(file_dir)

for file in file_list:
    image = mpimg.imread(file_dir+'/'+file)

    image_gray = rgb2gray(image)
    #image_gray = rgb2gray(image)
    #plt.imshow(image)
    #plt.show()
    plt.imshow(image_gray)
    plt.show()
    blobs_log = blob_log(image_gray, max_sigma=30, min_sigma=10, num_sigma=10, threshold=.1,overlap=0.2)
    #blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.2)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * math.sqrt(2)

    blobs_dog = blob_dog(image_gray,  max_sigma=40, min_sigma=10, threshold=0.1,overlap=0.3)
    blobs_dog[:, 2] = blobs_dog[:, 2] * math.sqrt(2)

    blobs_doh = blob_doh(image_gray, max_sigma=40,min_sigma=10, threshold=.01,overlap=0.3)

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
              'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image)
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()

    plt.tight_layout()
    plt.show()