import numpy as np
import matplotlib.pyplot as plt


def im2gray(image):
    return np.dot(image, [[0.2989], [0.5870], [0.1140]])


# this form of normalization gives very poor results
def contrast_normalization(image, pixel_depth=255.0):
    return (image - pixel_depth / 2) / pixel_depth


# adjusts the contrast of each image individually, performs much better than the method above
def global_contrast_normalization(images, min_divisor=1e-4):
    num_images = images.shape[0]
    mean = np.mean(images, axis=(1, 2), dtype=float)
    std = np.std(images, axis=(1, 2), dtype=float, ddof=1)
    std[std < min_divisor] = 1.

    images_gcn = np.zeros(images.shape, dtype=float)
    for i in np.arange(num_images):
        images_gcn[i, :, :] = (images[i, :, :] - mean[i]) / std[i]

    return images_gcn


def visualize_some_examples(data, labels):
    plt.rcParams['figure.figsize'] = (15.0, 15.0)
    f, ax = plt.subplots(nrows=1, ncols=10)

    for i, j in enumerate(np.random.randint(0, len(labels), size=10)):
        ax[i].axis('off')
        ax[i].set_title(labels[j], loc='center')
        ax[i].imshow(data[j].squeeze())
    plt.show()