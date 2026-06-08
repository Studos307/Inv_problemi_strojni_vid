import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import laplace
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

original = img_as_float(data.camera())

# Gaussov šum
noisy = random_noise(
    original,
    mode='gaussian',
    var=0.01
)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(original, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(noisy, cmap='gray')
plt.title("Noisy")
plt.axis('off')

plt.tight_layout()
plt.show()

def linear_diffusion(image, iterations=50, dt=0.1):
    u = image.copy()

    for _ in range(iterations):
        u = u + dt * laplace(u)

    return u

def nonlinear_diffusion(image, iterations=50, dt=0.15, kappa=20):
    u = image.copy()

    for _ in range(iterations):

        # gradijenti
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u

        # Perona-Malik 
        cx = 1 / (1 + (ux / kappa)**2)
        cy = 1 / (1 + (uy / kappa)**2)

        # divergencija difuzijskog toka
        div_x = cx * ux - np.roll(cx * ux, 1, axis=1)
        div_y = cy * uy - np.roll(cy * uy, 1, axis=0)

        u = u + dt * (div_x + div_y)

    return u


def evaluate(reference, image, name):
    psnr = peak_signal_noise_ratio(reference, image)
    ssim = structural_similarity(reference, image, data_range=1.0)

    print(f"{name}")
    print(f"PSNR: {psnr:.3f}")
    print(f"SSIM: {ssim:.3f}")
    print()

iterations_list = [10, 20, 50, 100, 500]

for i in iterations_list:
    linear_result = linear_diffusion(
        noisy,
        iterations = i,
        dt = 0.1
    )

    nonlinear_result = nonlinear_diffusion(
        noisy,
        iterations = i,
        dt = 0.1,
        kappa = 0.1
    )

    print(f"broj iteracija = {i}")
    print()

    evaluate(original, linear_result, "Linear diffusion")
    evaluate(original, nonlinear_result, "Nonlinear diffusion")

    plt.subplot(2, 2, 3)
    plt.imshow(linear_result, cmap='gray')
    plt.title("Linear diffusion")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(nonlinear_result, cmap='gray')
    plt.title(f"Nonlinear diffusion")
    plt.axis('off')

    

    plt.tight_layout()
    plt.show()

kappa_list = [0.01, 0.05, 0.1, 0.5, 2]

linear_result = linear_diffusion(
        noisy,
        iterations = 50,
        dt = 0.1
    )



evaluate(original, linear_result, "Linear diffusion")

dt_list = [0.01, 0.05, 0.1, 0.15, 0.25, 0.26, 0.4]

for i in dt_list:
    linear_result = linear_diffusion(
        noisy,
        iterations = 50,
        dt = i
    )

    nonlinear_result = nonlinear_diffusion(
        noisy,
        iterations = 50,
        dt = i,
        kappa = 0.1
    )

    print(f"dt = {i}")
    print()
    
    evaluate(original, linear_result, "Linear diffusion")
    evaluate(original, nonlinear_result, "Nonlinear diffusion")

    plt.subplot(2, 2, 3)
    plt.imshow(linear_result, cmap='gray')
    plt.title("Linear diffusion")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(nonlinear_result, cmap='gray')
    plt.title(f"Nonlinear diffusion")
    plt.axis('off')

    

    plt.tight_layout()
    plt.show()

for i in kappa_list:
    print(f"kappa = {i}")
    print()

    nonlinear_result = nonlinear_diffusion(
        noisy,
        iterations = 50,
        dt = 0.1,
        kappa = i
    )

    
    evaluate(original, nonlinear_result, "Nonlinear diffusion")

    plt.subplot(2, 2, 3)
    plt.imshow(linear_result, cmap='gray')
    plt.title("Linear diffusion")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(nonlinear_result, cmap='gray')
    plt.title(f"Nonlinear diffusion")
    plt.axis('off')

    

    plt.tight_layout()
    plt.show()
