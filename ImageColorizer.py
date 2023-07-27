# NOTE:  This must be the first call in order to work properly!
from deoldify import device
from deoldify.device_id import DeviceId

# choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU0)

from deoldify.visualize import *

plt.style.use("dark_background")
torch.backends.cudnn.benchmark = True
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*?Your .*? set is empty.*?"
)

colorizer = get_image_colorizer(artistic=True)

import os
from tqdm import tqdm
import requests
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from PIL import ImageOps
from pathlib import Path


def generate_comparison(original, colorized, path="compare_images/"):
    filename = original.split(os.sep)[-1]
    original_img = Image.open(original)
    colorized_img = Image.open(colorized)
    separator = Image.new("RGB", (30, original_img.size[1]), (0, 0, 0))
    comparison = np.concatenate(
        (np.array(original_img), np.array(separator), np.array(colorized_img)), axis=1
    )

    comparison_img = Image.fromarray(comparison)

    border_color = (0, 0, 0)  # RGB para negro
    border_width = 40  # Cambia este valor para ajustar el ancho del borde
    comparison_img_with_border = ImageOps.expand(
        comparison_img, border=border_width, fill=border_color
    )

    basename = os.path.splitext(filename)[
        0
    ]  # Obtiene el nombre base del archivo sin la extensión
    new_filename = f"{basename}.jpg"

    compared_path = f"{path}{new_filename}"
    comparison_img_with_border.save(compared_path)


# NOTE:  Max is 45 with 11GB video cards. 35 is a good default
render_factor = 40
source_path = "D:/webp"  # "/mnt/d/webp"
result_path = Path("D:/colorized/")  # Path("/mnt/d/colorized/")
compare_path = "D:/compared/"  # "/mnt/d/compared/"

# Download images to folder test_images/
# 400
files = os.listdir(source_path)[400:1000]
for index, picture in tqdm(enumerate(files), total=len(files)):
    try:
        full_path = os.path.join(source_path, picture)
        result_file = colorizer.plot_transformed_image(
            path=full_path,
            results_dir=result_path,
            render_factor=render_factor,
            compare=False,
            watermarked=True,
        )
        generate_comparison(full_path, result_file, path=compare_path)
        plt.close("all")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error: {e} at index {index}")
