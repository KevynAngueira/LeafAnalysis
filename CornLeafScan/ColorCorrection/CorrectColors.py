# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20

import os
import sys
import cv2
import numpy as np
import colour
from colour_checker_detection import detect_colour_checkers_segmentation

sys.path.append("..")

from Scripts.ResizeForDisplay import resize_for_display


class ColorCorrector:
    def __init__(self, config_dir="CorrectionConfigs"):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
        self.matrix = None

    def srgb_to_linear(self, image):
        threshold = 0.04045
        return np.where(image <= threshold, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)

    def linear_to_srgb(self, image):
        threshold = 0.0031308
        return np.where(image <= threshold, image * 12.92, 1.055 * (image ** (1 / 2.4)) - 0.055)

    def calculate_and_save_matrix(self, image_path, display = True, config_name=None):        
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        checkers = detect_colour_checkers_segmentation(image_rgb)

        if not checkers:
            raise ValueError("No ColorChecker detected in the image!")

        detected_checker = checkers[0]

        reference_srgb_255 = np.array([
            [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67],
            [133, 128, 177], [103, 189, 170], [214, 126, 44], [80, 91, 166],
            [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
            [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31],
            [187, 86, 149], [8, 133, 161], [243, 243, 242], [200, 200, 200],
            [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52]
        ]) / 255.0

        patches_array = np.array(detected_checker.data)
        patches = patches_array.reshape(24, -1, 3)
        detected_values = np.array([np.mean(patch, axis=0) for patch in patches])

        detected_lin = self.srgb_to_linear(detected_values)
        reference_lin = self.srgb_to_linear(reference_srgb_255)

        M = colour.characterisation.matrix_colour_correction(
            detected_lin, reference_lin, method='Finlayson 2015'
        )

        if display:
            preview = self.apply_matrix_to_image(image_rgb / 255.0, M)
            preview_uint8 = (preview * 255).astype(np.uint8)
            preview_bgr = cv2.cvtColor(preview_uint8, cv2.COLOR_RGB2BGR)
            cv2.imshow("Correction Preview", resize_for_display(preview_bgr))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        if config_name is None:
            config_name = input("Select a name for your color correction configuration: ")

        np.save(os.path.join(self.config_dir, f"{config_name}.npy"), M)

        

    def load_matrix(self, config_name):
        path = os.path.join(self.config_dir, f"{config_name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Matrix configuration '{config_name}' not found.")
        self.matrix = np.load(path)

    def prompt_for_matrix(self):
        configs = [f[:-4] for f in os.listdir(self.config_dir) if f.endswith(".npy")]
        if not configs:
            raise ValueError("No saved correction matrices found.")

        print("Available configurations:")
        for i, name in enumerate(configs):
            print(f"[{i}] {name}")

        index = int(input("Select a configuration by number: "))
        self.load_matrix(configs[index])

    def apply_matrix_to_image(self, image_rgb, matrix):
        lin = self.srgb_to_linear(image_rgb)
        corrected_lin = lin.reshape(-1, 3) @ matrix.T
        corrected_lin = np.clip(corrected_lin, 0, 1)
        corrected_srgb = self.linear_to_srgb(corrected_lin.reshape(image_rgb.shape))
        return np.clip(corrected_srgb, 0, 1)

    def apply_matrix(self, image_input, output_path=None):
        if self.matrix is None:
            self.prompt_for_matrix()

        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        else:
            image = image_input

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        corrected = self.apply_matrix_to_image(image_rgb, self.matrix)
        corrected_uint8 = (corrected * 255).astype(np.uint8)
        corrected_bgr = cv2.cvtColor(corrected_uint8, cv2.COLOR_RGB2BGR)

        if output_path:
            cv2.imwrite(output_path, corrected_bgr)
        return corrected_bgr

if __name__ == "__main__":
    colorCorrector = ColorCorrector()

    REFERENCE_PATH = "Images/Reference.jpg"

    corrected_bgr = colorCorrector.apply_matrix(REFERENCE_PATH)
    
    cv2.imshow("Correction BGR", resize_for_display(corrected_bgr))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
