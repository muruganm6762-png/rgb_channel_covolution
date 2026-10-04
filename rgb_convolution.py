from PIL import Image
import numpy as np
import os


def convolution2d(image, kernel):
    """
    Perform 2D convolution on a single image channel.
    """
    image = image.astype(float)

    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    padded = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant"
    )

    output = np.zeros_like(image, dtype=float)

    for i in range(image_height):
        for j in range(image_width):
            region = padded[
                i:i + kernel_height,
                j:j + kernel_width
            ]

            output[i, j] = np.sum(region * kernel)

    return output


def normalize(feature_map):
    """
    Normalize values to the range 0-255.
    """
    min_value = feature_map.min()
    max_value = feature_map.max()

    if max_value == min_value:
        return np.zeros_like(feature_map, dtype=np.uint8)

    normalized = (
        (feature_map - min_value)
        / (max_value - min_value)
        * 255
    )

    return normalized.astype(np.uint8)


def main():
    input_path = "input.jpg"
    output_dir = "results"

    os.makedirs(output_dir, exist_ok=True)

    # Read RGB image
    image = Image.open(input_path).convert("RGB")
    image_array = np.array(image)

    # Separate RGB channels
    red = image_array[:, :, 0]
    green = image_array[:, :, 1]
    blue = image_array[:, :, 2]

    # Edge detection kernel
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])

    # Apply convolution separately
    red_feature = convolution2d(red, kernel)
    green_feature = convolution2d(green, kernel)
    blue_feature = convolution2d(blue, kernel)

    # Normalize individual feature maps
    red_feature = normalize(red_feature)
    green_feature = normalize(green_feature)
    blue_feature = normalize(blue_feature)

    # Combine feature maps
    combined = np.stack(
        [red_feature, green_feature, blue_feature],
        axis=2
    )

    # Save results
    Image.fromarray(red_feature).save(
        f"{output_dir}/red_feature_map.png"
    )

    Image.fromarray(green_feature).save(
        f"{output_dir}/green_feature_map.png"
    )

    Image.fromarray(blue_feature).save(
        f"{output_dir}/blue_feature_map.png"
    )

    Image.fromarray(combined).save(
        f"{output_dir}/combined_feature_map.png"
    )

    print("RGB convolution completed successfully.")
    print("Red channel convolution completed.")
    print("Green channel convolution completed.")
    print("Blue channel convolution completed.")
    print("Feature maps combined successfully.")
    print(f"Results saved in: {output_dir}/")


if __name__ == "__main__":
    main()
