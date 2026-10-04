# RGB Channel Convolution

## Aim

Implement convolution separately on the Red, Green, and Blue channels of an image and combine the resulting feature maps.

## Objective

The objective of this project is to understand how convolution works on a color image.

Instead of applying convolution directly to the complete RGB image, the image is divided into three separate channels:

- Red
- Green
- Blue

The same convolution kernel is applied independently to each channel. The resulting feature maps are then combined to produce a final RGB feature map.

## Technologies Used

- Python
- NumPy
- Pillow

## Project Structure

```text
rgb-channel-convolution/
│
├── rgb_convolution.py
├── README.md
├── requirements.txt
└── results/
    └── .gitkeep
