# ImageDetech

ImageDetech is a deep learning-based application for detecting car damage from images. It uses a convolutional neural network (CNN) to classify different types of car damage such as scratches, seat damage, tire wear, and dents.

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ImageDetech.git
    cd ImageDetech
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Download the pre-trained model and place it in the root directory of the repository:
    ```sh
    # Example command to download the model
    wget https://example.com/path/to/car_damage_model.pth -O car_damage_model.pth
    ```

## Usage

To predict the type of damage in an image, run the `predict_image.py` script with the path to the image:

```sh
python predict_image.py --image_path ./dataset/test/test_2.jpg
```

Replace `./dataset/test/test_2.jpg` with the path to your image.

## Example

```sh
python predict_image.py --image_path ./dataset/test/test_2.jpg
```

Output:
```
Image ./dataset/test/test_2.jpg is predicted as: scratch
```

## License

This project is licensed under the MIT License.