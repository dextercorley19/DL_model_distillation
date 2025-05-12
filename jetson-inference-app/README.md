# Jetson Inference App

This project provides a framework for running inference on a trained model using a Jetson Nano. The model is designed for image classification tasks and utilizes a lightweight architecture suitable for edge devices.

## Project Structure

```
jetson-inference-app
├── model
│   └── trained_model.pth        # Trained model weights
├── src
│   ├── inference.py             # Main script for running inference
│   ├── preprocess_utils.py       # Utility functions for image preprocessing
│   └── class_labels.py           # Class labels for model outputs
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd jetson-inference-app
   ```

2. **Install Dependencies**:
   Ensure you have Python installed on your Jetson Nano. Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Load the Trained Model**:
   The trained model weights are located in the `model` directory. Ensure that `trained_model.pth` is present.

## Usage

1. **Run Inference**:
   To run inference on an image, use the following command:
   ```bash
   python src/inference.py --image <path_to_image>
   ```

2. **Preprocessing**:
   The `preprocess_utils.py` file contains functions that will automatically preprocess the input image before it is fed into the model.

3. **Class Labels**:
   The `class_labels.py` file defines the mapping of class indices to human-readable labels. You can modify this file to suit your specific classification task.

## Example

Here is an example of how to run inference on an image:

```bash
python src/inference.py --image /path/to/your/image.jpg
```

This will output the predicted class label for the input image.

## Notes

- Ensure that your Jetson Nano has sufficient resources to run the model efficiently.
- You may need to adjust the preprocessing steps based on the specific requirements of your trained model.

For any issues or contributions, please refer to the project's GitHub page.