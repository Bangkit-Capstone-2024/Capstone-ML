# TensorFlow Serving Endpoint

This repository contains a setup to deploy and serve a TensorFlow model using TensorFlow Serving.

## Getting Started

Follow the instructions below to get the TensorFlow Serving Docker container running and how to send requests to the endpoint.

### Prerequisites

- Docker
- Python 3.x
- `requests` library for Python
- `opencv-python` library for Python (for image processing)

### Setting Up TensorFlow Serving

1. **Ensure your model directory structure is correct.** It should look something like this:

   ```
   models/
   └── image_model/
       └── 1/
           ├── saved_model.pb
           └── variables/
               ├── variables.data-00000-of-00001
               └── variables.index
   ```

2. **Run TensorFlow Serving using Docker:**

   ```sh
   docker run -p 8501:8501 --name=tf_serving \
     --mount type=bind,source="C:\path\to\your\models\image_model",target=/models/image_model \
     -e MODEL_NAME=image_model -t tensorflow/serving
   ```

3. **Verify TensorFlow Serving is running:**

   Open your browser and navigate to `http://localhost:8501/v1/models/image_model`.

### Using the Endpoint

You can send POST requests to the TensorFlow Serving endpoint to get predictions from the model.

The URL for the endpoint is:
