# ObjectDetection

# 🧠 Fashion MNIST Object Detection with TensorFlow

This project implements an object detection model that predicts both class and bounding box coordinates for clothing items in the Fashion MNIST dataset.

## 📦 Features

- Dual-headed neural network: class prediction and bounding box regression
- Uses MobileNetV2 as a feature extractor (transfer learning)
- Custom training pipeline on Fashion MNIST with synthetic bounding boxes
- Visualizes both predicted and true bounding boxes
- Model checkpointing and evaluation

## Setup
```bash
pip install -r requirements.txt
```

## Train the model
```bash
model.fit(
    train_images,
    {'bbox': train_bboxes, 'class': train_labels},
    validation_data=(test_images, {'bbox': test_bboxes, 'class': test_labels}),
    epochs=2,
    batch_size=32,
    callbacks=[checkpoint_callback]
)

```
## Evaluate the model
```bash
test_metrics = model.evaluate(test_images, {'bbox': test_bboxes, 'class': test_labels})
print("Test Losses and Accuracy:", test_metrics)
```
##  Load and use saved model
```bash
from tensorflow.keras.models import load_model
model = load_model("model_epoch_01.h5")
```

## Visualize predictions
```bash
def show_prediction(index):
    img = test_images[index]
    true_bbox = test_bboxes[index]
    pred_bbox, pred_class = model.predict(tf.expand_dims(img, 0))
    pred_bbox = pred_bbox[0]

    plt.imshow(img)
    h, w = img.shape[:2]

    # Draw predicted box
    x_min, y_min, x_max, y_max = pred_bbox * [w, h, w, h]
    plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                      edgecolor='green', linewidth=2, fill=False, label='Predicted'))

    # Draw true box
    x_min, y_min, x_max, y_max = true_bbox * [w, h, w, h]
    plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                      edgecolor='red', linewidth=2, fill=False, label='Ground Truth'))

    plt.legend()
    plt.title(f"Predicted Class: {np.argmax(pred_class)} | True Class: {test_labels[index]}")
    plt.show()

show_prediction(0)
```

## Result
| Metric              | Value  |
| ------------------- | ------ |
| Classification Acc. | \~87%  |
| BBox Loss (MSE)     | \~0.16 |
| Total Loss          | \~0.63 |

## Loss Functions
bbox: Mean Squared Error (MSE)

class: Sparse Categorical Crossentropy

