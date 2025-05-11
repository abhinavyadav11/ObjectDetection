import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load Fashion MNIST and preprocess
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Expand channels and resize to 224x224
train_images = tf.image.resize(tf.expand_dims(train_images, -1), (224, 224)) / 255.0
test_images = tf.image.resize(tf.expand_dims(test_images, -1), (224, 224)) / 255.0

# Convert grayscale to RGB
train_images = tf.image.grayscale_to_rgb(train_images)
test_images = tf.image.grayscale_to_rgb(test_images)

# 2. Generate realistic bounding boxes from non-zero pixel regions
def compute_bounding_boxes(images):
    bboxes = []
    for img in images:
        img_np = img.numpy().squeeze() if isinstance(img, tf.Tensor) else img.squeeze()
        non_zero_indices = np.argwhere(img_np > 0)

        if non_zero_indices.size == 0:
            bboxes.append([0.0, 0.0, 1.0, 1.0])  # fallback
            continue

        y_min, x_min = non_zero_indices.min(axis=0)
        y_max, x_max = non_zero_indices.max(axis=0)

        x_min /= 224.0
        y_min /= 224.0
        x_max /= 224.0
        y_max /= 224.0

        bboxes.append([x_min, y_min, x_max, y_max])
    
    return np.array(bboxes, dtype=np.float32)

train_bboxes = compute_bounding_boxes(train_images)
test_bboxes = compute_bounding_boxes(test_images)

# 3. Load pre-trained backbone (MobileNetV2)
backbone = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# 4. Freeze most layers, unfreeze last few
for layer in backbone.layers:
    layer.trainable = False
for layer in backbone.layers[-20:]:
    layer.trainable = True

# 5. Add detection head
num_classes = 10
x = backbone.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)

bbox_output = layers.Dense(4, name='bbox')(x)
class_output = layers.Dense(num_classes, activation='softmax', name='class')(x)

model = tf.keras.Model(inputs=backbone.input, outputs=[bbox_output, class_output])

# 6. Compile
model.compile(
    optimizer='adam',
    loss={
        'bbox': tf.keras.losses.MeanSquaredError(),
        'class': tf.keras.losses.SparseCategoricalCrossentropy()
    },
    loss_weights={'bbox': 1.0, 'class': 1.0},
    metrics={'class': 'accuracy'}
)

# 7. Save checkpoint callback
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    filepath='model_epoch_{epoch:02d}.h5',
    save_freq='epoch',
    save_weights_only=False,
    save_best_only=False,
    verbose=1
)

# 8. Train
model.fit(
    train_images,
    {'bbox': train_bboxes, 'class': train_labels},
    validation_data=(test_images, {'bbox': test_bboxes, 'class': test_labels}),
    epochs=2,
    batch_size=32,
    callbacks=[checkpoint_callback]
)

# 9. Evaluate
test_metrics = model.evaluate(test_images, {'bbox': test_bboxes, 'class': test_labels})
print("Test Losses and Accuracy:", test_metrics)

# 10. Visualize prediction
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
