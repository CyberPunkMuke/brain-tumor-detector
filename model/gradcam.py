import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
grad_model = Model(
    inputs=base_model.input,
    outputs=[base_model.get_layer('block5_conv3').output, base_model.output]
)

def generate_gradcam(img_path, save_path):
    # Load image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Get gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        top_pred = tf.reduce_max(predictions)

    grads = tape.gradient(top_pred, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Apply heatmap on original image
    original = cv2.imread(img_path)
    original = cv2.resize(original, (224, 224))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    superimposed = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

    cv2.imwrite(save_path, superimposed)
    return save_path