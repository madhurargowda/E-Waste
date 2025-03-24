import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("your_model.h5")

# Convert it to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS  # Use only built-in TensorFlow Lite ops
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optimize model

tflite_model = converter.convert()

# Save the converted model
with open("C:/Users/medhi/e_waste/flask-project/model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model converted successfully with compatible ops!")