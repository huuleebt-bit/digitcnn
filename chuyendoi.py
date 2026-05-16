import tensorflow as tf

# 1. Load model .h5 cũ của bạn
model = tf.keras.models.load_model('model.h5')

# 2. Khởi tạo bộ chuyển đổi
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Tối ưu hóa dung lượng (Quantization) - Giúp model nhẹ hơn nữa
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Thực hiện chuyển đổi
tflite_model = converter.convert()

# 5. Lưu thành file mới
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print("Đã chuyển đổi sang TFLite thành công!")