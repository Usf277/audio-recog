from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

model = load_model('model_with_avoid_overfitting/voice_recognition_model.h5')
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
