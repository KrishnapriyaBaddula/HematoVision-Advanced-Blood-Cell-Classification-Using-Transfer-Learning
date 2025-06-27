# Sample: Load model and predict
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('blood_cell_model.h5')
img = image.load_img('test.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
pred = model.predict(x)
print("Predicted class:", np.argmax(pred))
