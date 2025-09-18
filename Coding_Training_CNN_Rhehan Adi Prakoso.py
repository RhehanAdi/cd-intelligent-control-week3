import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import json

# Dataset
train_dir = "dataset/seg_train/seg_train"
val_dir = "dataset/seg_test/seg_test"

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20, 
    zoom_range=0.2, 
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(150, 150), batch_size=32, class_mode='categorical'
)

# Model CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Simpan model
model.save('cnn_model.h5')

# Simpan label kelas
with open("class_labels.json", "w") as f:
    json.dump(list(train_generator.class_indices.keys()), f)

print("✅ Model dan label kelas berhasil disimpan!")
