import os
import numpy as np
from tqdm import tqdm
from config import *
from model.autoencoder import build_autoencoder
from utils.conversion_utils import ConversionUtils
from utils.visualization import Visualization
from keras.callbacks import EarlyStopping, ModelCheckpoint


os.makedirs(DATASET_DIR_STL, exist_ok=True)
os.makedirs(DATASET_DIR_PLY, exist_ok=True)
os.makedirs(CHECKPOINT_PATH.rsplit("/", 1)[0], exist_ok=True)
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)


for filename in tqdm(ConversionUtils.list_files_in_directory(DATASET_DIR_STL), desc="STL → PLY"):
    ConversionUtils.stl_to_ply(os.path.join(
        DATASET_DIR_STL, filename), POINT_CLOUD_DENSITY, DATASET_DIR_PLY)


dataset = []
for file in tqdm(ConversionUtils.list_files_in_directory(DATASET_DIR_PLY), desc="PLY → Voxels"):
    voxel = ConversionUtils.convert_to_binvox(
        os.path.join(DATASET_DIR_PLY, file), VOXEL_DIM)
    dataset.append(voxel)
dataset = np.expand_dims(np.array(dataset), axis=-1)


split = int(0.8 * len(dataset))
train_data, test_data = dataset[:split], dataset[split:]


autoencoder, encoder = build_autoencoder()
autoencoder.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor="loss", patience=5, min_delta=0.0001),
    ModelCheckpoint(filepath=CHECKPOINT_PATH,
                    save_weights_only=True, save_best_only=True)
]


autoencoder.fit(train_data, train_data, batch_size=BATCH_SIZE, epochs=EPOCHS,
                validation_data=(test_data, test_data), callbacks=callbacks)


reconstructed = autoencoder.predict(test_data, batch_size=BATCH_SIZE)
encoded = encoder.predict(test_data, batch_size=BATCH_SIZE)


autoencoder.save(os.path.join(SAVED_MODEL_DIR, "autoencoder.keras"))
encoder.save(os.path.join(SAVED_MODEL_DIR, "encoder.keras"))
