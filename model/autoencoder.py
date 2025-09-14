from keras.models import Model
from keras.layers import Input, Conv3D, UpSampling3D, AveragePooling3D
from config import INPUT_SHAPE

def build_autoencoder():
    input_data = Input(shape=INPUT_SHAPE)

    x = Conv3D(32, (3, 3, 3), activation='selu', padding='same')(input_data)
    x = AveragePooling3D((2, 2, 2), padding='same')(x)
    x = Conv3D(16, (3, 3, 3), activation='selu', padding='same')(x)
    encoded = AveragePooling3D((2, 2, 2), padding='same')(x)

    x = Conv3D(16, (3, 3, 3), activation='selu', padding='same')(encoded)
    x = UpSampling3D((2, 2, 2))(x)
    x = Conv3D(32, (3, 3, 3), activation='selu', padding='same')(x)
    x = UpSampling3D((2, 2, 2))(x)
    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_data, decoded)
    encoder = Model(input_data, encoded)

    return autoencoder, encoder
