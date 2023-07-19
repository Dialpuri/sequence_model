import tensorflow as tf
import tensorflow_addons as tfa

_ARGS = {"padding": "same", "activation": "relu", "kernel_initializer": "he_normal"}
_downsampling_args = {
    "padding": "same",
    "use_bias": True,
    "kernel_size": 4,
    "strides": 1,
}


def cnn():
    inputs = x = tf.keras.Input(shape=(16, 16, 16, 1))
    x = tf.keras.layers.Conv3D(filters=16, kernel_size=3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras. layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    # x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    # x = tf.keras.layers.BatchNormalization()(x)

    # x = tf.keras.layers.Conv3D(filters=8, kernel_size=3, activation="relu")(x)
    # x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    # x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(units=64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(units=2, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="3dcnn")
    return model

def t_model():
    inputs = x = tf.keras.Input(shape=(8, 8, 8, 1))
    skip_list = []

    filter_list = [16, 32, 64]

    for filters in filter_list:
        x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tfa.layers.InstanceNormalization(axis=-1,
        #                                      center=True,
        #                                      scale=True,
        #                                      beta_initializer="random_uniform",
        #                                      gamma_initializer="random_uniform")(x)

        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(.2)(x)

        # x = tfa.layers.InstanceNormalization(axis=-1,
        #                                      center=True,
        #                                      scale=True,
        #                                      beta_initializer="random_uniform",
        #                                      gamma_initializer="random_uniform")(x)

        # x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool3D()(x)


    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(.2)(x)

    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(.2)(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dropout(.2)(x)
    x = tf.keras.layers.Dense(8, activation='relu')(x)

    x = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    # outputs = tf.keras.layers.Conv3D(2, 3, padding="same", activation="softmax")(x)

    return tf.keras.Model(inputs, x)

if __name__ == "__main__":
    t_model().summary()