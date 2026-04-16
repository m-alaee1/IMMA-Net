import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- (IMMA-Net) ---

def squeeze_excitation_block(input_tensor, ratio=8):
    channels = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling1D()(input_tensor)
    se = layers.Dense(channels // ratio, activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    return layers.Multiply()([input_tensor, se])

def inception_mamba_block(x, filters):
    branch1 = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
    branch2 = layers.Conv1D(filters, 11, padding='same', activation='relu')(x)

    merged = layers.Concatenate()([branch1, branch2])

    #Mamba-Inspired (Selective Gating)
    gate = layers.Conv1D(filters*2, 1, padding='same', activation='sigmoid')(merged)
    res = layers.Conv1D(filters*2, 3, padding='same', activation='tanh')(merged)
    mamba_out = layers.Multiply()([res, gate])

    # Attention
    return squeeze_excitation_block(mamba_out)

def create_ultra_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, 7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # IMA
    x = inception_mamba_block(x, 64)
    x = layers.MaxPooling1D(2)(x)
    x = inception_mamba_block(x, 128)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)

# ---Model Training---
print(" Model Training( IMMA-Net)...")
ultra_model = create_ultra_model(input_shape, num_classes)
ultra_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                    loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_ultra = ultra_model.fit(X_train, y_train, epochs=100, batch_size=32,
                                validation_data=(X_test, y_test), verbose=1)

# --- Final Diagram ---
ultra_acc = max(history_ultra.history['val_accuracy'])

