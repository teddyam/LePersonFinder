import os

os.environ["KERAS_BACKEND"] = "jax"  # @param ["tensorflow", "jax", "torch"]

import tensorflow as tf
import keras
# from keras imp
from keras import layers
from prep import XMLtoJSON, create_filtered_dataset
import numpy as np
import matplotlib.pyplot as plt
from params import fields

# Configure the hyperparams 
num_classes = 4
input_shape = (256, 256, 3) # If we implement patch creation as a layer 
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 2
num_epochs = 10  # For real training, use num_epochs=100. 10 is a test value
image_size = 256  # We'll resize input images to this size (NK - kept it at same resolution)
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 2
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 1
mlp_head_units = [
    2048,
    1024,
]  # Size of the dense layers of the final classifier

# Get the training dataset - make patches 
train_dataset = create_filtered_dataset(images_path=fields['new_images_path'], annotations_path=fields['new_annotations_path'], subset_prefix='train', img_size=256, max_bbox=1, exclude_type='none', no_patch=False)
train_dataset = train_dataset.batch(batch_size)

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.expand_dims(tf.range(start=0, limit=self.num_patches, delta=1), axis=0)
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config
    
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)

# Compute the mean and the variance of the training data for normalization.
# data_augmentation.layers[0].adapt(x_train)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def create_vit_classifier():
    inputs = keras.Input(shape=input_shape)
    print(inputs.shape)
    augmented = data_augmentation(inputs)
    print(augmented.shape)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(augmented) # Since we already made the patches I just passed the inputs in directly 
    print(encoded_patches.shape)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        print(x1.shape)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        print(attention_output.shape)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        print(x2.shape)
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        print(x3.shape)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        print(x3.shape)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
        print(encoded_patches.shape)

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    print('representation shape', representation.shape)

    ## Jank reshaping operations to save my computer
    representation = tf.reshape(representation, (batch_size, representation.shape[1], representation.shape[2]*representation.shape[3]))
    print(representation.shape)
    representation = layers.Dense(32)(representation) # Reduce that last dimension -> 32 
    print(representation.shape)
    representation = layers.Flatten()(representation) 
    print('flattened', representation.shape)
    representation = layers.Dropout(0.5)(representation)
    print('dropout', representation.shape)
    # Add MLP.
    features = mlp(representation, hidden_units=[16, 16], dropout_rate=0.5)
    print('feats', features.shape)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    print('logits', logits.shape)
    print(logits)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


def run_experiment(model):
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate, decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[
            keras.metrics.RootMeanSquaredError()
        ],
    )

    for epoch in range(num_epochs): 
        for batch, (patches_x, bbox_context, bboxes_y, labels) in enumerate(train_dataset): 
            print('patches shape', patches_x.shape)
            patches_x = tf.reshape(patches_x, (batch_size, 256, 256, 3))
            print('patches shape', patches_x.shape)
            loss, r_squared = model.train_on_batch(patches_x, bboxes_y) # <- completely hangs here, no idea what's happening
            print(loss)

vit_classifier = create_vit_classifier()
run_experiment(vit_classifier)