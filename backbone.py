import tensorflow as tf
from keras import layers, models
from keras.applications import ResNet50
from typing import List, Tuple

'''
NOT USED **
NK: Copied and pasted directly from the Facebook DETR repo. You want to ensure stability of pre-trained models, 
so freeze the parameters so that there's not a huge discrepancy as you're fine tuning on the new dataset.  
'''
class FrozenBatchNorm2D(tf.keras.layers.Layer):
    def __init__(self, num_channels, epsilon=1e-5):
        super(FrozenBatchNorm2D, self).__init__()
        self.weight = self.add_weight(name="weight", shape=[num_channels], initializer="ones", trainable=False)
        self.bias = self.add_weight(name="bias", shape=[num_channels], initializer="zeros", trainable=False)
        self.running_mean = self.add_weight(name="running_mean", shape=[num_channels], initializer="zeros", trainable=False)
        self.running_var = self.add_weight(name="running_var", shape=[num_channels], initializer="ones", trainable=False)
        self.epsilon = epsilon
    def call(self, x):
        scale = self.weight / tf.sqrt(self.running_var + self.epsilon)
        bias = self.bias - self.running_mean * scale
        return x * scale + bias

'''
NOT USED ? 
NK: This backbone base will iterate through the model's layers and do the following: 
    0. Do a forward pass thru the model, passing in x -> each successive layer: layer(x) -> layer(layer(x)), etc..
    1. If return interm layers is specified: return a list of the CONVOLUTIONAL layers in the network 
    2. If return interm layer is not specified: return the final layer within the network 
'''
class BackboneBase(tf.keras.Model):
    def __init__(self, backbone: tf.keras.Model, train_backbone: bool):
        super(BackboneBase, self).__init__()
        self.body = backbone
    def call(self, x):
        for layer in self.body.layers:
            x = layer(x)
        return x

'''
NOT USED ** 
NK: This specifies the Backbone for use. For our purposes we will use a ResNet50 architecture and 
specify a custom, FrozenBatchNorm layer using imagenet weights. 
'''
class Backbone(BackboneBase):
    def __init__(self, train_backbone: bool):
        backbone = ResNet50(include_top=False, weights="imagenet")
        super().__init__(backbone, train_backbone)

'''
NK: This is our Joiner wrapper class. Because it subclasses tf.keras.Sequential, it can call 
methods like self.layers, where it will be able to access the list of layers that Sequential 
requires to be specified. So "xs" in this case is our ResNet50 backbone: we will run our input
through the architecture, then because BackboneBase returns a list of layers we will iterate
through that and a) append that layer to the list "out" and b) append the embedding for that
layer to self.layers. 
'''
class Joiner(tf.keras.Sequential):
    def __init__(self, backbone: BackboneBase, position_embedding):
        super().__init__([backbone, position_embedding])
    def call(self, x):
        xs = self.layers[0](x)
        out = []
        pos = []
        for x, l in xs.items():
            out.append(l)
            pos.append(self.layers[1](l))
        return out, pos

def build_backbone(lr_backbone: float, dilation: bool) -> Tuple[Backbone, Joiner]:
    train_backbone = lr_backbone > 0
    position_embedding = tf.keras.layers.Embedding(1000, 512)
    backbone = Backbone(train_backbone)
    return backbone