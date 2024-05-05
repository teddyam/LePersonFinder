import tensorflow as tf
import numpy as np
import os
import json
import time
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow import keras
from keras.layers import Dense, Rescaling
from prep import train_dataset, test_dataset
from params import hp

#--------------------------------------------------------------------------------------------------------------# 
# Positional embedding class that will look a token's embedding vector and add the corresponding position vector 
#--------------------------------------------------------------------------------------------------------------# 
class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, emb_sz):
    super().__init__()
    self.emb_sz = emb_sz
    self.embedding = tf.keras.layers.Dense(emb_sz)
    self.pos_encoding = self.positional_encoding(length=2048, depth=emb_sz)

  def positional_encoding(self, length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis] # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth # (1, depth)
    angle_rates = 1 / (10000**depths) # (1, depth)
    angle_rads = positions * angle_rates # (pos, depth)
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1) 
    return tf.cast(pos_encoding, dtype=tf.float32)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)
  
  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    x *= tf.math.sqrt(tf.cast(self.emb_sz, tf.float32)) # This factor sets the relative scale of the embedding and positonal_encoding.
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x
  
#--------------------------------------------------------------------------------------------------------------# 
# Attention Modules
#--------------------------------------------------------------------------------------------------------------# 
## Base Attention Class
class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

## Self-attention
class GlobalSelfAttention(BaseAttention): 
    def call(self, x): 
      attn_output, attn_scores = self.mha(query=x, key=x, value=x, return_attention_scores=True)
      # Cache the attention scores for plotting later.
      self.last_attn_scores = attn_scores 
      x = self.add([x, attn_output])
      x = self.layernorm(x)
      return x
    
## Masked self-attention
class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

#--------------------------------------------------------------------------------------------------------------# 
# FFN: Class that combines FNN + Residual Connection + Layer Normalization 
#--------------------------------------------------------------------------------------------------------------# 
class FeedForward(tf.keras.layers.Layer):
  def __init__(self, emb_sz, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation=tf.keras.layers.activations.gelu),
      tf.keras.layers.Dense(emb_sz),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    # [LAYER NORM -> MHA] -> LAYER NORM -> FFN
    x = self.layer_norm(x) # Layer norm on the residual output
    ffout = self.seq(x) # Get output from FFN
    x = self.add([x, ffout]) # Residual connection from FFN output and input
    return x
  
#--------------------------------------------------------------------------------------------------------------# 
# Encoder: Consists of Encoder Layers, which are blocks of Self Attention -> FFN -> Residual Connection -> LN
#--------------------------------------------------------------------------------------------------------------# 
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, emb_sz, num_heads, dff, dropout_rate=0.1):
    super().__init__()
    self.self_attention = BaseAttention(
        num_heads=num_heads,
        key_dim=emb_sz,
        dropout=dropout_rate)
    self.layer_norm = tf.keras.layers.LayerNormalization()
    self.ffn = FeedForward(emb_sz, dff)

  def call(self, x):
    # LAYER NORM -> MHA -> [LAYER NORM -> FFN]
    x = self.layer_norm(x) 
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, emb_sz, num_heads,
               dff, vocab_size, dropout_rate=0.1):
    super().__init__()
    self.emb_sz = emb_sz
    self.num_layers = num_layers
    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size, emb_sz=emb_sz)
    self.enc_layers = [
        EncoderLayer(emb_sz=emb_sz,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # x is token-IDs shape: (batch, seq_len). When you pass into pos_embedding, will do x = x + pos_embedding(x)
    x = self.pos_embedding(x)  # Should result in shape of (batch_size, seq_len, emb_sz).
    x = self.dropout(x)  
    for i in range(self.num_layers):
      x = self.enc_layers[i](x)
    return x  # Should result in shape of (batch_size, seq_len, emb_sz).
  
#--------------------------------------------------------------------------------------------------------------# 
# Transformer Model: 
#--------------------------------------------------------------------------------------------------------------# 
class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, emb_sz, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
    super().__init__()
    self.emb_context = tf.keras.layers.Dense(emb_sz)
    self.emb_patches = tf.keras.layers.Dense(emb_sz)
    self.encoder = Encoder(num_layers=num_layers, emb_sz=emb_sz,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate)
    # self.classification head for the ViT architecture
    self.classification_head = tf.keras.layers.Sequential([
      tf.keras.layers.LayerNormalization(epsilon=1e-6),
      tf.keras.layers.Dense(), 
      tf.keras.layers.Dropout(), 
      tf.keras.layers.Dense() 
    ])
    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  # Compile the model 
  def compile(self, optimizer, loss, metrics):
    self.optimizer = optimizer
    self.loss_function = loss 
    self.accuracy_function = metrics[0]

  # Forward pass 
  def call(self, inputs):
    patches, bbox_context = inputs  
    print(patches)
    bbox_context = tf.reshape(bbox_context, (bbox_context.shape[0], -1, 1))  # flatten & upsample the context, input: the following should return (batch_sz, 256, emb_sz)
    bbox_context = self.emb_context(bbox_context) # Embed your context 
    patches = self.emb_patches(patches) # Embed your img patches 
    embedded = bbox_context + patches
    x = self.encoder(embedded) # The following should return: (batch_sz, 256, emb_sz), {(batch_size, context_len, emb_sz)}
    logits = tf.squeeze(x, axis=2) # The following should return: (batch_size, 256, 8) {(batch_size, target_len, target_vocab_size)}
    logits = self.final_layer(logits)
    smax = tf.nn.softmax(logits)  
    return smax # Return the final output and the attention weights.

  def loss(label, pred):
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='none')
    loss = loss_object(label, pred)
    return tf.reduce_sum(loss)/(label.shape[0])
  
  def accuracy(label, pred):
    pred_indices = tf.argmax(pred, axis=1)
    true_indices = tf.argmax(label, axis=1)
    correct_predictions = tf.equal(pred_indices, true_indices)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
  
  # Train the model 
  def train(self, train_dataset, params, train_metrics_dict): 
      train_loss_per_epoch, train_accuracy_per_epoch = 0, 0
      train_dataset.shuffle(buffer_size=3) # Shuffle the dataset every epoch
      num_batches = 0
      for patches, bbox_context, labels in train_dataset:
          with tf.GradientTape() as tape: 
            out = self((patches, bbox_context))
            loss = self.loss(labels, out)
            accuracy = self.accuracy(labels, out) 
            train_loss_per_epoch+=loss.numpy()
            train_accuracy_per_epoch+=accuracy.numpy()
            num_batches += 1
          grads = tape.gradient(loss, self.trainable_variables) # Compute the gradients 
          self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
      train_loss_per_epoch = train_loss_per_epoch / num_batches
      train_accuracy_per_epoch = train_accuracy_per_epoch / num_batches
      train_metrics_dict['Loss'].append(train_loss_per_epoch)
      train_metrics_dict['Accuracy'].append(train_accuracy_per_epoch)

  
from prep import XMLtoJSON, create_filtered_dataset

def build_vit(): 
  pass 