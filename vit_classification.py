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
from params import hp_vit, fields
from prep import XMLtoJSON, create_filtered_dataset
from metrics import linear_combo_loss, r_squared, binary_accuracy, cross_entropy

#--------------------------------------------------------------------------------------------------------------# 
# Positional embedding class that will look a token's embedding vector and add the corresponding position vector 
#--------------------------------------------------------------------------------------------------------------# 
class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, input_size, emb_sz):
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
  def __init__(self, emb_sz, dff, dropout_rate=hp_vit['dropout_rate']):
    super().__init__()
    # self.initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.02)
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(emb_sz, activation='relu'),
      tf.keras.layers.Dropout(dropout_rate),
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
  def __init__(self,*, emb_sz, num_heads, dff, dropout_rate=hp_vit['dropout_rate']):
    super().__init__()

    # Old attention class: 
    # self.self_attention = CausalSelfAttention(
    #     num_heads=num_heads,
    #     key_dim=emb_sz,
    #     dropout=dropout_rate)

    # General self attention: 
    self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim =emb_sz, dropout=dropout_rate)
    self.layer_norm = tf.keras.layers.LayerNormalization()
    self.ffn = FeedForward(emb_sz, dff)

  def call(self, x):
    # LAYER NORM -> MHA -> [LAYER NORM -> FFN]
    x = self.layer_norm(x) 
    x = self.self_attention(x,x)
    x = self.ffn(x)
    return x

class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, emb_sz, num_heads,
               dff, input_size, dropout_rate=hp_vit['dropout_rate']):
    super().__init__()
    self.emb_sz = emb_sz
    self.num_layers = num_layers
    self.pos_embedding = PositionalEmbedding(input_size=input_size, emb_sz=emb_sz)
    self.enc_layers = [
        EncoderLayer(emb_sz=emb_sz,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.jank_shape_adj = tf.keras.layers.Dense(1)

  def call(self, x):
    # x is token-IDs shape: (batch, seq_len). When you pass into pos_embedding, will do x = x + pos_embedding(x)
    x = self.pos_embedding(x)  # Should result in shape of (batch_size, seq_len, emb_sz).
    x = self.dropout(x)  
    for i in range(self.num_layers):
      x = self.enc_layers[i](x)
    
    # Necessary reshape operations to get the right output shape -> only when reg token is not being used explicitly
    # x = tf.transpose(x, [0, 2, 1])
    # x = self.jank_shape_adj(x)
    # x = tf.transpose(x, [0, 2, 1])

    return x  # Should result in shape of (batch_size, seq_len, emb_sz).
  
#--------------------------------------------------------------------------------------------------------------# 
# Transformer Model: 
#--------------------------------------------------------------------------------------------------------------# 
class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, emb_sz, num_heads, dff,
               input_size, target_size, dropout_rate=hp_vit['dropout_rate']):
    super().__init__()

    # init top level layers
    self.emb_sz = emb_sz 
    self.initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.02)
    self.emb_context = tf.keras.layers.Dense(emb_sz)
    self.emb_patches = tf.keras.layers.Dense(emb_sz)
    self.encoder = Encoder(num_layers=num_layers, emb_sz=emb_sz,
                           num_heads=num_heads, dff=dff,
                           input_size=input_size,
                           dropout_rate=dropout_rate)
    self.num_classes = target_size

    # make the appropriate reg token 
    self.cls_token = self.add_weight(name="reg_token", 
                                     shape=[1,1,emb_sz],
                                     initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.02), 
                                     dtype=tf.float32) 
    self.to_cls_token = tf.identity

    # self."classification" head for the ViT architecture
    self.cls_head = tf.keras.Sequential([
      tf.keras.layers.LayerNormalization(epsilon=1e-6),
      tf.keras.layers.Dense(self.emb_sz, activation='relu'), # <- emb sz doesn't rly matter here, don't need
      tf.keras.layers.Dropout(hp_vit['dropout_rate']), 
      tf.keras.layers.Dense(self.num_classes, activation='softmax') # <- should output a 4-tuple of coords, so no activation function needed here...should just be raw coords.
    ])

  # Forward pass 
  @tf.function
  def call(self, inputs):
    '''
    Some parts of this method (specifically cls_token logic) are inspired from: Credit: https://github.com/ashishpatel26/Vision-Transformer-Keras-Tensorflow-Pytorch-Examples/blob/main/Vision_Transformer_with_tf2.ipynb
    '''
    patches, bbox_context = inputs 
    
    # Process bbox_context -> add to cls_token.
    bbox_context = tf.reshape(bbox_context, (bbox_context.shape[0], -1, 1))  # (batch_sz, 256, 768)
    bbox_context = self.emb_context(bbox_context) # Embed your context 
    patches = self.emb_patches(patches) # Embed your img patches 
    embedded = bbox_context + patches # Add bbox_context to the patches themselves

    # Concatenate with class token and pass thru encoder 
    cls_tokens = tf.broadcast_to(self.cls_token, (embedded.shape[0], 1, self.emb_sz)) # Broadcast the self.cls_token first dimension -> batch dimension
    embedded = tf.concat([cls_tokens, embedded], axis=1) # (64, 257, 132). (batch_sz, seq_length, emb_sz)
    x = self.encoder(embedded) # The following should return: (batch_sz, 256, emb_sz), {(batch_size, context_len, emb_sz)}

    # Pass the REGRESSION token SPECIFICALLY through the regression head. Nothing else.  
    x = self.to_cls_token(x[:, 0])
    x = self.cls_head(x)
    x = tf.expand_dims(x, axis=1) # So that you get back out (64, 1, 4) instead of just (64, 4)
    return x
  
  # Compile the model 
  def compile(self, optimizer, loss, metrics):
    self.optimizer = optimizer
    self.loss_function = loss 
    self.accuracy_function = metrics[0]
  
  '''
  Training loop for the core model. 
  '''
  def train(self, train_dataset, train_metrics_dict): 
      train_loss_per_epoch, train_accuracy_per_epoch = 0, 0
      train_dataset.shuffle(buffer_size=3) # Shuffle the dataset every epoch
      num_batches = 0 
      for patches, bbox_context, bboxes, labels in train_dataset:
          with tf.GradientTape() as tape: 
            out = self((patches, bbox_context))
            labels = tf.expand_dims(labels, axis=1)
            loss = self.loss_function(out, labels)
            train_loss_per_epoch+=loss.numpy()
            accuracy = self.accuracy_function(out, labels) 
            train_accuracy_per_epoch+=accuracy.numpy()
            num_batches += 1
          grads = tape.gradient(loss, self.trainable_variables) # Compute the gradients 
          self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
      train_loss_per_epoch = train_loss_per_epoch / num_batches
      train_accuracy_per_epoch = train_accuracy_per_epoch / num_batches
      train_metrics_dict['Loss'].append(train_loss_per_epoch)
      train_metrics_dict['Accuracy'].append(train_accuracy_per_epoch)

  def test(self, test_dataset, test_metrics_dict): 
    test_loss_per_epoch, test_accuracy_per_epoch = 0, 0
    num_batches = 0
    for patches, bbox_context, bboxes, labels in test_dataset:
       out = self((patches, bbox_context))
       loss = self.loss_function(out, labels)
       labels = tf.expand_dims(labels, axis=1)
       accuracy = self.accuracy_function(out, labels) 
       test_loss_per_epoch+=loss.numpy()
       test_accuracy_per_epoch+=accuracy.numpy()
       num_batches += 1
    test_loss_per_epoch = test_loss_per_epoch / num_batches
    test_accuracy_per_epoch = test_accuracy_per_epoch / num_batches
    test_metrics_dict['Loss'].append(test_loss_per_epoch) # <- this loss won't really make sense, would just be loss over the whole test dataset. 
    test_metrics_dict['Accuracy'].append(test_accuracy_per_epoch)

def build_vit(): 

  # set seed for rando initialization 
  tf.random.set_seed(42)

  # set the datasets
  train_dataset = create_filtered_dataset(images_path=fields['new_images_path'], annotations_path=fields['new_annotations_path'], subset_prefix='train', img_size=1024, max_bbox=2, exclude_type='none', no_patch=False)
  batched_train_dataset = train_dataset.batch(batch_size=hp_vit['batch_sz'], drop_remainder=False)
  test_dataset = create_filtered_dataset(images_path=fields['new_images_path'], annotations_path=fields['new_annotations_path'], subset_prefix='test', img_size=1024, max_bbox=2, exclude_type='none', no_patch=False)
  batched_test_dataset = test_dataset.batch(batch_size=hp_vit['batch_sz'], drop_remainder=False)

  # instantiate model 
  vit_model = Transformer(num_layers=hp_vit['num_layers'], emb_sz=hp_vit['emb_sz'], dff=hp_vit['num_features'], num_heads=hp_vit['num_att_heads'], input_size=hp_vit['target_sz'], target_size=hp_vit['target_sz'], dropout_rate=hp_vit['dropout_rate'])
  
  # compile the model & init metrics - classification
  vit_model.compile(
      loss=cross_entropy,
      optimizer=hp_vit['optimizer'],
      metrics=[binary_accuracy]
  )

  # train the model - over num_epochs.
  training_start = time.time()
  train_metrics_dict = {'Loss': [], 'Accuracy':[]}
  for e in range(hp_vit['num_epochs']): 
    batched_train_dataset.shuffle(buffer_size=64) # Shuffle the dataset every epoch  
    vit_model.train(batched_train_dataset, train_metrics_dict)
    epoch_loss = train_metrics_dict['Loss'][e]
    epoch_r_squared = train_metrics_dict['Accuracy'][e]
    print(f'Epoch: {e} | Loss: {epoch_loss} | Accuracy: {epoch_r_squared}')
  training_end = time.time()
  print(f'Training took: {(training_end-training_start) // 60} minutes')

  # test the model.  
  testing_start = time.time()
  test_metrics_dict = {'Loss': [], 'Accuracy':[]}
  vit_model.test(batched_test_dataset, test_metrics_dict)
  test_loss = test_metrics_dict['Loss']
  test_r_squared = test_metrics_dict['Accuracy']
  testing_end = time.time()
  print(f'Loss: {test_loss} | Accuracy: {test_r_squared}')
  print(f'Testing took: {(testing_end-testing_start) // 60} minutes')

# Build the vit
build_vit()

# Old printlines 
# print('this is the max of patches', tf.reduce_max(patches))
# tf.print(patches, summarize=-1)
# print('this is the min of patches', tf.reduce_min(patches))
# print('this is our model weights: ', self.get_weights())