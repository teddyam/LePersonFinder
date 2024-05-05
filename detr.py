import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow import keras
from keras.layers import Dense, Rescaling
from prep import train_dataset, test_dataset
from params import hp, fields
from losses import hungarian_matching

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
      self.last_attn_scores = attn_scores # Cache the attention scores for plotting later
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

## This class connects the Encoder and Decoder: takes in context sequence for keys and values as opposed to the input sequence.  
class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(query=x, key=context, value=context, return_attention_scores=True) # Multi-headed attention 
    self.last_attn_scores = attn_scores # Cache the attention scores for plotting later 
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
      tf.keras.layers.Dense(dff, activation='relu'),
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
# Decoder: Consists of Encoder Layers, which are blocks of Self Attention -> FFN -> Residual Connection -> LN
#--------------------------------------------------------------------------------------------------------------# 
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, emb_sz, num_heads, dff, dropout_rate=0.1):
    super(DecoderLayer, self).__init__()
    self.causal_self_attention = CausalSelfAttention(num_heads=num_heads, key_dim=emb_sz, dropout=dropout_rate) # Masked self attention 
    self.cross_attention = CrossAttention(num_heads=num_heads, key_dim=emb_sz, dropout=dropout_rate)  # Encoder-decoder attention 
    self.ffn = FeedForward(emb_sz, dff)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)
    self.last_attn_scores = self.cross_attention.last_attn_scores # Cache the last attention scores for plotting later
    x = self.ffn(x)  # shape: (batch_size, seq_len, emb_sz) 
    return x

class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, emb_sz, num_heads, dff, vocab_size,
               dropout_rate=0.1, all_patches):
    super(Decoder, self).__init__()
    self.emb_sz = emb_sz
    self.num_layers = num_layers
    self.all_patches = all_patches
    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, emb_sz=emb_sz)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [DecoderLayer(emb_sz=emb_sz, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate) for _ in range(num_layers)]
    self.ffout = tf.keras.layers.Dense(1)
    self.last_attn_scores = None

  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, emb_sz)
    x = self.dropout(x)
    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)
    self.last_attn_scores = self.dec_layers[-1].last_attn_scores
    if self.all_patches: 
      x = tf.transpose(x, perm=[0, 2, 1])
      x = self.ffout(x)
    return x # The shape of x is (batch_size, target_seq_len, emb_sz) (ideally) - if not (batch_size, 1, emb_sz)

#--------------------------------------------------------------------------------------------------------------# 
# Transformer Model: this is the core model used in detr 
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
    self.decoder = Decoder(num_layers=num_layers, emb_sz=emb_sz,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate,
                           all_patches=True)
    self.jank_transpose = tf.keras.layers.Dense(fields['MAX_COUNT_BBOXES'])

  # Compile the model 
  def compile(self, optimizer, loss, metrics):
    self.optimizer = optimizer
    self.loss_function = loss 
    self.accuracy_function = metrics[0]

  # Forward pass 
  def call(self, inputs):
    
    # Unpack the input
    patches, bbox_context = inputs  

    # Reshape to flatten the mid dimensions 
    bbox_context = tf.reshape(bbox_context, (bbox_context.shape[0], -1, 1))  # flatten & upsample the context, input: the following should return (batch_sz, 256, emb_sz)
    patches = tf.reshape(patches, (patches.shape[0], -1, 1))  # flatten & upsample the context, input: the following should return (batch_sz, 256, emb_sz)
    
    # Embed the context and patches 
    bbox_context = self.emb_context(bbox_context) # Embed your context 
    patches = self.emb_patches(patches) # Embed your img patches 

    # Flip the bbox_context, patches -> Dense layer (otherwise we can't get back the 50 sz we want)
    patches = tf.transpose(patches, [0, 2, 1]) # <- [batch_sz, 132, 196]
    bbox_context = tf.transpose(bbox_context, [0, 2, 1]) 

    # Project -> number of bboxes space (N)
    patches = self.jank_transpose(patches)
    bbox_context = self.jank_transpose(bbox_context)

    # Flip back (rly shit workaround)
    patches = tf.transpose(patches, [0, 2, 1])
    bbox_context = tf.transpose(bbox_context, [0, 2, 1]) 

    embedded = bbox_context + patches 
    
    # Pass into the encoder 
    encoder_output = self.encoder(embedded) # The following should return: (batch_sz, 256, emb_sz), {(batch_size, context_len, emb_sz)}
    return encoder_output # Return the final output and the attention weights.

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

#--------------------------------------------------------------------------------------------------------------# 
'''
NK: This is our DETR module (model). Notably it is customized so that it omits the use of a custom NestedTensor class that 
the original authors of the paper used for simplicity. 
'''
#--------------------------------------------------------------------------------------------------------------# 
class DETR(tf.keras.Model): 
  def __init__(self, backbone, transformer, num_classes, loss_fn, aux_loss=False):
    super().__init__()

    self.transformer = transformer

    self.class_embed = tf.keras.layers.Dense(num_classes, activation='softmax') # <- This will transform the output of the transformer -> class embedding (YES human or NO human)
    self.bbox_embed = tf.keras.Sequential([
      tf.keras.layers.Dense(4, activation='relu'), 
    ]) # <- This will convert the output of the transformer -> bbox embedding (coordinates)

    self.input_proj = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same') # <- Project the img into the same subspace as the bbox context
    self.backbone = backbone
    self.aux_loss = aux_loss
    self.loss_fn = loss_fn

  def call(self, inputs):

    # * NOTES ON THEIR CODE (this is not at all what's written below -- ignore)
    # Samples is tuple of: 
        # Samples: [batch_sz, 3, H, W]
        # Binary mask: [batch_sz, H, W]
    # 1. First, pass this tuple -> the backbone, which was pre-built and passed to this module beforehand 
        # .build_backbone() will a) instantiate a Backbone, b) instantiate a Joiner, which takes in a Backbone -> model
    # 2. Then extract the tuple of: Samples, Mask from backbone out list (not the pos list) SPECIFICALLY the last one...
        # Pass in => Transformer model 
            # 1. Projection of the Samples
            # 2. Mask
            # 3. Take the last pos embedding too...(corresponding to in puts)
    # 3. Then you will take the output of the Transformer => the following: 
        # 1. Pass into an embed class => outputs the class 
        # 2. Pass into a coords class => outputs the coords 
    # 4. Init a dictionary for logits and boxes and then just return that
    
    # Extract outputs from the pre-trained backbone model
    src, bbox_contents = inputs
    final_outputs = self.backbone(src)
    src = final_outputs[len(final_outputs)-2] # (64, 14, 14, 1024) 
    print("Backbone output we're extracting", src.shape)
    src = self.input_proj(src) # The last layer will have dimension: (b, 7, 7, 2048). Project that into the same input space as bbox content
    print("Projection output", src.shape)
    tout = self.transformer((src, bbox_contents))
    print("Transformer output:", tout.shape)
    outputs_class = self.class_embed(tout)
    print("Output class", outputs_class.shape)
    outputs_coords = self.bbox_embed(tout)
    print("Outputs coords", outputs_coords.shape)
    out = {'classes': outputs_class, 'coords': outputs_coords}
    print("OUTPUT:", out['classes'], out['coords'])
    return out
  
  def train(self, train_set):
    ''' Per epoch train pass'''

    train_set.shuffle(buffer_size=3) # Shuffle the dataset every epoch
    for batch_number, (imgs, overlap_counts, bboxes, one_hot_bboxes) in enumerate(train_set):
      with tf.GradientTape() as tape: 
        output = self((imgs, overlap_counts))
        loss = self.loss_fn(bboxes, one_hot_bboxes, output['coords'], output['classes'])[0]
        print("Loss is: ", loss)

      grads = tape.gradient(loss, self.trainable_variables) # Compute the gradients 
      self.optimizer.apply_gradients(zip(grads, self.trainable_variables))


  

'''
Method that runs the model from top to bottom 
'''
from prep import XMLtoJSON, create_filtered_dataset
from keras.applications import ResNet50

def build_detr(): 
  train_dataset = create_filtered_dataset(fields['images_path'], fields['annotations_path'], 'train', no_patch=True)
  batched_train_dataset = train_dataset.batch(batch_size=hp['batch_sz'], drop_remainder=False)

  # Initialize a ResNet50 model & pre-train it 
  model = ResNet50(include_top=False, weights="imagenet")
  layer_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'] 
  layers = [model.get_layer(name).output for name in layer_names]
  wrapper_model = tf.keras.Model(inputs=model.input, outputs=layers)
  transformer_model = Transformer(num_layers=hp['num_layers'], emb_sz=hp['emb_sz'], dff=hp['num_features'], num_heads=hp['num_att_heads'], input_vocab_size=hp['num_classes'], target_vocab_size=hp['num_classes'], dropout_rate=hp['dropout_rate'])
  detr_model = DETR(wrapper_model, transformer_model, hp['num_classes'], hungarian_matching, True)

  detr_model.train(batched_train_dataset)

  # Iterate through the dataset and pass it through the model 
  for imgs, overlap_counts, bboxes, one_hot_bboxes in batched_train_dataset: 
    out_dict = detr_model((imgs, overlap_counts))


'''
Run the DETR model 
'''
build_detr() 






  

