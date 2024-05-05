import tensorflow as tf

# Hyperparameters & general stats for VIT. Note "vocab_sz" = "num_classes"
hp_vit = {
    'batch_sz': 64, 
    'target_sz': 4, # <- for each of the four coords to predict
    'num_att_heads': 3, 
    'key_dim': 10, 
    'query_dim': 10, 
    'value_dim': 10, 
    'window_sz': 256, # <- specs seq length processed at a time
    'num_features': 768, # <- 256 (window_sz) * 3 (num_channels)
    'emb_sz': 132, # <- arbitrarily selected
    'num_layers': 2, # <- specs number of encoder layers to use 
    'learning_rate': 0.01, 
    'num_epochs': 100, 
    'dropout_rate': 0.5,
    'optimizer': tf.keras.optimizers.Adam(learning_rate=0.01)
}

# Hyperparameters & general stats for DETR. Note "vocab_sz" = "num_classes"
hp_detr = {
    'batch_sz': 64, 
    'num_classes': 2, # <- since we'll predict HUMAN or NOT HUMAN here. Note that I just manually specified the sz of coord output in the detr class
    'num_att_heads': 1, 
    'key_dim': 10, 
    'query_dim': 10, 
    'value_dim': 10, 
    'window_sz': 196, # <- specs seq length processed at a time
    'num_features': 672, 
    'emb_sz': 132, 
    'num_layers': 2, # <- specs number of encoder layers to use 
    'learning_rate': 0.01, 
    'num_epochs': 20, 
    'dropout_rate': 0.5,
    'optimizer': tf.keras.optimizers.Adam(learning_rate=0.01)
}

# Link specifications 
fields = {
    'data_folder_path': "data/heridal_keras_retinanet_voc",
    'images_path': "data/heridal_keras_retinanet_voc/JPEGImages",
    'annotations_path': "data/heridal_keras_retinanet_voc/Annotations/JSON",
    'new_annotations_path': "data/heridal_keras_retinanet_voc/NewAnnotations/JSON",
    'new_images_path': "data/heridal_keras_retinanet_voc/NewJPEGImages",
    'ImageSets': "data/heridal_keras_retinanet_voc/ImageSets/Main",
    'xml_folder': "data/heridal_keras_retinanet_voc/Annotations",
    'json_folder': "data/heridal_keras_retinanet_voc/Annotations/JSON",
}