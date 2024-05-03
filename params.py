## Hyperparameters & general stats
hp = {
    'batch_sz': 64, 
    'num_classes': 2,
    'vocab_sz': 35,  
    'num_att_heads': 1, 
    'key_dim': 10, 
    'query_dim': 10, 
    'value_dim': 10, 
    'output_sz': 8, 
    'window_sz': 256,
    'num_features': 768, 
    'emb_sz': 132, 
    'num_layers': 2, 
    'learning_rate': 0.01, 
    'num_epochs': 20, 
    'dropout_rate': 0.5,
}

## General fields to use for model running
fields = {
    'data_folder_path': "data/heridal_keras_retinanet_voc",
    'images_path': "data/heridal_keras_retinanet_voc/JPEGImages",
    'annotations_path': "data/heridal_keras_retinanet_voc/Annotations/JSON",
    'ImageSets': "data/heridal_keras_retinanet_voc/ImageSets/Main",
    'xml_folder': "data/heridal_keras_retinanet_voc/Annotations",
    'json_folder': "data/heridal_keras_retinanet_voc/Annotations/JSON",
    'IMAGE_SIZE': 224, # <- resize to match resnet input 
    'PATCH_SIZE': 16,
    'NUM_PATCHES': 196,
    'PROJECTION_DIM': 64,
    'MAX_COUNT_BBOXES': 35
}