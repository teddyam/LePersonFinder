import tensorflow as tf
from keras import layers, models
from keras.applications import ResNet50
from typing import List, Tuple
from backbone import Backbone, Joiner, build_backbone
from prep import XMLtoJSON, create_filtered_dataset
from keras.applications import ResNet50

'''
TOP LEVEL FIELDS (taken from preprocessing)
'''
data_folder_path = "data/heridal_keras_retinanet_voc"
images_path = data_folder_path + "/JPEGImages"
annotations_path = data_folder_path + "/Annotations/JSON"
ImageSets = data_folder_path + "/ImageSets/Main"
xml_folder = data_folder_path + "/Annotations"
json_folder = annotations_path
IMAGE_SIZE = 224 # <- resize to match resnet input 
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 64
MAX_COUNT_BBOXES = 35

'''
Runs RESNET thru 1 pass (using 1 sample) 
Expected outputs: 
1. conv1 (3x): (b, 112, 112, 64)
2. conv2 (4x): (b, 56, 56, 256)
3. conv3 (6x): (b, 28, 28, 512)
4. conv4 (3x): (b, 7, 7, 2048)
'''
def test_resnet(): 
    print("=============== TEST RESNET ===============")
    train_dataset = create_filtered_dataset(images_path, annotations_path, 'train')
    batched_train_dataset = train_dataset.batch(batch_size=64, drop_remainder=False)

    model = ResNet50(include_top=False, weights="imagenet")
    layer_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'] 
    layers = [model.get_layer(name).output for name in layer_names]
    wrapper_model = tf.keras.Model(inputs=model.input, outputs=layers)

    for imgs, overlap_counts, bboxes, one_hot_bboxes in batched_train_dataset.take(1): 
        final_outputs = wrapper_model(imgs)
        for name, output in zip(layer_names, final_outputs):
            print(f"Layer {name} output shape: {output.shape}")
    print("\n")

def test_whole_model(): 
    pass 

'''
Tests the size of bbox coords and if padding works for them. 
'''
def test_bbox_coords(): 
    print("=============== TEST BBOX ===============")
    MAX_COUNT_BBOXES=50
    sample_bbox = tf.random.normal([1,4])
    padded_bbox = tf.pad(sample_bbox, tf.constant([[0, MAX_COUNT_BBOXES - sample_bbox.shape[0]],[0,0]]), 'CONSTANT')
    print("Shape of padded bbox:", padded_bbox.shape)
    print("\n")

'''
Run all the tests
'''
test_resnet()
test_bbox_coords()