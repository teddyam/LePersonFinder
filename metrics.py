import tensorflow as tf
from keras import backend as K

'''
BOX AREA
'''
def box_area(self, boxes):
    xmin, ymin, xmax, ymax = tf.split(boxes, 4, axis=1)

    # Compute the width and height of each bounding box
    width = tf.squeeze(xmax - xmin, axis=1)
    height = tf.squeeze(ymax - ymin, axis=1)

    # Compute the area of each bounding box
    areas = width * height

    return areas

'''
BOX IOU
'''
def box_iou(self, box_preds, box_trues):

    # Compute areas of bounding boxes
    area_preds = self.box_area(box_preds)
    area_trues = self.box_area(box_trues)

    # Compute intersection coordinates
    xmin_inter = tf.maximum(box_preds[:, 0], box_trues[:, 0])
    ymin_inter = tf.maximum(box_preds[:, 1], box_trues[:, 1])
    xmax_inter = tf.minimum(box_preds[:, 2], box_trues[:, 2])
    ymax_inter = tf.minimum(box_preds[:, 3], box_trues[:, 3])

    # Compute intersection areas
    width_inter = tf.maximum(0.0, xmax_inter - xmin_inter)
    height_inter = tf.maximum(0.0, ymax_inter - ymin_inter)
    area_inter = width_inter * height_inter

    # Compute union areas
    area_union = area_preds + area_trues - area_inter

    # Compute IoU
    iou = area_inter / area_union
    return iou, area_union

'''
Linear combo loss
'''
def linear_combo_loss(box_preds, box_trues): 
    giou_coeff, l1_coeff = -1, 3

    '''
    GIOU loss
    '''
    def giou_loss(box_preds, box_trues):
        box_preds_expanded = tf.expand_dims(box_preds, axis=2)  # shape: (batch_size, N, 1, 4)
        box_trues_expanded = tf.expand_dims(box_trues, axis=1)  # shape: (batch_size, 1, M, 4)

        # Calculate intersection
        inter_mins = tf.maximum(box_preds_expanded[..., :2], box_trues_expanded[..., :2])
        inter_maxs = tf.minimum(box_preds_expanded[..., 2:], box_trues_expanded[..., 2:])
        intersection = tf.maximum(inter_maxs - inter_mins, 0.0)
        intersection_area = tf.reduce_prod(intersection, axis=-1)

        # Calculate individual box areas
        area_preds = tf.reduce_prod(box_preds_expanded[..., 2:] - box_preds_expanded[..., :2], axis=-1)
        area_trues = tf.reduce_prod(box_trues_expanded[..., 2:] - box_trues_expanded[..., :2], axis=-1)

        # Calculate union
        union = area_preds + area_trues - intersection_area
        union = tf.maximum(union, 1e-10)

        # Calculate enclosing box
        enclose_min = tf.minimum(box_preds_expanded[..., :2], box_trues_expanded[..., :2])
        enclose_max = tf.maximum(box_preds_expanded[..., 2:], box_trues_expanded[..., 2:])
        enclose_area = tf.reduce_prod(tf.maximum(enclose_max - enclose_min, 0.0), axis=-1)
        enclose_area = tf.maximum(enclose_area, 1e-10)

        # Generalized IoU calculation
        giou = intersection_area / union - (enclose_area - union) / enclose_area

        return giou
    
    '''
    Huber/L1 loss
    '''
    def l1_loss(box_preds, box_trues):
        box_preds_expanded = tf.expand_dims(box_preds, axis=2)  # shape: (batch_size, num_preds, 1, 4)
        box_trues_expanded = tf.expand_dims(box_trues, axis=1)  # shape: (batch_size, 1, num_trues, 4)

        loss = tf.reduce_sum(tf.abs(box_preds_expanded - box_trues_expanded), axis=-1)  # shape: (batch_size, num_preds, num_trues)
        return loss

    return giou_coeff * tf.reduce_mean(tf.linalg.diag_part(giou_loss(box_preds, box_trues))) + l1_coeff * tf.reduce_mean(tf.linalg.diag_part(l1_loss(box_preds, box_trues)))


'''
Mean Squared Error
'''
def mse(y_pred, y_label):
    loss = tf.keras.losses.MeanSquaredError()(y_label, y_pred)
    return loss / (y_label.shape[0])

'''
Cross Entropy Loss
'''
def cross_entropy(y_pred, y_label):
    loss = tf.keras.losses.CategoricalCrossentropy()(y_label, y_pred)
    return loss / (y_label.shape[0])

'''
Accuracy (custom)
'''
def binary_accuracy(y_pred, y_label):
    pred_indices = tf.argmax(y_pred, axis=1)
    true_indices = tf.argmax(y_label, axis=1)
    correct_predictions = tf.equal(pred_indices, true_indices)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy

'''
R squared(custom)
'''
def r_squared(y_pred, y_true):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return (1 - SS_res/(SS_tot + K.epsilon())) 