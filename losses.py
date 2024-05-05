import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment

def box_area(boxes):
    """ 
    Helper function used to calculate a bboxes' area given two corners.
    
    Params:
    - boxes: Tensor of shape (N, 4) containing N bounding boxes in (x_topleft, y_topleft, x_bottomright, y_bottomright) format.
    Returns:
    - areas: Tensor of shape (N,) containing the area of each bounding box.
    """

    xmin, ymin, xmax, ymax = tf.split(boxes, 4, axis=1)
    
    # Compute the width and height of each bounding box
    width = tf.squeeze(xmax - xmin, axis=1)
    height = tf.squeeze(ymax - ymin, axis=1)
    
    # Compute the area of each bounding box
    areas = width * height
    
    return areas

def box_iou(box_preds, box_trues):
    """
    Compute the intersection over union (IoU) between pairs of bounding boxes with matching indices.

    Args:
    - box_preds: Tensor of shape (N, 4) containing N predicted bounding boxes in (x_topleft, y_topleft, x_bottomright, y_bottomright) format.
    - box_trues: Tensor of shape (N, 4) containing N ground truth bounding boxes in (x_topleft, y_topleft, x_bottomright, y_bottomright) format.

    Returns:
    - iou: Tensor of shape (N,) containing the IoU between each pair of bounding boxes with matching indices.
    """
    # Compute areas of bounding boxes
    area_preds = box_area(box_preds)
    area_trues = box_area(box_trues)

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


def np_tf_linear_sum_assignment(cost_matrix_batch):
    """
    Apply Hungarian algorithm (linear_sum_assignment) to a batch of cost matrices.

    Args:
    - cost_matrix_batch: A 3D numpy array of shape (batch_size, num_preds, num_trues).

    Returns:
    - batch_row_ind: A list of arrays, each containing the row indices of the matched pairs for each batch.
    - batch_col_ind: A list of arrays, each containing the column indices of the matched pairs for each batch.
    - batch_target_selector: A 3D numpy array of shape (batch_size, num_preds) with the matched target selectors.
    - batch_pred_selector: A 3D numpy array of shape (batch_size, num_trues) with the matched prediction selectors.
    """
    batch_size = cost_matrix_batch.shape[0]
    batch_row_ind = []
    batch_col_ind = []
    batch_target_selector = np.zeros((batch_size, cost_matrix_batch.shape[1]), dtype=bool)
    batch_pred_selector = np.zeros((batch_size, cost_matrix_batch.shape[2]), dtype=bool)

    for i in range(batch_size):
        row_ind, col_ind = linear_sum_assignment(cost_matrix_batch[i])
        batch_row_ind.append(row_ind)
        batch_col_ind.append(col_ind)
        batch_target_selector[i, row_ind] = True
        batch_pred_selector[i, col_ind] = True

    return batch_row_ind, batch_col_ind, batch_target_selector, batch_pred_selector


def giou(box_preds, box_trues):
    """
    Calculate the Generalized IoU (GIoU) between all pairs of bounding boxes 

    Args:
    - box_preds: Tensor of shape (batch_size, N, 4) containing N predicted bounding boxes in (xmin, ymin, xmax, ymax) format.
    - box_trues: Tensor of shape (batch_size, M, 4) containing M ground truth bounding boxes in (xmin, ymin, xmax, ymax) format.

    Returns:
    - giou: Tensor of shape (batch_size, N, M) containing the Generalized IoU (GIoU) between all pairs of bounding boxes.
    """
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


def l1_loss(box_preds, box_trues):
    """
    Calculate the L1 loss between all pairs of bounding boxes using TensorFlow.

    Args:
    - box_preds: Tensor of shape (batch_size, num_preds, 4) containing predicted bounding boxes.
    - box_trues: Tensor of shape (batch_size, num_trues, 4) containing ground truth bounding boxes.

    Returns:
    - loss: Tensor of shape (batch_size, num_preds, num_trues) containing the L1 loss between all pairs.
    """
    box_preds_expanded = tf.expand_dims(box_preds, axis=2)  # shape: (batch_size, num_preds, 1, 4)
    box_trues_expanded = tf.expand_dims(box_trues, axis=1)  # shape: (batch_size, 1, num_trues, 4)

    loss = tf.reduce_sum(tf.abs(box_preds_expanded - box_trues_expanded), axis=-1)  # shape: (batch_size, num_preds, num_trues)
    return loss


def crossentropy_loss(class_preds, class_trues):
    """
    Calculate cross-entropy loss between predictions and true classes directly for binary classification.

    Args:
    - class_preds: Tensor of shape (batch_size, num_preds, 2) containing class probabilities for N predicted bounding boxes.
    - class_trues: Tensor of shape (batch_size, num_trues) containing true class labels (0 or 1) for M ground truth bounding boxes.

    Returns:
    - loss_matrix: Tensor of shape (batch_size, num_preds, num_trues) containing cross-entropy losses.
    """
    # Ensure class_preds are probabilities (apply softmax if they are logits)
    class_preds = tf.nn.softmax(class_preds, axis=-1)

    # Convert class_trues to one-hot encoding
    class_trues = tf.cast(class_trues, tf.int32)
    class_trues_one_hot = tf.one_hot(class_trues, depth=2)

    # Expand dimensions to allow pairwise computation
    expanded_preds = tf.expand_dims(class_preds, axis=2)  # shape: (batch_size, num_preds, 1, 2)
    expanded_trues = tf.expand_dims(class_trues_one_hot, axis=1)  # shape: (batch_size, 1, num_trues, 2)

    # Compute cross-entropy loss pairwise
    loss_matrix = -tf.reduce_sum(expanded_trues * tf.math.log(expanded_preds + 1e-9), axis=-1)

    # Check for perfect matches and set their loss to zero
    is_perfect_match = tf.reduce_all(tf.equal(expanded_preds, expanded_trues), axis=-1)
    loss_matrix = tf.where(is_perfect_match, 0.0, loss_matrix)

    return loss_matrix


# Optimized Matching Function
def hungarian_matching(t_bbox, t_class, p_bbox, p_class, fcost_class=1, fcost_bbox=5, fcost_giou=2):
    cost_class = crossentropy_loss(p_class, t_class)
    cost_bbox = l1_loss(p_bbox, t_bbox)
    cost_giou = -giou(p_bbox, t_bbox)

    print("CE", cost_class.shape)
    print("BBOX", cost_bbox.shape)
    print("IOU", cost_giou.shape)

    cost_matrix = fcost_bbox * cost_bbox + fcost_class * cost_class + fcost_giou * cost_giou

    selectors = tf.numpy_function(np_tf_linear_sum_assignment, [cost_matrix], [tf.int64, tf.int64, tf.bool, tf.bool])
    target_indices = tf.cast(selectors[0], tf.int32)
    pred_indices = tf.cast(selectors[1], tf.int32)
    target_selector = selectors[2]
    pred_selector = selectors[3]

    optimized_cost = 0
    for i in range (len(target_indices)):
        optimized_cost += cost_matrix[target_indices[i]][pred_indices[i]]

    return optimized_cost, pred_indices, target_indices, pred_selector, target_selector, t_bbox, t_class






