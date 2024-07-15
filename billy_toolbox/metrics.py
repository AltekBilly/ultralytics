import tensorflow as tf


class NormalizedMeanError(tf.keras.metrics.Metric):
    def __init__(self, pts=None, ref=None, name='normalized_mean_error', **kwargs):
        super(NormalizedMeanError, self).__init__(name=name, **kwargs)
        self.pts = pts
        self.ref = ref
        self.total_objects = self.add_weight(name='total_objects', shape=(), initializer='zeros', dtype=tf.int32)
        self.total_error = self.add_weight(name='total_error', shape=(), initializer='zeros')

    def reset_states(self):
        self.total_objects.assign(tf.zeros_like(self.total_objects))
        self.total_error.assign(tf.zeros_like(self.total_error))

    @tf.function
    def update_state(self, y_true, y_pred):
        """
        y_true, y_pred (float32 tensor): Landmarks. Shape: [bs, pts, 2].
        """
        bs = tf.shape(y_true)[0]
        self.total_objects.assign_add(bs)

        norm_factor = tf.norm(y_true[:, self.ref[0]] - y_true[:, self.ref[1]], axis=1)   # [bs]
        
        error = (tf.reduce_mean(tf.norm(y_true - y_pred, axis=2), axis=1) / norm_factor)
        self.total_error.assign_add(tf.reduce_sum(error))

    def result(self):
        return self.total_error / tf.cast(self.total_objects, tf.float32)


class MeanAbsoluteError(NormalizedMeanError):
    def __init__(self, input_size, name='mean_absolute_error', **kwargs):
        super(MeanAbsoluteError, self).__init__(name=name, **kwargs)
        self.input_size = input_size

    @tf.function
    def update_state(self, y_true, y_pred):
        """
        y_true, y_pred (float32 tensor): Landmarks. Shape: [bs, pts, 2].
        """
        bs = tf.shape(y_true)[0]
        self.total_objects.assign_add(bs)

        y_true = y_true * self.input_size
        y_pred = y_pred * self.input_size

        norm_factor = 1.
        error = (tf.reduce_mean(tf.reduce_sum(tf.abs(y_true - y_pred),axis=2), axis=1) / norm_factor)
        self.total_error.assign_add(tf.reduce_sum(error))


class MeanEuclideanError(NormalizedMeanError):
    def __init__(self, input_size, name='mean_euclidean_error', **kwargs):
        super(MeanEuclideanError, self).__init__(name=name, **kwargs)
        self.input_size = input_size

    @tf.function
    def update_state(self, y_true, y_pred):
        """
        y_true, y_pred (float32 tensor): Landmarks. Shape: [bs, pts, 2].
        """
        bs = tf.shape(y_true)[0]
        self.total_objects.assign_add(bs)

        y_true = y_true * self.input_size
        y_pred = y_pred * self.input_size

        norm_factor = 1.
        error = (tf.reduce_mean(tf.norm(y_true - y_pred, axis=2), axis=1) / norm_factor)
        self.total_error.assign_add(tf.reduce_sum(error))

class ObjectKeypointSimilarity(NormalizedMeanError):
    def __init__(self, input_size, name='object_keypoint_similarity', **kwargs):
        super(ObjectKeypointSimilarity, self).__init__(name=name, **kwargs)
        self.input_size = input_size
        self.sigmas = tf.ones(24) / 24
        self.area = input_size*input_size
        self.alpha = 1
        
    @tf.function
    def update_state(self, y_true, y_pred):
        """
        y_true, y_pred (float32 tensor): Landmarks. Shape: [bs, pts, 2].
        """
        bs = tf.shape(y_true)[0]
        self.total_objects.assign_add(bs)
        
        area = 1
        
        d = tf.square(y_pred[..., 0] - y_true[..., 0]) + tf.square(y_pred[..., 1] - y_true[..., 1])
        kpt_loss_factor = tf.cast(bs / bs, tf.float32) 
        e = d / (tf.square(2 * self.sigmas) * (area + 1e-32) * 2 * self.alpha )  # from cocoeval
        error = tf.reduce_sum(kpt_loss_factor * (1 - tf.exp(-e)))
        
        self.total_error.assign_add(error)
        
import os
import numpy as np

def load_keypoints_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        keypoints = []
        for line in lines:
            parts = line.strip().split()
            kp = np.array([(float(parts[i]), float(parts[i+1])) for i in range(5, len(parts), 3)]).reshape(-1, 2)
            keypoints.extend(kp)
            break
        return np.array(keypoints)

def load_keypoints_from_dirs(true_label_dir, pred_label_dir):
    true_label_files = sorted(os.listdir(true_label_dir))
    pred_label_files = sorted(os.listdir(pred_label_dir))

    all_true_keypoints = []
    all_pred_keypoints = []

    for true_file, pred_file in zip(true_label_files, pred_label_files):
        true_file_path = os.path.join(true_label_dir, true_file)
        pred_file_path = os.path.join(pred_label_dir, pred_file)

        true_keypoints = load_keypoints_from_txt(true_file_path)
        pred_keypoints = load_keypoints_from_txt(pred_file_path)


        all_true_keypoints.append(true_keypoints)
        all_pred_keypoints.append(pred_keypoints)

    all_true_keypoints = np.array(all_true_keypoints, dtype='float32')
    all_pred_keypoints = np.array(all_pred_keypoints, dtype='float32')

    return all_true_keypoints, all_pred_keypoints

def main(pts: int, nme_ref: list, input_size: int, targets, preds):
    # Evaluation
    metrics = [
        NormalizedMeanError(pts=pts, ref=nme_ref),
        MeanAbsoluteError(input_size=input_size),
        MeanEuclideanError(input_size=input_size),
        ObjectKeypointSimilarity(input_size=input_size),
    ]
    
    count = 0
    
    for m in metrics:
        m.update_state(targets, preds)

    count += targets.shape[0]
    print("# samples: %d/%d" % (count, targets.shape[0]), end='\r')
    print()

    for m in metrics:
        print("%s: %.4f" % (m.name, m.result()))

if __name__ == '__main__':
    home_path = os.path.expanduser("~")
    true_label_dir = f'{home_path}/dataset/FacialLandmark_for_yolov8-pose-20240628/labels/val'
    name = 'Altek_Landmark-FacialLandmark-test-20240709-stride64-3'
    pred_label_dir = f'./runs/altek_landmark/preidct_{name}/labels' 
    # pred_label_dir = f'./runs/pose/predict6/labels' #
    
    true_keypoints, pred_keypoints = load_keypoints_from_dirs(true_label_dir, pred_label_dir)
    
    main(pts=24, nme_ref=[10, 19], input_size=256, targets=true_keypoints, preds=pred_keypoints)