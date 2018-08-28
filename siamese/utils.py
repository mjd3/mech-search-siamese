import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt

def l2_distance(vects):
    x, y = vects
    # return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
    return K.square(x - y)

def l1_distance(vects):
    x, y = vects
    # return K.mean(K.abs(x - y), axis=1, keepdims=True)
    return K.abs(x - y)

# def dist_output_shape(shapes):
#     shape1, shape2 = shapes
#     return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(K.maximum(margin - y_pred, 0)) + (1 - y_true) * K.square(y_pred))

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() > 0.5].mean()

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred > 0.5, y_true.dtype)))

def set_tf_config():

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        set_session(sess)

def show_output(im1s, im2s, preds, gts):
    fig, m_axs = plt.subplots(2, im1s.shape[0], figsize = (12,6))
    for im1, im2, p, gt, (ax1, ax2) in zip(im1s, im2s, preds, gts, m_axs.T):
        ax1.imshow(im1)
        ax1.set_title('Image A\n Actual: %3.0f%%' % (100*gt))
        ax1.axis('off')
        ax2.imshow(im2)
        ax2.set_title('Image B\n Predicted: %3.0f%%' % (100*p))
        ax2.axis('off')
    plt.show()