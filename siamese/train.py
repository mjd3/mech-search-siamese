import numpy as np
import os
import datetime
import argparse
import matplotlib.pyplot as plt
import multiprocessing

import keras
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda
from keras.optimizers import Adam, RMSprop

from autolab_core import YamlConfig

import utils
from dataset import ImageDataset, DataGenerator
from resnet_fused import ResNet50Fused

def train(config):

    model_dir = config['model']['dir']
    model_name = config['model']['name']
    dataset_path = config['dataset']['path']

    train_config = config['training']

    # Parameters
    params = {'dim': train_config['img_dims'],
            'batch_size': train_config['batch_size'],
            'shuffle': train_config['shuffle_training_inputs']}

    # Datasets
    train_dataset = ImageDataset(dataset_path, 'train')
    train_dataset.prepare(1000)

    val_dataset = ImageDataset(dataset_path, 'validation')
    val_dataset.prepare(1000)

    # Generators
    training_generator = DataGenerator(train_dataset, **params)
    validation_generator = DataGenerator(val_dataset, **params)

    if config['model']['weights'] == 'random':
        feature_model = ResNet50Fused(include_top=False, weights=None, input_shape=train_config['img_dims'])
    elif config['model']['weights'] == 'imagenet':
        feature_model = ResNet50Fused(include_top=False, weights='imagenet', input_shape=train_config['img_dims'])
    elif config['model']['weights'] == 'last':
        if os.path.exists(os.path.join(model_dir, model_name + '.h5')):
            feature_model = load_model(os.path.join(model_dir, model_name + '.h5'))
        else:
            print('No model exists at the given path; starting new randomly initialized model')
            feature_model = ResNet50Fused(include_top=False, weights=None, input_shape=train_config['img_dims'])
    else:
        print('Invalid weights value given, exiting....')
        exit(1)

    img_a_in = Input(shape = train_config['img_dims'], name = 'ImageA_Input')
    img_b_in = Input(shape = train_config['img_dims'], name = 'ImageB_Input')
    
    if train_config['use_two_gpus']:
        with tf.device('/gpu:0'):
            img_a_feat = feature_model(img_a_in)
        with tf.device('/gpu:1'):
            img_b_feat = feature_model(img_b_in)
    else:
        img_a_feat = feature_model(img_a_in)
        img_b_feat = feature_model(img_b_in)
    if train_config['distance'] == 'l1':
        distance = Lambda(utils.l1_distance)([img_a_feat, img_b_feat])
    elif train_config['distance'] == 'l2':
        distance = Lambda(utils.l2_distance)([img_a_feat, img_b_feat])
    else:
        print('Distance function not supported')
        exit(1)
    # distance = Dense(3840, activation = 'linear')(distance)
    # distance = BatchNormalization()(distance)
    # distance = Activation('relu')(distance)
    distance = Dense(1, activation = 'sigmoid')(distance)
    similarity_model = Model(inputs = [img_a_in, img_b_in], outputs = distance, name = 'Similarity_Model')
    similarity_model.compile(loss=utils.contrastive_loss, optimizer=Adam(lr=train_config['learning_rate']), metrics=[utils.accuracy])
    similarity_model.summary()

    # Directory for training logs
    now = datetime.datetime.now()
    log_dir = os.path.join(model_dir, "{}{:%Y%m%dT%H%M}".format(model_name.lower(), now))

    # Create log_dir if not exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Path to save after each epoch. Include placeholders that get filled by Keras.
    checkpoint_path = os.path.join(log_dir, "{}_*epoch*.h5".format(model_name.lower()))
    checkpoint_path = checkpoint_path.replace("*epoch*", "{epoch:04d}")

    callbacks = [
        keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False),
        keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=0, save_weights_only=True),
    ]

    similarity_model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=50,
                        use_multiprocessing=True,
                        callbacks=callbacks,
                        workers=multiprocessing.cpu_count())

    similarity_model.save(os.path.join(model_dir, model_name+'.h5'))  # creates a HDF5 file 'my_model.h5'

if __name__ == "__main__":

    # parse the provided configuration file, set tf settings, and train
    conf_parser = argparse.ArgumentParser(description="Train Siamese model")
    conf_parser.add_argument("--config", action="store", default="cfg/train.yaml",
                               dest="conf_file", type=str, help="path to the configuration file")
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)
    # set_tf_config()
    train(config)
