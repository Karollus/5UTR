import numpy as np
import keras

from keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout, Concatenate, Lambda, Flatten, ZeroPadding1D, MaxPooling1D, BatchNormalization, ThresholdedReLU, Masking, Add
from keras.models import Model
from keras.layers import Layer
from keras import losses
from keras import backend as K
import tensorflow as tf

# Layer which slices input tensor into three tensors, one for each frame w.r.t. the canonical start
class FrameSliceLayer(Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape) 
    
    def call(self, x):
        shape = K.shape(x)
        x = K.reverse(x, axes=1) # reverse, so that frameness is related to fixed point (start codon)
        frame_1 = tf.gather(x, K.arange(start=0, stop=shape[1], step=3), axis=1)
        frame_2 = tf.gather(x, K.arange(start=1, stop=shape[1], step=3), axis=1)
        frame_3 = tf.gather(x, K.arange(start=2, stop=shape[1], step=3), axis=1)
        return [frame_1, frame_2, frame_3]
    
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], None, input_shape[2]),(input_shape[0], None, input_shape[2]),(input_shape[0], None, input_shape[2])]

# Function to compute an interaction term between a value and a one-hot vector
def interaction_term(tensors):
    prediction = tensors[0]
    experiment_indicator = tensors[1]
    prediction = K.repeat_elements(prediction, rep=K.int_shape(experiment_indicator)[1], axis=1)
    return K.tf.multiply(prediction, experiment_indicator)

# Masking to prevent zero padding to influence results
def compute_pad_mask(x):
    return K.sum(x, axis=2)

def apply_pad_mask(input_tensors):
    tensor = input_tensors[0]
    mask = input_tensors[1]
    mask = K.expand_dims(mask, axis=2)
    mask = K.repeat_elements(mask, rep=K.int_shape(tensor)[2], axis=2)
    return K.tf.multiply(tensor, mask)

# Average pooling that accounts for masking
def global_avg_pool_masked(input_tensors):
    tensor = input_tensors[0]
    mask = input_tensors[1]
    mask = K.expand_dims(mask, axis=2)
    mask = K.repeat_elements(mask, rep=K.int_shape(tensor)[2], axis=2)
    return K.sum(tensor, axis=1)/K.sum(mask, axis=1)

def convolve_and_mask(conv_features, pad_mask, n_filters, kernel_size, suffix, dilation=1):
    conv_features = Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=dilation, activation='relu', 
                           padding="causal", name="convolution_"+suffix)(conv_features)
    conv_features = Lambda(apply_pad_mask, name="apply_pad_mask_"+suffix)([conv_features, pad_mask]) # Mask padding
    return conv_features

def inception_block(conv_features, pad_mask, n_filters, suffix):
    conv_features_3 = convolve_and_mask(conv_features, pad_mask, n_filters[0], kernel_size=3, suffix="incept3_"+suffix)
    conv_features_5 = convolve_and_mask(conv_features, pad_mask, n_filters[1], kernel_size=5, suffix="incept5_"+suffix)
    conv_features_7 = convolve_and_mask(conv_features, pad_mask, n_filters[2], kernel_size=7, suffix="incept7_"+suffix)
    conv_features = Concatenate(name="incept_concat"+suffix)([conv_features_3, conv_features_5, conv_features_7])
    return conv_features

def create_model_masked_bordered(n_conv_layers=3, 
                        kernel_size=[8,8,8], n_filters=128, dilations=[1, 1, 1],
                        use_inception=False, skip_connections="", 
                        n_dense_layers=1, fc_neurons=[64], fc_drop_rate=0.2,
                        loss = 'mean_squared_error',
                        extract_tis_context=False):
    # Inputs
    input_seq = Input(shape=(None, 4), name="input_seq")
    input_experiment = Input(shape=(6, ), name="input_experiment")
    conv_features = input_seq
    # Compute presence of zero padding
    pad_mask = Lambda(compute_pad_mask, name="compute_pad_mask")(conv_features)
    # Convolution
    for i in range(n_conv_layers):
        if skip_connections:
            conv_features_shortcut = conv_features #shortcut connections
        if use_inception:
            conv_features = inception_block(conv_features, pad_mask, n_filters, suffix=str(i))   
        else:
            conv_features = convolve_and_mask(conv_features, pad_mask, n_filters, kernel_size[i], suffix=str(i), dilation=dilations[i])   
        if skip_connections == "residual" and i > 0:
            conv_features = Add(name="add_residual_"+str(i))([conv_features, conv_features_shortcut])
        elif skip_connections == "dense":
            conv_features = Concatenate(axis=-1, name="concat_dense_"+str(i))([conv_features, conv_features_shortcut])
    # Extract out area before TIS
    tis_conv = []
    if extract_tis_context:
        tis_conv = tis_conv + [Lambda(lambda x: x[:,-1,:], name="extract_tis_context")(conv_features)]
        conv_features = Lambda(lambda x: x[:,:-1,:], name="extract_conv_without_tis")(conv_features)
    # Frame based masking    
    frame_masked_features = FrameSliceLayer(name="frame_masking")(conv_features)
    # Pooling
    pooled_features = []
    max_pooling = GlobalMaxPooling1D(name="pool_max_frame_conv")
    avg_pooling = Lambda(global_avg_pool_masked, name="pool_avg_frame_conv")
    pooled_features = pooled_features + \
                    [max_pooling(frame_masked_features[i]) for i in range(len(frame_masked_features))] + \
                    [avg_pooling([frame_masked_features[i], pad_mask]) for i in range(len(frame_masked_features))]
    pooled_features = Concatenate(axis=-1, name="concatenate_pooled")(pooled_features)
    convolution_out = [pooled_features] + tis_conv
    # Concatenation
    feature_list = convolution_out
    if len(feature_list) > 1:
        concat_features = Concatenate(axis=-1, name="concatenate_all_features")(convolution_out)
    else:
        concat_features = feature_list[0]
    # Prediction (Dense layer)
    predict = concat_features
    for i in range(n_dense_layers):
        predict = Dense(fc_neurons[i], activation='relu', name="fully_connected_"+str(i))(predict)
        predict = Dropout(rate=fc_drop_rate, name="fc_dropout_"+str(i))(predict)
    predict = Dense(1, name="mrl_output_unscaled")(predict) 
    # Scaling regression
    predict = Lambda(interaction_term, name="interaction_term")([predict, input_experiment])
    predict = Concatenate(axis = 1, name="prepare_regression")([predict, input_experiment])
    predict = Dense(1, name="scaling_regression", use_bias=False)(predict)
    """ Model """
    inputs = [input_seq, input_experiment]
    model = Model(inputs=inputs, outputs=predict)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=loss, optimizer=adam)
    return model