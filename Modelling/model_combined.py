import keras
from keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout, Concatenate, Lambda, Flatten, ZeroPadding1D, MaxPooling1D, BatchNormalization, ThresholdedReLU, Masking, Add, LSTM
from keras.models import Model
from keras.layers import Layer
from keras import losses
from keras import backend as K
import tensorflow as tf

from model import FrameSliceLayer, interaction_term, compute_pad_mask, apply_pad_mask, global_avg_pool_masked, convolve_and_mask, inception_block

def mask(input_tensors):
    tensor = input_tensors[0]
    mask = input_tensors[1]
    return K.tf.multiply(tensor, mask)

def framed_pooled_conv_model(n_conv_layers=3, 
                        kernel_size=[8,8,8], n_filters=128, dilations=[1, 1, 1],
                        padding="same", use_batchnorm=False,
                        use_inception=False, skip_connections="", 
                        n_dense_layers=1, fc_neurons=[64], fc_drop_rate=0.2,
                        single_output=True,
                        prefix=""):
    def conv_inner():
        input_seq = Input(shape=(None, 4), name=prefix+"input_seq")
        conv_features = input_seq
        # Compute presence of zero padding
        pad_mask = Lambda(compute_pad_mask, name=prefix+"compute_pad_mask")(conv_features)
        # Convolution
        layer_list = []
        for i in range(n_conv_layers):
            if skip_connections:
                conv_features_shortcut = conv_features #shortcut connections
            if use_inception:
                conv_features = inception_block(conv_features, pad_mask, n_filters, suffix=str(i))   
            else:
                conv_features, layer_list = convolve_and_mask(conv_features, pad_mask, n_filters,
                                                              kernel_size[i], 
                                                              suffix=str(i), prefix=prefix,
                                                              padding=padding, 
                                                              dilation=dilations[i], 
                                                              batchnorm=use_batchnorm, layer_list=layer_list)   
            if skip_connections == "residual" and i > 0:
                conv_features = Add(name=prefix+"add_residual_"+str(i))([conv_features,
                                                                         conv_features_shortcut])
            elif skip_connections == "dense":
                conv_features = Concatenate(axis=-1, name=prefix+"concat_dense_"+str(i))([conv_features,
                                                                                   conv_features_shortcut])
        # Frame based masking    
        frame_masked_features = FrameSliceLayer(name=prefix+"frame_masking")(conv_features)
        # Pooling
        pooled_features = []
        max_pooling = GlobalMaxPooling1D(name=prefix+"pool_max_frame_conv")
        avg_pooling = Lambda(global_avg_pool_masked, name=prefix+"pool_avg_frame_conv")
        pooled_features = pooled_features + \
                        [max_pooling(frame_masked_features[i]) for i in range(len(frame_masked_features))] \
                        + [avg_pooling([frame_masked_features[i], pad_mask]) for i in
                           range(len(frame_masked_features))]
        pooled_features = Concatenate(axis=-1, name=prefix+"concatenate_pooled")(pooled_features)
        # Prediction (Dense layer)
        predict = pooled_features
        for i in range(n_dense_layers):
            predict = Dense(fc_neurons[i], activation='relu', name=prefix+"fully_connected_"+str(i))(predict)
            predict = Dropout(rate=fc_drop_rate, name=prefix+"fc_dropout_"+str(i))(predict)
        if single_output:
            predict = Dense(1, name=prefix+"mrl_output_unscaled")(predict) 
        return [input_seq], predict
    return conv_inner

def pooled_conv_model(n_conv_layers=3, 
                        kernel_size=[8,8,8], n_filters=128, dilations=[1, 1, 1],
                        padding="same", use_batchnorm=False,
                        only_maxpool=True,
                        use_inception=False, skip_connections="",
                        n_dense_layers=1, fc_neurons=[64], fc_drop_rate=0.2,
                        single_output=True,
                     prefix=""):
    def conv_inner():
        input_seq = Input(shape=(None, 4), name=prefix+"input_seq")
        conv_features = input_seq
        # Compute presence of zero padding
        pad_mask = Lambda(compute_pad_mask, name=prefix+"compute_pad_mask")(conv_features)
        # Convolution
        layer_list = []
        for i in range(n_conv_layers):
            if skip_connections:
                conv_features_shortcut = conv_features #shortcut connections
            if use_inception:
                conv_features = inception_block(conv_features, pad_mask, n_filters, suffix=str(i))   
            else:
                conv_features, layer_list = convolve_and_mask(conv_features, pad_mask, n_filters,
                                                              kernel_size[i], 
                                                              suffix=str(i), prefix=prefix,
                                                              padding=padding, 
                                                              dilation=dilations[i], 
                                                              batchnorm=use_batchnorm, layer_list=layer_list)   
            if skip_connections == "residual" and i > 0:
                conv_features = Add(name=prefix+"add_residual_"+str(i))([conv_features,
                                                                         conv_features_shortcut])
            elif skip_connections == "dense":
                conv_features = Concatenate(axis=-1, name=prefix+"concat_dense_"+str(i))([conv_features,
                                                                                   conv_features_shortcut])
        # Pooling
        max_pooling = GlobalMaxPooling1D(name=prefix+"pool_max_conv")(conv_features)
        if not only_maxpool:
            avg_pooling = Lambda(global_avg_pool_masked, name=prefix+"pool_avg_conv")(
                [conv_features, pad_mask])
            pooled_features = [max_pooling, avg_pooling]
            pooled_features = Concatenate(axis=-1, name=prefix+"concatenate_pooled")(pooled_features)
        # Prediction (Dense layer)
        predict = pooled_features
        for i in range(n_dense_layers):
            predict = Dense(fc_neurons[i], activation='relu', name=prefix+"fully_connected_"+str(i))(predict)
            predict = Dropout(rate=fc_drop_rate, name=prefix+"fc_dropout_"+str(i))(predict)
        if single_output:
            predict = Dense(1, name=prefix+"mrl_output_unscaled")(predict) 
        return [input_seq], predict
    return conv_inner

def model_input(shape, name):
    return Input(shape=shape, name=name)

def kmer_linear_model(kmer_inputs,
                      n_kmer_layers=0,
                      kmer_activations=["relu"]
                      kmer_neurons=[64],
                      kmer_drop_rate=0.2,
                      single_output=False,
                      prefix=""):
    def kmer_inner():
        # Input
        kmer_predict = Concatenate(axis = -1, name=prefix+"combine_kmer_inputs")(kmer_inputs)
        # Kmer layers
        for i in range(n_kmer_layers):
            kmer_predict = Dense(fc_neurons[i], activation='relu', name=prefix+"fc_"+str(i))
            (kmer_predict)
            kmer_predict = Dropout(rate=fc_drop_rate, name=prefix+"fc_drop_"+str(i))(kmer_predict)
        if single_output:
            kmer_predict = Dense(1, name=prefix+"kmer_out_")(kmer_predict) 
        return kmer_inputs, kmer_predict
    return kmer_inner

def combined_conv_kmer(conv_model, kmer_model,
                      n_combine_layers=0,
                      combine_neurons=[64],
                      kmer_drop_rate=0.2,
                      prefix=""):
    def combined_inner():
        conv_inputs, conv_output = conv_model()
        kmer_inputs, kmer_output = kmer_model()
        combined_predict = Concatenate(axis = -1, name=prefix+"combine_conv_kmer")(
            [conv_output, kmer_output])
        # Kmer layers
        for i in range(n_combine_layers):
            combined_predict = Dense(fc_neurons[i], activation='relu', name=prefix+"fc_convkmer"+str(i))
            (combined_predict)
            combined_predict = Dropout(rate=fc_drop_rate, name=prefix+"drop_convkmer"+str(i))(kmer_predict)
        return conv_inputs + kmer_inputs, combined_predict
    return combined_inner
        

def transfer_model(utr5_model, cds_model, utr3_model,
                   transfer_inputs,
                   n_transfer_layers=1,
                   transfer_neurons=[64],
                   transfer_drop_rate=0.2,
                   n_combine_layers=0,
                   combine_neurons=[64],
                   combine_drop_rate=0.2,
                   loss='mean_squared_error'):
    # Get 5utr motives
    input_5utr, motives_5utr = utr5_model()
    # Combine with existing
    output_5utr = Concatenate(axis = -1, name="combine_with_transfer")([motives_5utr] + transfer_inputs)
    for i in range(n_transfer_layers):
        output_5utr = Dense(transfer_neurons[i], activation='relu', 
                            name="transfer_dense_"+str(i))(output_5utr)
        output_5utr = Dropout(rate=transfer_drop_rate, name="transfer_drop"+str(i))(output_5utr)
    #output_5utr = Dense(1, name="5utr_output")(output_5utr) 
    # Predict for cds
    input_cds, output_cds = cds_model()
    # Predict for 3utr
    input_3utr, output_3utr = utr3_model()
    # Combine
    output_combined = Concatenate(axis = -1, name="combine_outputs")([output_5utr, output_cds, output_3utr])
    for i in range(n_combine_layers):
        output_combined = Dense(combine_neurons[i], activation='relu', 
                            name="combine_dense_"+str(i))(output_combined)
        output_combined = Dropout(rate=combine_drop_rate, name="combine_drop"+str(i))(output_combined)
    mrl_prediction = Dense(1, name="output")(output_combined)
    """ Model """
    inputs = input_5utr + input_cds + input_3utr + transfer_inputs
    model = Model(inputs=inputs, outputs=mrl_prediction)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=loss, optimizer=adam)
    return model



#######################################################################################################

def scaling_regression_unit(mrl_prediction, input_library):
    # Scaling regression
    predict_scaled = Lambda(interaction_term, name="interaction_term")([mrl_prediction, input_library])
    predict_scaled = Concatenate(axis = 1, name="prepare_regression")([predict_scaled, input_library])
    predict_scaled = Dense(1, name="scaling_regression", use_bias=False)(predict_scaled)
    return predict_scaled

def thresholded_loss(threshold, power):
    def thresholded_mse(y_true, y_pred):
        masked_true = tf.multiply(y_true, y_pred[:,1])
        masked_pred = tf.multiply(y_pred[:,0], y_pred[:,1])
        loss = losses.mean_squared_error(masked_true, masked_pred)
        loss = loss + K.clip(K.pow(loss - threshold, power),0,None)
        return loss
    return thresholded_mse

def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)

###############################################################

def combined_model(utr5_model, cds_model, utr3_model,
                  threshold_shortcut, threshold_main, power=3):
    # Inputs
    input_5utr = Input(shape=(None, 4), name="input_5utr")
    input_cds = Input(shape=(None, 4), name="input_cds")
    input_3utr = Input(shape=(None, 4), name="input_3utr")
    input_library = Input(shape=(7, ), name="input_library")
    input_seqtype = Input(shape=(1, ), name="input_seqtype")
    inputs = [input_5utr, input_library, input_cds, input_3utr, input_seqtype]
    # Predict for 5utr
    output_5utr = utr5_model(input_5utr)
    mrl_shortcut = scaling_regression_unit(output_5utr, input_library)
    mrl_shortcut = Concatenate(axis=1,name="Concatenate_shortcut_mask")([mrl_shortcut, input_seqtype])
    # Predict for cds
    output_cds = cds_model(input_cds)
    # Predict for 3utr
    output_3utr = utr3_model(input_3utr)
    # Combine
    output_combined = Concatenate(axis = -1, name="combine_outputs")([output_5utr, output_cds, output_3utr])
    mrl_prediction = Dense(1, name="unscaled_output")(output_combined)
    #mrl_prediction = scaling_regression_unit(mrl_prediction, input_library)
    mask_main = Lambda(lambda x: 1-x, name="main_mask")(input_seqtype)
    mrl_main = Concatenate(axis=1,name="Concatenate_main_mask")([mrl_prediction, mask_main])
    """ Model """
    loss_shortcut = thresholded_loss(threshold_shortcut, power)
    loss_main = thresholded_loss(threshold_main, power)
    model = Model(inputs=inputs, outputs=[mrl_shortcut, mrl_main])
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=[loss_shortcut, loss_main], optimizer=adam)
    return model

def combined_model_noshortcut(utr5_model, cds_model, utr3_model, loss,
                             include_scaling_regression=False):
    # Inputs
    input_seqtype = Input(shape=(1, ), name="input_seqtype")
    inputs = [input_seqtype]
    # Predict for 5utr
    input_5utr, output_5utr = utr5_model()
    # Predict for cds
    input_cds, output_cds = cds_model()
    output_cds = Lambda(mask, name="mask_cds")([output_cds, input_seqtype])
    # Predict for 3utr
    input_3utr, output_3utr = utr3_model()
    output_3utr = Lambda(mask, name="mask_3utr")([output_3utr, input_seqtype])
    # Combine
    output_combined = Concatenate(axis = -1, name="combine_outputs")([output_5utr, output_cds, output_3utr])
    mrl_prediction = Dense(1, name="unscaled_output")(output_combined)
    if include_scaling_regression:
        input_library = Input(shape=(7, ), name="input_library")
        inputs = inputs + input_library
        mrl_prediction = scaling_regression_unit(mrl_prediction, input_library)
    """ Model """
    inputs = [input_5utr, input_cds, input_3utr] + inputs
    model = Model(inputs=inputs, outputs=mrl_prediction)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=loss, optimizer=adam)
    return model


#######################################################
