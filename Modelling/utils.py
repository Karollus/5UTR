import random
import re

import numpy as np
import pandas as pd
import scipy.stats as stats

import keras
from keras import backend as K
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns

""" FEATURE ENCODING """

# Dictionary returning one-hot encoding of nucleotides 
nuc_dict = {'a':[1.0,0.0,0.0,0.0],'c':[0.0,1.0,0.0,0.0],'g':[0.0,0.0,1.0,0.0], 'u':[0.0,0.0,0.0,1.0], 't':[0.0,0.0,0.0,1.0], 'n':[0.0,0.0,0.0,0.0], 'x':[1/4,1/4,1/4,1/4]}

# Dictionary encoding the experiments
experiment_dict = {"egfp_unmod_1":0, "egfp_unmod_2": 1, "mcherry_1":2, "mcherry_2":3, "ga": 4, "human":5} 


def encode_seq(seq, max_len=0):
    length = len(seq)
    if max_len > 0:
        padding_needed = max_len - length
        seq = "N"*padding_needed + seq
    seq = seq.lower()
    one_hot = np.array([nuc_dict[x] for x in seq]) # get stacked on top of each other
    return one_hot

def encode_experiment(df, col="library"):
    mask = np.array([experiment_dict[x] for x in df[col]])
    indicator = np.zeros((len(df),6))
    indicator[np.arange(len(df)), mask] = 1
    return indicator

def encode_df(df, col='utr', libcol="library", output_col="rl", variable_len=False):
    max_len = 0
    if variable_len:
        max_len = len(max(df[col], key=len))
    one_hot = np.stack(df[col].apply(encode_seq, max_len=max_len), axis = 0)
    indicator = encode_experiment(df, col=libcol)
    rl = None
    if output_col is not None:
        rl = np.array(df[output_col])
    return {"seq":one_hot, "library":indicator, "rl":rl}

""" TRAINING """

def train(model, data, libraries, batch_size=128, epochs=3, use_val=True, early_stop=True, patience=3, file="best_model.h5"):
    inputs = [np.concatenate([data["train"][library]["seq"] for library in libraries]), 
              np.concatenate([data["train"][library]["library"] for library in libraries])]
    outputs = np.concatenate([data["train"][library]["rl"] for library in libraries])
    if use_val:
        val_libs = set(data["val"].keys()) & set(libraries)
        val_inputs = [np.concatenate([data["val"][library]["seq"] for library in val_libs]), 
                  np.concatenate([data["val"][library]["library"] for library in val_libs])]
        val_outputs = np.concatenate([data["val"][library]["rl"] for library in val_libs])
    if early_stop:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
        mc = ModelCheckpoint(file, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    model.fit(inputs, outputs, batch_size, epochs, verbose=2, validation_data=(val_inputs, val_outputs), callbacks=[es, mc])
        
def freeze_all_except_scaling(model):
    for layer in model.layers:
        if layer.name != "scaling_regression":
            layer.trainable = False
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model

def retrain_only_scaling(model, 
                         data, 
                         libraries = ['egfp_unmod_1', 'mcherry_1', 'mcherry_2', 'egfp_unmod_2', 'ga'], 
                         batch_size=128, 
                         epochs=2):
    model = freeze_all_except_scaling(model)
    inputs = [np.concatenate([data["train"][library]["seq"] for library in libraries] + [data["val"]["human"]["seq"]]), 
              np.concatenate([data["train"][library]["library"] for library in libraries] + [data["val"]["human"]["library"]])]
    outputs = np.concatenate([data["train"][library]["rl"] for library in libraries] + [data["val"]["human"]["rl"]])
    model.fit(inputs, outputs, batch_size, epochs, verbose=2)
    return model

""" EVALUATION """

def rSquared(predictions, targets):
    ssr = np.sum(np.square(predictions - targets))
    ybar = np.average(targets)
    sst = np.sum(np.square(targets - ybar))
    return 1 - (ssr/sst)

def pearson_r(x, y, squared=True):
    pearson_r, p_val = stats.pearsonr(x, y)
    if squared:
        pearson_r = pearson_r ** 2
    return pearson_r, p_val

def evaluate(model, data, libraries, do_test=False):
    set_type = "test" if do_test else "val"
    preds = []
    for library in libraries:
        inputs = [data[set_type][library]["seq"], data[set_type][library]["library"]]
        outputs = data[set_type][library]["rl"]
        predict = model.predict(inputs)
        print("Rsquared on set " + library + " : " + str(rSquared(predict.reshape(-1), outputs.reshape(-1))) + \
              ", Pearson: " + str(pearson_r(predict.reshape(-1), outputs.reshape(-1))[0]))
        preds.append(predict)
    #predictions_array = np.concatenate([])
    #return pd.DataFrame({"predicted": predictions_array, "actual": outputs})

def plot(df, x_name='predicted', y_name="actual", add_line=True):
    c1 = (0.3, 0.45, 0.69)
    c2 = 'r'
    g = sns.JointGrid(x=x_name,y=y_name, data=df, space=0, ratio=6, height=8)
    g.plot_joint(plt.scatter,s=20, color=c1, linewidth=0.2, alpha='0.5', edgecolor='white')
    f = g.fig
    ax = f.gca()
    if add_line:
        x = np.linspace(*ax.get_xlim())
        plt.plot(x, x)

def eval_snv(model, data, snv_df):
    preds = []
    for library in ["snv", "wt"]:
        inputs = [data[library]["seq"], data[library]["library"]]
        predictions = model.predict(inputs)
        preds.append(predictions)
        print("Pearson " + library + " : " + str(pearson_r(predictions.reshape(-1), data[library]["rl"].reshape(-1))[0]))
    log_pred_diff = np.log2(preds[0]/preds[1])
    snv_df["log_pred_diff_new"] =  log_pred_diff.reshape(-1)
    print("Rsquared fold-change: " + str(pearson_r(snv_df["log_pred_diff_new"], snv_df["log_obs_diff"])[0]))

def plot_snv(snv_df):
    f, ax = plt.subplots()
    f.set_size_inches((10,10))
    point_size = 40
    sub = snv_df
    path_list = ['Pathogenic', 'Likely pathogenic', 'Pathogenic, other', 'Pathogenic/Likely pathogenic']
    benign_list = ['Benign/Likely benign', 'Benign', 'Likely Benign']
    uncertain_list = ['Conflicting interpretations of pathogenicity', 'Uncertain significance']
    path = sub[(sub['clin_sig'] == path_list[0]) | (sub['clin_sig'] == path_list[1]) |
           (sub['clin_sig'] == path_list[2]) | (sub['clin_sig'] == path_list[3])]
    non = sub[(sub['clin_sig'] == benign_list[0]) | (sub['clin_sig'] == benign_list[1]) | (sub['clin_sig'] == benign_list[2])]
    unsure = sub[(sub['clin_sig'] == uncertain_list[0]) | (sub['clin_sig'] == uncertain_list[1])]
    ax.scatter(unsure['log_obs_diff'], unsure['log_pred_diff_new'], alpha=0.8, color='grey', label='Uncertain',
               linewidth=1, edgecolors='k', s=point_size)
    ax.scatter(non['log_obs_diff'], non['log_pred_diff_new'], alpha=0.8, color='dodgerblue', label='Benign / likely',
               linewidth=1, edgecolors='k', s=point_size)
    ax.scatter(path['log_obs_diff'], path['log_pred_diff_new'], alpha=0.8, color='orangered', label='Pathogenic / likely',
               linewidth=1, edgecolors='k', s=point_size)
    ax.set_ylabel('Pred. MRL Change (log$_2$)')
    ax.set_xlabel('Obs. MRL Change (log$_2$)')
    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    ax.set_xticks(range(-2,3))
    ax.set_yticks(range(-2,3))
    ax.hlines(y=0,xmin=-2, xmax=2, linestyles='dashed', linewidth=1)
    ax.vlines(x=0,ymin=-2, ymax=2, linestyles='dashed', linewidth=1)
    ax.legend(loc=(-0.02,0.79), handletextpad=-0.2, fontsize=11.5)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def eval_ptr(model, data, ptr_df):
    inputs = [data["seq"], data["library"]]
    predictions = model.predict(inputs)
    ptr_df["MRL"] = predictions.reshape(-1)
    pearson = pearson_r(ptr_df["MRL"],ptr_df["PTR"])
    print("PTR Pearson: " + str(pearson[0]) + " p-val: " + str(pearson[1]))
    
    
""" DEBUG """

def check_layer(model, data, layer_names, node=0):
    return_dict = {}
    for name in layer_names:
        target_obj = model.get_layer(name).get_output_at(node)
        if type(target_obj) == list:
            target = [tensor for tensor in target_obj] 
        else:
            target = [target_obj]
        check_fn = K.function([model.get_layer("input_seq").input, model.get_layer("input_experiment").input], target)
        return_dict[name] = check_fn([data["val_input"]["seq"], data["val_input"]["indicator"]])
    return return_dict

def check_uAUG_detection(trained_model, kozak=False, seq_length=200, samples=1000):
    if kozak:
        out_df = pd.DataFrame({"idx": list(range(-seq_length, -8, 1)), "in_frame": [i % 3 == 0 for i in range(-seq_length, -8, 1)]})
    else:
        out_df = pd.DataFrame({"idx": list(range(-seq_length, -2, 1)), "in_frame": [i % 3 == 0 for i in range(-seq_length, -2, 1)]})
    predictions = []
    for i in range(samples):
         # Make a random sequence
        seq = ''.join(random.choices(["A","C","T","G"], k=seq_length))
        # Remove existing atg
        atg_present = [m.start() for m in re.finditer('ATG', seq)]
        for idx in atg_present:
            seq = seq[:idx] + ''.join(random.choices(["C","T","G"], k=3)) + seq[idx+3:]
        # Iterate over all possible atg/kozak locations
        uAUG_seqs = []
        if kozak:
            for i in range(len(seq) - 8):
                new_seq = seq
                new_seq = new_seq[:i] + "GCCACCATG" + new_seq[i+9:] 
                uAUG_seqs.append(new_seq)
            df = pd.DataFrame({"utr": uAUG_seqs, "library":"egfp_unmod_1"})
        else:
            for i in range(len(seq) - 2):
                new_seq = seq
                new_seq = new_seq[:i] + "ATG" + new_seq[i+3:] 
                uAUG_seqs.append(new_seq)
            df = pd.DataFrame({"utr": uAUG_seqs, "library":"egfp_unmod_1"})
        # Predict
        data = encode_df(df, output_col=None)
        predictions.append(trained_model.predict([data["seq"], data["library"]]))
    #get average prediction
    out_df["prediction"] = (sum(predictions)/samples).reshape(-1)
    return out_df

def uAUG_plot(uAUG_pred):
    in_frame = uAUG_pred.loc[uAUG_pred['in_frame']==True]
    out_frame = uAUG_pred.loc[uAUG_pred['in_frame']!=True]
    plt.plot( 'idx', 'prediction', data=in_frame, color='skyblue', linewidth=2)
    plt.plot( 'idx', 'prediction', data=out_frame, color='red', linewidth=2)