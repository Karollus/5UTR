import random
import re

import numpy as np
import pandas as pd
import scipy.stats as stats

import keras
from keras import backend as K
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import Sequence

import matplotlib.pyplot as plt
import seaborn as sns

""" FEATURE ENCODING """

# Dictionary returning one-hot encoding of nucleotides 
nuc_dict = {'a':[1.0,0.0,0.0,0.0],'c':[0.0,1.0,0.0,0.0],'g':[0.0,0.0,1.0,0.0], 'u':[0.0,0.0,0.0,1.0], 't':[0.0,0.0,0.0,1.0], 'n':[0.0,0.0,0.0,0.0], 'x':[1/4,1/4,1/4,1/4]}

# Dictionary encoding the experiments
experiment_dict = {"egfp_unmod_1":0, "egfp_unmod_2": 1, "mcherry_1":2, "mcherry_2":3, "ga": 4, "human":5,
                  "doudna":6}

def encode_seq(seq, max_len=0, min_len=None):
    length = len(seq)
    if max_len > 0 and min_len is None:
        padding_needed = max_len - length
        seq = "N"*padding_needed + seq
    if min_len is not None:
        if len(seq) < min_len:
            seq = "N"*(min_len - len(seq)) + seq
        if len(seq) > min_len:
            seq = seq[(len(seq) - min_len):]
    seq = seq.lower()
    one_hot = np.array([nuc_dict[x] for x in seq]) # get stacked on top of each other
    return one_hot

def encode_experiment(df, col="library", n_libs=6):
    mask = np.array([experiment_dict[x] for x in df[col]])
    indicator = np.zeros((len(df),n_libs))
    indicator[np.arange(len(df)), mask] = 1
    return indicator

def build_frame(length, n):
    frame = np.flip(np.arange(0, length))
    frame = np.transpose(np.array([((frame + shift) % 3 == 0).astype(int) for shift in range(3)]))
    return np.repeat(frame[np.newaxis,:,:],n,axis=0)

def build_canonical_kozak_indicator(length, n):
    utr = np.zeros(length)
    for i in range(6):
        if i < utr.shape[0]:
            utr[i] = 1
    utr = np.flip(utr)
    return np.repeat(utr[np.newaxis,:],n,axis=0)[:,:,np.newaxis]

def encode_df(df, col='utr', libcol="library", n_libs=6,
              output_col="rl", variable_len=False, tis_col=None,
             cds_col=None, utr3_col=None):
    max_len = 0
    if variable_len:
        max_len = len(max(df[col], key=len))
    one_hot = np.stack(df[col].apply(encode_seq, max_len=max_len), axis = 0)
    indicator = encode_experiment(df, col=libcol, n_libs=n_libs)
    # Output column
    rl = None
    if output_col is not None:
        rl = np.array(df[output_col])
    tis_one_hot = None
    if tis_col is not None:
        tis_one_hot = np.stack([encode_seq(x) for x in df[tis_col]])
    frame = build_frame(one_hot.shape[1], one_hot.shape[0])
    kozak = build_canonical_kozak_indicator(one_hot.shape[1], one_hot.shape[0])
    # Encdoe whether it is endogenous
    seqtype = 1 - np.sum(indicator[:,:6], axis=1)
    # Encode CDS
    cds_seq = None
    if cds_col is not None:
        max_len = len(max(df[cds_col], key=len))
        cds_seq = np.stack(df[cds_col].apply(encode_seq, max_len=max_len), axis = 0)
    # Encode 3utr
    utr3_seq = None
    if utr3_col is not None:
        max_len = len(max(df[utr3_col], key=len))
        utr3_seq = np.stack(df[utr3_col].apply(encode_seq, max_len=max_len), axis = 0)
    return {"seq":one_hot, "library":indicator, "tis":tis_one_hot, "frame":frame, "kozak":kozak, "rl":rl,
           "seqtype": seqtype, "cds_seq":cds_seq, "utr3_seq":utr3_seq}

def extract_tis(df, downstream_nt=6, upstream_nt=2, utr_col="utr", cds_col="CDS_Sequence", new_col="tis", attach_to_utr=False): 
    # extract upstream
    df = df.copy()
    up_seq = df[cds_col].str[0:3+upstream_nt]
    if attach_to_utr:
        df[utr_col] = df[utr_col].str.cat(others=up_seq)
        return df
    #extract downstream
    down_seq = df[utr_col].str[-downstream_nt:]
    tis = down_seq.str.cat(others=up_seq)
    if new_col is None:
        return tis
    else:
        df[new_col] = tis
        return df
    
def build_tis_score_dict(path="../Data/TIS/tis_efficiencies_aug.tsv", seq_col="sequence", score_col="efficiency", replace_u=True):
    tis_df = pd.read_csv(path, sep='\t')
    if replace_u:
        tis_df[seq_col] = tis_df[seq_col].str.replace("U", "T")
    score_dict = {k:v for k,v in zip(tis_df[seq_col], tis_df[score_col])}
    return score_dict

""" TRAINING """
        
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
                         epochs=2,
                        extra_keys=[]):
    model = freeze_all_except_scaling(model)
    inputs = [np.concatenate([data["train"][library]["seq"] for library in libraries] + [data["val"]["human"]["seq"]]), 
              np.concatenate([data["train"][library]["library"] for library in libraries] + [data["val"]["human"]["library"]])]
    if len(extra_keys) > 0:
        inputs = inputs + [np.concatenate([data["train"][library][key] for library in libraries] + [data["val"]["human"][key]]) 
                                                                                                    for key in extra_keys]
    outputs = np.concatenate([data["train"][library]["rl"] for library in libraries] + [data["val"]["human"]["rl"]])
    model.fit(inputs, outputs, batch_size, epochs, verbose=2)
    return model

""" EVALUATION """

def rSquared(predictions, targets):
    ssr = np.sum(np.square(predictions - targets))
    ybar = np.average(targets)
    sst = np.sum(np.square(targets - ybar))
    return 1 - (ssr/sst)

def adjust_r2(r2, n, p):
    return 1-((1-r2)*(n-1)/(n-p-1))

def pearson_r(x, y, squared=True):
    pearson_r, p_val = stats.pearsonr(x, y)
    if squared:
        pearson_r = pearson_r ** 2
    return pearson_r, p_val

def spearman_r(x, y):
    return stats.spearmanr(x, y)


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

def check_uAUG_detection(trained_model, kozak=False, seq_length=200, samples=1000, add_tis=False, extra_keys=[],
                        alphabet=["A","C","T","G"], remove_stops=False):
    if kozak:
        out_df = pd.DataFrame({"idx": list(range(-seq_length, -8, 1)), "in_frame": [i % 3 == 0 for i in range(-seq_length, -8, 1)]})
    else:
        out_df = pd.DataFrame({"idx": list(range(-seq_length, -2, 1)), "in_frame": [i % 3 == 0 for i in range(-seq_length, -2, 1)]})
    predictions = []
    for i in range(samples):
         # Make a random sequence
        seq = ''.join(random.choices(alphabet, k=seq_length))
        # Remove existing atg
        atg_present = [m.start() for m in re.finditer('ATG', seq)]
        for idx in atg_present:
            seq = seq[:idx] + "C" + ''.join(random.choices(["C","T","G"], k=2)) + seq[idx+3:]
        if remove_stops:
            stop_present = [m.start() for m in re.finditer('TGA', seq)] + \
                           [m.start() for m in re.finditer('TAA', seq)] + \
                           [m.start() for m in re.finditer('TAG', seq)]
            for idx in stop_present:
                seq = seq[:idx] + "C" + ''.join(random.choices(["C","G","A"], k=2)) + seq[idx+3:]                  
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
        if add_tis:
            df["utr"] = df["utr"] + "ATGGG"
        data = encode_df(df, output_col=None)
        inputs = [data["seq"], data["library"]]
        for key in extra_keys:
            inputs.append(data[key])
        predictions.append(trained_model.predict(inputs))
    #get average prediction
    out_df["prediction"] = (sum(predictions)/samples).reshape(-1)
    return out_df

def uAUG_plot(uAUG_pred):
    in_frame = uAUG_pred.loc[uAUG_pred['in_frame']==True]
    out_frame = uAUG_pred.loc[uAUG_pred['in_frame']!=True]
    plt.plot( 'idx', 'prediction', data=in_frame, color='skyblue', linewidth=2)
    plt.plot( 'idx', 'prediction', data=out_frame, color='red', linewidth=2)