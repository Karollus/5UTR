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

def encode_df(df, col='utr', libcol="library", output_col="rl", variable_len=False, tis_col=None):
    max_len = 0
    if variable_len:
        max_len = len(max(df[col], key=len))
    one_hot = np.stack(df[col].apply(encode_seq, max_len=max_len), axis = 0)
    indicator = encode_experiment(df, col=libcol)
    rl = None
    if output_col is not None:
        rl = np.array(df[output_col])
    tis_one_hot = None
    if tis_col is not None:
        tis_one_hot = np.stack([encode_seq(x) for x in df[tis_col]])
    frame = build_frame(one_hot.shape[1], one_hot.shape[0])
    kozak = build_canonical_kozak_indicator(one_hot.shape[1], one_hot.shape[0])
    return {"seq":one_hot, "library":indicator, "tis":tis_one_hot, "frame":frame, "kozak":kozak, "rl":rl}

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

# Generator class for input data (useful to batch small sequences to prevent insane padding)
class DataSequence(Sequence):
    
    def __init__(self, df, col="utr", libcol="library", 
                 output_col="rl", 
                 tis_col="tis", 
                 extra_keys=[], batch_size=128, shuffle=True):
        self.df = df
        self.col, self.libcol, self.output_col, self.tis_col = col, libcol, output_col, tis_col
        self.extra_keys = extra_keys
        self.indices = np.arange(len(self.df))
        self.batch_size = batch_size
        self.shuffle = shuffle
        super().__init__()

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]]
        # Prepare input data
        encoded_data = encode_df(batch_df, col=self.col, libcol=self.libcol, 
                                 output_col=self.output_col, variable_len=True,
                                 tis_col=self.tis_col)
        # Feed input
        inputs = [encoded_data["seq"], encoded_data["library"]]
        for key in self.extra_keys:
            inputs.append(encoded_data[key])
        if self.output_col is None:
            return inputs
        else:
            return (inputs, encoded_data["rl"])
            
    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    
def build_tis_score_dict(path="../Data/TIS/tis_efficiencies_aug.tsv", seq_col="sequence", score_col="efficiency", replace_u=True):
    tis_df = pd.read_csv(path, sep='\t')
    if replace_u:
        tis_df[seq_col] = tis_df[seq_col].str.replace("U", "T")
    score_dict = {k:v for k,v in zip(tis_df[seq_col], tis_df[score_col])}
    return score_dict

""" TRAINING """

def train(model, data, libraries, batch_size=128, epochs=3, use_val=True, early_stop=True, patience=3, file="best_model.h5",
         extra_keys=[]):
    inputs = [np.concatenate([data["train"][library]["seq"] for library in libraries]), 
              np.concatenate([data["train"][library]["library"] for library in libraries])]
    if len(extra_keys) > 0:
        inputs = inputs + [np.concatenate([data["train"][library][key] for library in libraries]) for key in extra_keys]
    outputs = np.concatenate([data["train"][library]["rl"] for library in libraries])
    if use_val:
        val_libs = set(data["val"].keys()) & set(libraries)
        val_inputs = [np.concatenate([data["val"][library]["seq"] for library in val_libs]), 
                  np.concatenate([data["val"][library]["library"] for library in val_libs])]
        if len(extra_keys) > 0:
            val_inputs = val_inputs + [np.concatenate([data["val"][library][key] for library in val_libs]) for key in extra_keys]
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

def print_corrs(x ,y):
    pearson = pearson_r(x, y, squared=False)
    spearman = spearman_r(x, y)
    print("Pearson: {:.3f}, p-val: {:.3f}, squared: {:.3f}, Spearman: {:.3f}, p-val: {:.3f}"
          .format(pearson[0], pearson[1], pearson[0] ** 2, spearman[0], spearman[1]))

def evaluate(model, data, libraries, do_test=False,
            extra_keys=[]):
    set_type = "test" if do_test else "val"
    preds = []
    for library in libraries:
        inputs = [data[set_type][library]["seq"], data[set_type][library]["library"]]
        if len(extra_keys) > 0:
            inputs = inputs + [np.concatenate([data[set_type][library][key] for library in libraries]) for key in extra_keys]
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

def eval_snv(model, data, snv_df, extra_keys=[]):
    preds = []
    for library in ["snv", "wt"]:
        inputs = [data[library]["seq"], data[library]["library"]]
        if len(extra_keys) > 0:
            inputs = inputs + [data[library][key] for key in extra_keys]
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

def eval_ptr(model, data, ptr_df, extra_keys=[]):
    inputs = [data["seq"], data["library"]]
    if len(extra_keys) > 0:
        inputs = inputs + [data[key] for key in extra_keys]
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