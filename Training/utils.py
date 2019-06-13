
"""  """

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

def encode_experiment(experiment, n):
    indicator = [0,0,0,0,0]
    indicator[experiment] = 1
    return np.repeat(np.array(indicator)[np.newaxis,:], n, axis=0)

def encode_df(df, col='seq', variable_len=False):
    out_dict = {}
    max_len = 0
    if variable_len:
        max_len = len(max(df[col], key=len))
    one_hot = np.stack(df[col].apply(encode_seq, max_len=max_len), axis = 0)
    indicator = build_indicator(experiment, one_hot.shape[0])
    out_dict["indicator"] = indicator
    return one_hot, indicator
