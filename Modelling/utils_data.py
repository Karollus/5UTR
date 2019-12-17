import re
import random
random.seed(1337)
import os
import pickle
import itertools
import functools
import operator

import numpy as np
np.random.seed(1337)
import pandas as pd
import scipy.stats as stats
import ahocorasick

import concise

from keras.utils import Sequence

class EncodingFunction:

    def __init__(self, name):
        self.name = name
    
    def __call__(self, df):
        pass

class PrecomputeFunction(EncodingFunction):
    
    def __init__(self, new_col, dims, method="stack"):
        self.new_col = new_col
        self.dims = dims
        self.method = method
        super().__init__(new_col)
        
        
### Precomputation functions ###

# Computes the sequence length
class SeqLenExtractor(PrecomputeFunction):
    
    def __init__(self, seq_col, new_col):
        self.seq_col = seq_col
        super().__init__(new_col, dims=(1,))
        
    def __call__(self, df):
        return df[self.seq_col].str.len()

# Counts along the sequence (e.g. gives a sequence 1,2,3,4,5,...)
class SeqCounterExtractor(PrecomputeFunction):
    
    def __init__(self, seq_col, new_col):
        self.seq_col = seq_col
        super().__init__(new_col, dims=(None,), method="pad_stack")
        
    def count(self, seq):
        length = len(seq)
        return np.flip(np.arange(1, length+1))/length
    
    def __call__(self, df):
        return df[self.seq_col].apply(self.count)

# Precomputes the secondary structure
class SecondaryStructureComputer(PrecomputeFunction):
    
    def __init__(self, seq_col, new_col):
        self.seq_col = seq_col
        super().__init__(new_col, dims=(None,), method="stack")
    
    def __call__(self, df):
        return concise.encodeRNAStructure(seq_vec, maxlen=None, seq_align='end', W=240, L=160, U=1, tmpdir='/tmp/RNAplfold/')
    

# Finds a pattern in the sequece and computes a mask
class RegexPosExtractor(PrecomputeFunction):
    
    def __init__(self, seq_col, new_col, pattern="A[TU]G", offset=1):
        self.seq_col = seq_col
        self.pattern = pattern
        super().__init__(new_col, dims=(1,), method="pad_stack")
        
    def find(self, seq):
        indices = [m.start()+offset for m in re.finditer(self.pattern, seq)]
        indicator = np.zeros((len(seq)))
        indicator[indices] = 1
        return indicator
        
    def __call__(self, df):
        return df[self.seq_col].apply(self.find)

# Records which frames are "stopped", i.e. lead to a stop codon
class StoppedFramesExtractor(PrecomputeFunction):
    
    def __init__(self, seq_col, new_col):
        self.seq_col = seq_col
        self.pattern = "[TU][GA]A|[TU]A[GA]"
        super().__init__(new_col, dims=(1,), method="pad_stack")
        
    def find(self, seq):
        indices = [m.start()+1 for m in re.finditer(self.pattern, seq)]
        within_stop = set()
        for idx in indices:
            within_stop.add(idx)
            while idx >= 3:
                idx = idx - 3
                within_stop.add(idx)
        indicator = np.zeros((len(seq)))
        indicator[list(within_stop)] = 1
        return indicator
        
    def __call__(self, df):
        return df[self.seq_col].apply(self.find)
    
# Extracts kmers
class KmerExtractor(PrecomputeFunction):
    
    def __init__(self, seq_col, new_col, k, jump=False, divide_counts=True):
        self.k = k
        self.seq_col = seq_col
        kmers = [''.join(i) for i in itertools.product(["A","C","T","G"], repeat = self.k)]
        self.n = len(kmers)
        self.kmer_dict = {kmers[k]:k for k in range(self.n)}
        self.jump = jump
        self.divide_counts = divide_counts
        super().__init__(new_col, dims=(self.n,))
    
    def extract(self, seq):
        i = 0
        arr = np.zeros(self.n)
        while i < len(seq) - (self.k - 1):
            arr[self.kmer_dict[seq[i:i+self.k]]] = arr[self.kmer_dict[seq[i:i+self.k]]] + 1
            if self.jump:
                i = i + self.k
            else:
                i += 1
        if self.divide_counts:
            arr/np.sum(arr)
        return arr

    def __call__(self, df):
        return df[self.seq_col].apply(self.extract)

# Extracts kmers at specific positions (e.g. start or end of sequence)
class KmerAtPosExtractor(PrecomputeFunction):

    def __init__(self, seq_col, new_col, positions):
        self.seq_col = seq_col
        i = 0
        self.pos_kmer_dict = {}
        for start, stop in positions:
            k = np.abs(stop - start)
            kmers = [''.join(i) for i in itertools.product(["A","C","T","G"], repeat = k)]
            kmer_dict = {kmers[k]:k+i for k in range(len(kmers))}
            i += len(kmers)
            self.pos_kmer_dict[(start, stop)] = kmer_dict
        self.n = i
        self.pos = positions
        super().__init__(new_col, dims=(self.n,))
    
    def extract(self, seq):
        arr = np.zeros(self.n)
        for interval in self.pos:
            kmer_dict = self.pos_kmer_dict[interval]
            start, stop = interval
            arr[kmer_dict[seq[start:stop]]] = arr[kmer_dict[seq[start:stop]]] + 1
        return arr

    def __call__(self, df):
        return df[self.seq_col].apply(self.extract)

# Extracts GC content
class GCContentExtractor(PrecomputeFunction):
    
    def __init__(self, seq_col, new_col):
        self.seq_col = seq_col
        super().__init__(new_col, dims=(1,))
        
    def __call__(self, df):
        return (df[self.seq_col].str.count("G") + 
                df[self.seq_col].str.count("C"))/df[self.seq_col].str.len()
    
# Counts specific motifs (e.g. PolyA sites)
class MotifExtractor(PrecomputeFunction):
     
    def __init__(self, seq_col, new_col, motifs):
        self.seq_col = seq_col
        self.n = len(motifs)
        self.motifs = motifs
        self.ahoAutomat = ahocorasick.Automaton()
        for idx, key in enumerate(motifs):
            self.ahoAutomat.add_word(key, idx)
        self.ahoAutomat.make_automaton()
        super().__init__(new_col, dims=(self.n,))
    
    def extract_motives(self, seq):
        # Use ahocorasick
        arr = np.zeros(self.n)
        for end_index, idx in self.ahoAutomat.iter(seq):
            arr[idx] += 1
        return arr
            
    def __call__(self, df):
        return df[self.seq_col].apply(self.extract_motives)
    

class NodererScore(PrecomputeFunction):
    
    def __init__(self, noderer_df_aug, noderer_df_nonaug, new_col="noderer",
                utr_col="utr", cds_col="cds",
                seq_col="sequence", score_col="efficiency"):
        self.utr_col, self.cds_col = utr_col, cds_col
        # replace U with T in Noderer dataframe
        noderer_df_aug[seq_col] = noderer_df_aug[seq_col].str.replace("U", "T")
        noderer_df_nonaug[seq_col] = noderer_df_nonaug[seq_col].str.replace("U", "T")
        self.avg_score = noderer_df_nonaug[score_col].median()
        # build dictionary
        self.score_dict_aug = {k:v for k,v in zip(noderer_df_aug[seq_col], 
                                                  noderer_df_aug[score_col])}
        self.score_dict_nonaug = {k:v for k,v in zip(noderer_df_nonaug[seq_col], 
                                              noderer_df_nonaug[score_col])}
        super().__init__(new_col, dims=(1,))
    
    def score(self, tis):
        score = self.score_dict_aug.get(tis)
        if score is None:
            score = self.score_dict_nonaug.get(tis)
            if score is None:
                score = self.avg_score
        return score
    
    def __call__(self, df):
        tis = df[self.utr_col].str[-6:]
        tis = tis.str.cat(df[self.cds_col].str[:5])
        return tis.apply(self.score)

class PrecomputeEmbeddings(PrecomputeFunction):
    
    def __init__(self, new_col, model, layer_name, input_layer_names,
                 generator_encoding_functions, 
                 dim_select=None,
                 node=0):
        target_obj = model.get_layer(layer_name).get_output_at(node)
        target = [target_obj]
        self.check_fn = K.function([model.get_layer(x).input for x in input_layer_names], target)
        self.generator_encoding_functions = generator_encoding_functions.copy()
        if dim_select is None:
            self.dim_select = np.ones(int(target_obj.shape[1]))
        else:
            self.dim_select = np.array(dim_select)
        super().__init__(new_col, dims=(np.sum(self.dim_select),), method="concat")
    
    def __call__(self, df):
        generator = DataSequence(df, encoding_functions=self.generator_encoding_functions, 
                                 shuffle=False)
        l = [self.check_fn(x)[0][:,self.dim_select] for x in generator]
        return functools.reduce(operator.concat, [np.vsplit(x, x.shape[0]) for x in l])
    
### Encoding functions ###

class DataFrameExtractor(EncodingFunction):
    
    def __init__(self, col, method="stack"):
        self.col = col
        super().__init__(col)
        self.method = method
        
    def __call__(self, df):
        if self.method == "direct":
            return np.array(df[self.col])
        elif self.method == "stack":
            return np.stack(df[self.col], axis = 0)
        elif self.method == "pad_stack":
            max_len = len(max(df[self.col], key=len))
            col_list = []
            for vector in df[self.col]:
                if len(vector) < max_len:
                    vector = np.concatenate([np.zeros(max_len - len(vector)), vector])
                col_list.append(vector)
            return np.stack(col_list, axis = 0)
        else:
            return np.concatenate(list(df[self.col]), axis = 0)
        
class OneHotEncoder(EncodingFunction):

    def __init__(self, col, min_len=None):
        self.col = col
        self.min_len = min_len
        self.nuc_dict = {'a':[1.0,0.0,0.0,0.0],'c':[0.0,1.0,0.0,0.0],'g':[0.0,0.0,1.0,0.0], 
                         'u':[0.0,0.0,0.0,1.0], 't':[0.0,0.0,0.0,1.0], 
                         'n':[0.0,0.0,0.0,0.0], 'x':[1/4,1/4,1/4,1/4]}
        super().__init__(col)
        

        
    def encode_seq(self, seq, max_len=0):
        length = len(seq)
        if max_len > 0 and self.min_len is None:
            padding_needed = max_len - length
            seq = "N"*padding_needed + seq
        if self.min_len is not None:
            if len(seq) < self.min_len:
                seq = "N"*(self.min_len - len(seq)) + seq
            if len(seq) > self.min_len:
                seq = seq[(len(seq) - self.min_len):]
        seq = seq.lower()
        one_hot = np.array([self.nuc_dict[x] for x in seq]) # get stacked on top of each other
        return one_hot
    
    def __call__(self, df):
        max_len = len(max(df[self.col], key=len))
        return np.stack([self.encode_seq(seq, max_len) 
                         for seq in df[self.col]], axis = 0)
    
class FrameEncoder(EncodingFunction):
    
    def __init__(self, col, min_len=None):
        self.col = col
        super().__init__(col)
    
    def build_frame(self, length, n):
        frame = np.flip(np.arange(0, length))
        frame = np.transpose(np.array([((frame + shift) % 3 == 0).astype(int) for shift in range(3)]))
        return np.repeat(frame[np.newaxis,:,:],n,axis=0)
    
    def __call__(self, df):
        max_len = len(max(df[self.col], key=len))
        return self.build_frame(max_len, len(df))

class LibraryEncoder(EncodingFunction):
    
    def __init__(self, col, library_dict, default=0):
        self.col = col
        self.dict = library_dict
        self.default = default
        super().__init__(col)
    
    def __call__(self, df):
        if self.col not in df.keys():
            mask = np.array([self.default]*len(df))
        else:
            mask = np.array([self.dict.get(x, self.default) for x in df[self.col]])
        indicator = np.zeros((len(df),len(self.dict)))
        indicator[np.arange(len(df)), mask] = 1
        return indicator
    
### Dataloader ###

class DataSequence(Sequence):
    
    def __init__(self, df, precomputations=[], 
                 encoding_functions=[],
                 input_order=None,
                 output_encoding_fn=None,
                 batch_size=128, shuffle=True):
        self.df = df.copy()
        self.df = df.reset_index(drop=True)
        self.encoding_functions = encoding_functions.copy()
        self.output_encoding_fn = output_encoding_fn
        self.indices = np.arange(len(self.df))
        self.batch_size = batch_size
        for fn in precomputations:
            print("Doing precomputation: " + fn.new_col)
            self.encoding_functions.append(DataFrameExtractor(fn.new_col, fn.method))
            self.df[fn.new_col] = fn(self.df)
        if input_order is not None:
            fn_dict = {fn.name:fn for fn in self.encoding_functions}
            self.encoding_functions = [fn_dict[name] for name in input_order]
        self.shuffle = shuffle
        super().__init__()

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]]
        # Feed input
        inputs = [fn(batch_df) for fn in self.encoding_functions]
        if self.output_encoding_fn is None:
            return inputs
        else:
            return (inputs, self.output_encoding_fn(batch_df))
            
    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)
