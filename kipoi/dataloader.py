import os
import tempfile
import subprocess

import pandas as pd
import numpy as np

from kipoi.metadata import GenomicRanges
from kipoi.data import Dataset

from kipoiseq.extractors import FastaStringExtractor
from kipoiseq.extractors import SingleSeqVCFSeqExtractor
from kipoiseq.dataloaders.sequence import BedDataset
from kipoiseq.transforms.functional import resize_interval

import pybedtools
from pybedtools import BedTool, Interval

class StringSeqIntervalDlWithStrandAndVariants(Dataset):
    """Dataloader for a combination of fasta, bgzip compressed vcf and bed3+ input files, where a specific user-specified 
       column (>3) of the bed denotes the strand. 
       All columns of the bed3+ except the first three and the strand column are ignored. 
       The bed and vcf must both be sorted (by position).
       If a tabix index for the vcf is not present (must lie in the same directory and have the same name + .tbi), it will
       be generated (and deleted afterwards).
       The dataloader finds all intervals in the bed3 which contain at least one variant in the vcf, then, for these
       intervals, it extracts the reference sequence from the fasta file, injects the applicable variants and reverse 
       complements according to the strand information.
       Returns the reference sequence and variant sequence as np.array([reference_sequence, variant_sequence]). 
       Region metadata is additionally provided """
    def __init__(self,
                 intervals_file,
                 fasta_file,
                 vcf_file,
                 strand_column=6,
                 num_chr_fasta=True
                ):

        self.num_chr_fasta = num_chr_fasta
        self.intervals_file = intervals_file
        self.fasta_file = fasta_file
        self.vcf_file = vcf_file
        
        if strand_column <= 3:
            raise ValueError("Strand column is given as {} but must be larger than 3. NB: 1-based! 
                             ".format(self.strand_column))
        self.strand_column = strand_column - 4
                
        self.auto_resize_len = None
        self.force_upper = True
        
        self.tmp_files = []
        
        # Presort, if necessary
        # if not bed_sorted:
            # tmp = tempfile.mkstemp(suffix=".bed")[1]
            # pybedtools.BedTool(self.intervals_file).sort(output=tmp)
            # self.intervals_file = tmp
            # self.tmp_files.add(tmp)
        # if not vcf_sorted:
            # tmp = tempfile.mkstemp(suffix=".vcf.gz")[1]
            # pybedtools.BedTool(self.vcf_file).sort(output=tmp)
            # self.vcf_file = tmp
            # self.tmp_files.add(tmp)
       
        # "Parse" bed file
        self.bed = BedDataset(self.intervals_file,
                              num_chr=self.num_chr_fasta,
                              bed_columns=3,
                              label_dtype=str,
                              ignore_targets=False)
        
        # Intersect bed and vcf using bedtools
        # interval_generator = (self.bed[idx][0] for idx in range(len(self.bed)))
        bed_tool = pybedtools.BedTool(self.intervals_file)
        intersect_counts = list(bed_tool.intersect(self.vcf_file, c=True, sorted=True)) # c flag: for each bed interval,  counts number of vcf entries it overlaps
        intersect_counts = np.array([intersect_counts[i].count for i in range(len(intersect_counts))])
        
        # Retain only those intervals that intersect a variant
        self.bed.df = self.bed.df[intersect_counts > 0] # yes, this breaks information hiding and is bad. Better sol: 
                                                        # adding a subset method to Bed dataset 
        
        # if tabix index is not present, create it
        tabix_path = self.vcf_file + ".tbi"
        if not os.path.isfile(tabix_path):
            try:
                subprocess.check_output(['tabix','-p','vcf',self.vcf_file])
                self.tmp_files.append(tabix_path)
            except: 
                raise ValueError("Failed to create tabix index. Possible reasons: vcf corrupt; vcf not bgzipped; \
                                 vcf not sorted by position")
        
        self.fasta_extractor = None
        self.vcf_extractor = None
        

    def __len__(self):
        return len(self.bed)

    def __getitem__(self, idx):
        if self.fasta_extractor is None:
            self.fasta_extractor = FastaStringExtractor(self.fasta_file, use_strand=True,
                                                         force_upper=self.force_upper)
        if self.vcf_extractor is None:
            self.vcf_extractor = SingleSeqVCFSeqExtractor(self.fasta_file, self.vcf_file)
        
        interval, labels = self.bed[idx]
        strand = labels[self.strand_column]
        
        # We copy because editing the interval directly causes a seg fault. Why? I have no clue.
        interval = pybedtools.Interval(interval.chrom, interval.start, interval.stop, strand=strand)
        
        if self.auto_resize_len:
            interval = resize_interval(interval, self.auto_resize_len, anchor='center')

        # We get the reference sequence
        ref_seq = self.fasta_extractor.extract(interval)
        
        # We get the sequence with variants
        var_seq = self.vcf_extractor.extract(interval, anchor= 0, fixed_len=False)
        
        return {
            "inputs": np.array([ref_seq, var_seq]),
            "metadata": {
                "ranges": GenomicRanges(interval.chrom, interval.start, interval.stop, str(idx), strand=interval.strand)
            }
        }
    
    def __del__(self):
        for file in self.tmp_files:
            os.remove(file)