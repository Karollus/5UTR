# 5UTR

In the paper "Human 5′ UTR design and variant effect prediction from a massively parallel translation assay" (Sample et al), MPRA data is used to train a powerful deep model to predict ribsome load (a measure of translation efficiency) from the seuqnece of the 5 untranslated region. This model can be used to predict the effect of variants (mutations) on the ribosome load (and thus translation efficiency), which could be used to investigate the causes of rare genetic dieseases.

However, the published model, due to its use of a dense layer in the architecture, is inherently limited to specific sequence lengths. We remove this restriction and allow the model to produce predictions for arbitrary length sequences. To achieve this, we replace the dense layer with global max and average pooling operations on the output of the convolutional layers to provide an aggregated record of which sequence motifs were detected. In order to differentiate in which frame a motif is found (which plays a role for some motifs, such as upstream AUG), we perform these pooling operations on each frame seperately. Only then is the pooled motif data fed into a dense layer.

We show that such a model can provide similar performance as the published fixed-length on the same test set, while generalizing better to other contexts, such as longer MPRA sequences and endogenous data. We also show that the model has learnt to detect functionally relevant nucleotides and can correctly quantify the relative strength of uTIS motifs.

To replicate the main results, clone the repository. Next download the data from [Placeholder]. Place the data_dict.pkl in the Data directory. Then uncompress the All_Variants.tar.gz and place the contents in the All_Variants directory. Then you can run the Modelling.ipynb notebook to replicate the results.

In case you want to use the model to predict on your own, you can use the kipoi API. The kipoi_example.ipynb notebook in the Kipoi directory provides an example for how this can be done.

