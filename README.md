# DEDPUL: new Method for Positive-Unlabeled Classification and Mixture Proportions Estimation based on Density Estimation



The repo is based on (Ivanov 2019). Here you may find implementations of DEDPUL and several other PU learning methods: EN (Elkan and Noto 2008), non-negative Risk Estimation (Kiryo et al. 2017).

Dive in PU learning, try DEDPUL and other PU Learning methods in 'sandbox_synth.ipynb'.

Also, raw experimental data and the code that generated the data is provided in 'experiments_\*.ipynb'.

Note that in the original paper proportions and posteriors of Positive class are presented and estimated, while methods in this notebook by default estimate those for Negative class. To convert, just substract the estimates from 1.
