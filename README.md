# DEDPUL: Difference-of-Estimated-Densities-based Positive-Unlabeled Learning



The repo is based on (Ivanov 2019): https://arxiv.org/abs/1902.06965. Here you may find implementations of DEDPUL and several other PU learning methods: EN (Elkan and Noto 2008), non-negative Risk Estimation (Kiryo et al. 2017).

The notebook sandbox_synth.ipynb is kind of a small tutorial. Try DEDPUL and other methods there.

The algorithm DEDPUL (also, EN) is implemented in algorithms.py. There are functions to obtain predictions of Non-Traditional Classifier, to estimate density ratio, to apply EM algorithm and Bayes rule - all the stuff used in DEDPUL. Secondary functions that are connected to training of neural networks are in NN_functions.py. Other secondary functions like data generation are in utils.py. KMPE.py is retrieved from (Ramaswamy, Scott, and Tewari 2016): http://web.eecs.umich.edu/~cscott/code.html##kmpe 

The notebooks experiments_UCI_MNIST.ipynb and experiments_synth.ipynb contain the code that was used to gather experimental data and to build figures 2,3,4,5. The corresponding data is in ‘experimental_data’ folder. The notebook experiment_processing.ipynb is about verifying statistical significance of results — whether algorithms differ significantly — using wilcoxon signed-rank test. The corresponding data is in ‘raw_data’ folder.

Note: In the paper, $\alpha$ is prior probability of a random unlabeled instance to be a latent POSITIVE, and f(p | x) is a posterior probability of being POSITIVE. However, due to legacy reasons, in the code a prior probability of being NEGATIVE and corresponding posterior probability of being NEGATIVE are computed. To convert, just subtract estimates from 1.

One mainly needs python 3 with numpy, pandas, sklearn, matplotlib, skipy, and pytorch, to run the code. One may comment catboost and keras parts.