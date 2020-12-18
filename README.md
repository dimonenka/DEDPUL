# DEDPUL: Difference-of-Estimated-Densities-based Positive-Unlabeled Learning

The repo is based on (Ivanov 2019): https://arxiv.org/abs/1902.06965. Here you may find implementations of DEDPUL and several other PU learning methods: EN (Elkan and Noto 2008), non-negative Risk Estimation (Kiryo et al. 2017), KM (Ramaswamy, Scott, and Tewari 2016), TIcE (Bekker and Davis 2018).

The notebook sandbox_synth.ipynb is kind of a small tutorial. Try DEDPUL and other methods there.

The algorithm DEDPUL (also, EN) is implemented in algorithms.py. There are functions to obtain predictions of Non-Traditional Classifier, to estimate density ratio, to apply EM algorithm and Bayes rule - all the stuff used in DEDPUL. Secondary functions that are related to the training of neural networks are in NN_functions.py, nnPU is also there. Other secondary functions like data generation are in utils.py. KM in KMPE.py is retrieved from (Ramaswamy, Scott, and Tewari 2016): http://web.eecs.umich.edu/~cscott/code.html##kmpe. TIcE in tice.py is retrieved from (Bekker and Davis 2018) and adapted for python 3: https://dtai.cs.kuleuven.be/software/tice.

The notebooks experiments_UCI_MNIST.ipynb and experiments_synth.ipynb contain the code that was used to gather experimental data and to build figures in the paper.

Note: In the paper, $\alpha$ is prior probability of a random unlabeled instance to be a latent POSITIVE, and p_p(x) is a posterior probability of being POSITIVE. However, due to legacy reasons, in the code a prior probability of being NEGATIVE and corresponding posterior probability of being NEGATIVE are computed. To convert, just subtract the estimates from 1.

Preinstall numpy, pandas, sklearn, matplotlib, skipy, catboost, and pytorch, to run the code.

_____________________________________________________________________________________

The paper is published at ICMLA 2020.

Video presentation:
https://youtu.be/ypP94W1PVCY
