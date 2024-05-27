# MLSP_LCP_Baseline

An LCP baseline for the Multilingual Lexical Simplification Pipeline 2024 Shared Task modelled as a linear regression on log-frequency. The frequency baseline is trained using log-frequency (minimum value if the target consists of multiple tokens) on the trial set for each language. We use frequencies provided by the [`wordfreq` package](https://pypi.org/project/wordfreq/) when possible. Additionally, since the package uses an incompatible tokenization for Japanese and does not provide any data for Sinahala, we use [TUBELEX-JA](https://github.com/adno/tubelex) for Japanese, and the [word frequency list for Sinhala](https://github.com/nlpcuom/Word-Frequency-List-for-Sinhala).


## Reproducing the baseline
 
Note that the trained [models](models) and [output](output) of the baseline are already included in the repository. You can reproduce them by following the steps below.

1. Install the Git submodule for [MLSP_Data](https://github.com/MLSP2024/MLSP_Data), [Word-Frequency-List-for-Sinhala](https://github.com/nlpcuom/Word-Frequency-List-for-Sinhala) and [tubelex](https://github.com/adno/tubelex):

    ```git submodule init && git submodule update```
    
2. Install the [requirements](requirements.txt):
	
	```python -m pip install -r requirements.txt```
    
3. Run the baseline (both training and prediction):

    ```bash experiments.sh```


## Links

- [MLSP shared task web site](https://sites.google.com/view/mlsp-sharedtask-2024)
- [shared task data repository](https://github.com/MLSP2024/MLSP_Data)
- [cleaned gold test data](https://huggingface.co/datasets/MLSP2024/MLSP2024) on Hugging Face
- [LLM-based lexical simplification baseline](https://github.com/MLSP2024/MLSP_LS_LLM_Baseline)
