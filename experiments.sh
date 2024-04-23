#!/bin/bash

# When using labelled data (to predict and print metrics at the same time:)
# if ! ( \
# 	[ -d MLSP_Organisers ] || \
# 	cp -R ../MLSP_Organisers MLSP_Organisers \
# 	)
# then
# 	echo "Error: MLSP_Organisers data missing. Please copy the repo to the MLSP_Organisers directory." >&2
# 	exit 1
# fi


python baseline.py --train MLSP_Data/Data/Trial/[B-Z]*/multilex_trial_*_lcp.tsv
python baseline.py MLSP_Data/Data/Test/[B-Z]*/multilex_test_*_lcp_unlabelled.tsv	

# Attic:

# python baseline.py --train MLSP_Organisers/*/multilex_trial_*_lcp.tsv

# python baseline.py --correlation MLSP_Organisers/*/multilex_trial_*_lcp.tsv
# python baseline.py --correlation MLSP_Organisers/Gold/*/multilex_test_*_lcp_labels.tsv

# python baseline.py --correlation MLSP_Organisers/*/multilex_trial_*_lcp.tsv
# python baseline.py --correlation MLSP_Organisers/Gold/*/multilex_test_*_lcp_labels.tsv

# python baseline.py MLSP_Organisers/*/multilex_trial_*_lcp.tsv
# python baseline.py MLSP_Organisers/Gold/*/multilex_test_*_lcp_labels.tsv

# python baseline.py --train MLSP_Organisers/*/multilex_trial_*_lcp.tsv
# python baseline.py ja -- MLSP_Organisers/Gold/*/multilex_test_*_lcp_labels.tsv

# TRAIN ON TEST python baseline.py ja --train MLSP_Organisers/Gold/*/multilex_test_*_lcp_labels.tsv
