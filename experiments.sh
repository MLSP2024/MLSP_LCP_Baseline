#!/bin/bash

if ! ( \
	[ -d MLSP_Organisers ] || \
	cp -R ../MLSP_Organisers MLSP_Organisers \
	)
then
	echo "Error: MLSP_Organisers data missing. Please copy the repo to the MLSP_Organisers directory." >&2
	exit 1
fi


# Use SUBTLEX for ca en es pt:

echo "With SUBTLEX:"
echo

python baseline.py --train MLSP_Organisers/*/multilex_trial_*_lcp.tsv
python baseline.py MLSP_Organisers/Gold/[B-Z]*/multilex_test_*_lcp_labels.tsv


# Only use TUBELEX for ja:

echo
echo "Without SUBTLEX:"
echo

python baseline.py --train --sub ja --models models_no_subtlex MLSP_Organisers/*/multilex_trial_*_lcp.tsv
python baseline.py \
	--sub ja \
	-o output_no_subtlex \
	--models models_no_subtlex \
	MLSP_Organisers/Gold/[B-Z]*/multilex_test_*_lcp_labels.tsv

# Attic:

# python baseline.py --train MLSP_Organisers/*/multilex_trial_*_lcp.tsv

# python baseline.py --correlation MLSP_Organisers/*/multilex_trial_*_lcp.tsv
# python baseline.py --correlation MLSP_Organisers/Gold/*/multilex_test_*_lcp_labels.tsv

# python baseline.py --sub ca en es pt --correlation MLSP_Organisers/*/multilex_trial_*_lcp.tsv
# python baseline.py --sub ca en es pt --correlation MLSP_Organisers/Gold/*/multilex_test_*_lcp_labels.tsv

# python baseline.py MLSP_Organisers/*/multilex_trial_*_lcp.tsv
# python baseline.py MLSP_Organisers/Gold/*/multilex_test_*_lcp_labels.tsv

# python baseline.py --train MLSP_Organisers/*/multilex_trial_*_lcp.tsv
# python baseline.py --sub ca en es pt ja -- MLSP_Organisers/Gold/*/multilex_test_*_lcp_labels.tsv

# TRAIN ON TEST python baseline.py --sub ca en es pt ja --train MLSP_Organisers/Gold/*/multilex_test_*_lcp_labels.tsv
