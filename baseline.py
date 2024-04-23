import argparse
import zipfile
from collections import defaultdict, Counter
from typing import Optional
import sys
import os
import io
import re
from itertools import zip_longest
import numpy as np
from wordfreq import word_frequency, tokenize as wf_tokenize
import fugashi

from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    mean_squared_error as mse_score, mean_absolute_error as mae_score, r2_score
    )
from joblib import dump, load
from frequency_data import FrequencyData

OPT_WAKATI = '-O wakati'

RE_JA_WORD = re.compile(
    r'^([^\d]*(?!\d)[\w][^\d]*)$'
    )   # All word-froming, at least one non-digit

FREQ_EPS = 1e-9

LANG2FULL_NAME: dict[str, str] = {
    'en': 'English',
    'ca': 'Catalan',
    'fil': 'Filipino',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'ja': 'Japanese',
    'pt': 'Portuguese',
    'si': 'Sinhala',
    'es': 'Spanish'
    }


def pearson_r(x: np.ndarray, y: np.ndarray):
    return np.corrcoef(x, y)[0][1]


def fugashi_tagger(
    dicdir: Optional[str],
    option: str = OPT_WAKATI
    ) -> fugashi.GenericTagger:
    if dicdir is None:
        return fugashi.Tagger(option)  # -d/-r supplied automatically
    # GenericTagger: we do not supply wrapper (not needed for -O wakati)
    mecabrc = os.path.join(dicdir, 'mecabrc')
    return fugashi.GenericTagger(f'{option} -d {dicdir} -r {mecabrc}')


def tagger_from_args(
    args: argparse.Namespace,
    option: str = OPT_WAKATI
    ) -> fugashi.GenericTagger:

    # We always specify dicdir EXPLICITLY
    if args.dicdir is not None:
        dicdir = args.dicdir
    else:
        if args.dictionary == 'unidic':
            import unidic  # type: ignore
            dicdir = unidic.DICDIR
        else:
            assert args.dictionary is None or args.dictionary == 'unidic-lite'
            import unidic_lite  # type: ignore
            dicdir = unidic_lite.DICDIR
    return fugashi_tagger(dicdir, option)


def add_tagger_arg_group(
    parser: argparse.ArgumentParser,
    title: Optional[str] = None
    ):
    titled_group = parser.add_argument_group(title=title)
    dic_group = titled_group.add_mutually_exclusive_group()
    dic_group.add_argument(
        '--dicdir', type=str, default=None,
        help='Dictionary directory for fugashi/MeCab.'
        )
    dic_group.add_argument(
        '--dictionary', '-D', choices=('unidic', 'unidic-lite'), default=None,
        help=(
            'Dictionary (installed as a Python package) for fugashi/MeCab.'
            'Default: unidic-lite.'
            )
        )


def get_sub_counts_total(language: str) -> tuple[dict[str, int], int]:
    fd = FrequencyData.from_subtitles(language)
    return (fd.f, fd.f_total)


def wfl_si_counts_total(
    path: str = 'Word-Frequency-List-for-Sinhala/word_frequency_list_2M.zip'
    ) -> dict[str, int]:

    archive = zipfile.ZipFile(path, 'r')
    name = archive.namelist()[0]
    total = 0
    counts = {}
    lines = io.TextIOWrapper(archive.open(name, 'r'), encoding='utf-8')
    _ = next(lines)  # Skip header
    for line in lines:
        w, _, n = line.rstrip('\n').split(' ')
        n = int(n)
        counts[w] = n
        total += n
    return (counts, total)


def tecla_ca_counts_total(
    path: str = 'data/wordCntTeclaSpacy.tsv'
    ) -> dict[str, int]:
    total = 0
    counts = {}
    lines = open(path)
    _ = next(lines)  # Skip header
    for line in lines:
        w, n = line.rstrip('\n').rsplit('\t', 1)  # handle tokens, which include tabs
        n = int(n)
        counts[w] = n
        total += n
    return (counts, total)


def tgl_counts_total(
    path: str = 'data/tgl/tgl_community_2017/tgl_community_2017-words.txt'
    # 'data/tgl/tgl_wikipedia_2021_100K/tgl_wikipedia_2021_100K-words.txt'
    ) -> dict[str, int]:
    total = 0
    counts = Counter()
    for line in open(path):
        _, w, *_, n = line.rstrip('\n').split('\t')
        n = int(n)
        counts[w.lower()] += n  # not lowercased in files
        total += n
    return (counts, total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--train', action='store_true')
    action.add_argument('--correlation', action='store_true')
    parser.add_argument(
        '--force', '-f', action='store_true', help='Force train (even with test data).'
        )
    parser.add_argument(
        '--metrics', '-m', action='store_true',
        help='Print metrics when doing inference.'
        )
    parser.add_argument(
        '--skip-errors', '-s', action='store_true',
        help='Warn on parsing errors and skip.'
        )
    parser.add_argument(
        'input_files', nargs='*',
        help='Input files in LCP format (default all train or test).'
        )
    parser.add_argument('--output-files', '-o', nargs='*', default=[], help=(
        'Output files in LCP format when doing inference. '
        'If single name is given for multiple input files, '
        'it is understood as a directory name for multiple files. '
        'Default: output (output.tsv).'
        ))
    parser.add_argument(
        '--subtitles', nargs='*',
        default=['ja'], help=(
            'Use subtitles for these language codes. Default: ja.'
            )
        )
    parser.add_argument(
        '--tecla', action='store_true', help='Use TeCla for Catalan.'
        )
    tgl = parser.add_mutually_exclusive_group()
    tgl.add_argument('--tgl-community', action='store_true', help=(
        'Use tgl_community_2017 for Filipino.'
        ))
    tgl.add_argument('--tgl-wikipedia', action='store_true', help=(
        'Use tgl_wikipedia_2021 for Filipino.'
        ))
    parser.add_argument('--models', default='models', help='Model directory.')
    parser.add_argument('--verbose', '-v', action='store_true')
    add_tagger_arg_group(parser)

    return parser.parse_args()


def main(args: argparse.Namespace):
    input_files = args.input_files
    output_files = args.output_files
    model_dir = args.models
    train = args.train
    correlation = args.correlation
    read_gold = train or correlation
    tecla = args.tecla

    if not input_files:
        input_files = [
            'MLSP_Data/Data/Trial/All/multilex_trial_all_lcp.tsv' if read_gold else
            'MLSP_Data/Data/Test/All/multilex_test_all_combined_lcp_unlabelled.tsv'
            ]
    if train and (not args.force) and any('test' in path for path in input_files):
        raise Exception(
            'Supplied --train without --force, but input filenames contain "test".'
            )

    if not output_files and not train:
        output_files = ['output.tsv'] if (len(input_files) == 1) else ['output']

    if train:
        if output_files:
            raise Exception('Supplied --train with --output-files.')
        os.makedirs(model_dir, exist_ok=True)
    elif len(output_files) == 1 and len(input_files) > 1:
        output_dir = output_files[0]
        os.makedirs(output_dir, exist_ok=True)
        output_files = [
            os.path.join(
                output_dir,
                re.sub(r'_unlabelled|_labels', '', os.path.basename(path))
                )
            for path in input_files
            ]

    assert (
        train or
        (len(input_files) == len(output_files))
        ), (train, len(input_files), len(output_files))

    ja_tagger = tagger_from_args(args)

    def ja_tokenize(s: str) -> list[str]:
        return [w for w in ja_tagger.parse(s).split(' ') if RE_JA_WORD.match(w)]

    def space_tokenize(s: str) -> list[str]:
        return s.split(' ')

    lang2counts_total = {}
    for lang in args.subtitles:
        if lang == 'ca' and tecla:
            print('Warning: --tecla overrides --subtiles ca. Using TeCla for Catalan.',
                  file=sys.stderr)
            continue
        lang2counts_total[lang] = get_sub_counts_total(lang)

    si_counts, si_total = wfl_si_counts_total()

    def si_frequency(s: str):
        return si_counts.get(s, 0) / si_total

    if tecla:
        ca_counts, ca_total = tecla_ca_counts_total()

        def ca_frequency(s: str):
            return ca_counts.get(s.lower(), 0) / ca_total

    tgl = False
    if args.tgl_community or args.tgl_wikipedia:
        tgl = True
        tgl_path = (
            'data/tgl/tgl_wikipedia_2021_100K/tgl_wikipedia_2021_100K-words.txt'
            if args.tgl_wikipedia else
            'data/tgl/tgl_community_2017/tgl_community_2017-words.txt'
            )
        tgl_counts, tgl_total = tgl_counts_total(tgl_path)

        def fil_frequency(s: str):
            return tgl_counts.get(s.lower(), 0) / tgl_total

    def tokenize(s, lang):
        return (
            ja_tokenize(s) if lang == 'ja' else
            space_tokenize(s) if lang == 'si' else
            wf_tokenize(s, lang)
            )

    def frequency(w, lang):
        if (
            counts_total := lang2counts_total.get(lang)
            ) is not None:
            counts, total = counts_total
            return counts.get(w, 0) / total
        return (
            si_frequency(w) if lang == 'si' else
            ca_frequency(w) if ((lang == 'ca') and tecla) else
            fil_frequency(w) if ((lang == 'fil') and tgl) else
            word_frequency(w, lang)
            )

    def min_frequency(s, lang):
        return min(frequency(w, lang) for w in tokenize(s, lang))

    if correlation:
        print(
            'file\tlanguage\tcorrelation\tf_zeros\t'
            'f_mean\tf_sd\tf_min\tf_max\t'
            'c_mean\tc_sd\tc_min\tc_max'
            )
    elif not train and args.metrics:
        print('file\tlanguage\tPearson\'s r\tMAE\tMSE\tR2')

    for path_in, path_out in zip_longest(input_files, output_files):
        with open(path_in, 'r') as fi:
            lang2data_targets_frequencies_gold = defaultdict(lambda: ([], [], [], []))
            for lineno, line in enumerate(fi, 1):
                try:
                    fields = line.rstrip('\n').split('\t')
                    if read_gold:
                        assert len(fields) == 5, f'Unexpected #fields: {len(fields)}'
                    else:
                        assert 4 <= len(fields) <= 5, (
                            f'Unexpected #fields: {len(fields)}'
                            )
                    lang, idx_str = fields[0].split('_', 1)
                    assert 2 <= len(lang) <= 3, f'Unexpected language: {lang}'
                    (
                        data, targets, frequencies, gold
                        ) = lang2data_targets_frequencies_gold[lang]
                    t = fields[3]
                    if len(fields) == 5:
                        g = float(fields[4])
                    f = min_frequency(t, lang)
                except Exception as e:
                    msg = f'Parsing failed: {path_in}:{lineno}: {e}:\n{line}\n'
                    if args.skip_errors:
                        print(f'{msg}Skipping.\n', file=sys.stderr)
                        continue  # Skip this item
                    raise Exception(msg) from None
                data.append(fields)
                targets.append(t)
                frequencies.append(f)
                if len(fields) == 5:
                    gold.append(g)

            if not (correlation or train):
                fo = open(path_out, 'w')
            else:
                fo = None

            for lang, (
                data, targets, frequencies, gold
                ) in lang2data_targets_frequencies_gold.items():
                zero_t = [t for t, f in zip(targets, frequencies) if f == 0]
                if args.verbose:
                    print(
                        f'{lang}: #={len(targets)}, '
                        f'zero frequency #={len(zero_t)}: {zero_t}'
                        )
                f = np.array(frequencies)
                f_zero = (f == 0)
                f[f_zero] = FREQ_EPS
                logf = np.log10(f)
                c = np.array(gold) if gold else None

                if correlation:
                    assert c is not None
                    r = pearson_r(logf, c)
                    prop_zero   = f_zero.mean()
                    logf_min    = np.min(logf[~f_zero])
                    logf_max    = np.max(logf)
                    print(
                        f'{path_in}\t{LANG2FULL_NAME[lang]}\t{r:.4f}\t{prop_zero:.4f}\t'
                        f'{logf.mean():.4f}\t{logf.std():.4f}\t'
                        f'{logf_min:.4f}\t{logf_max:.4f}\t'
                        f'{c.mean():.4f}\t{c.std():.4f}\t{c.min():.4f}\t{c.max():.4f}'
                        )
                elif train:
                    assert c is not None
                    linear_regression = RidgeCV().fit(
                        logf.reshape(-1, 1), c
                        )
                    model_path = os.path.join(model_dir, f'{lang}.model')
                    dump(linear_regression, model_path)
                else:
                    model_path = os.path.join(model_dir, f'{lang}.model')
                    linear_regression = load(model_path)
                    pred    = linear_regression.predict(logf.reshape(-1, 1))
                    pred    = np.minimum(1.0, np.maximum(0.0, pred))    # clip to 0...1
                    if args.metrics:
                        if c is None:
                            raise Exception(
                                'Missing gold labels, cannot compute metrics.'
                                )
                        r = pearson_r(c, pred)
                        mae = mae_score(c, pred)
                        mse = mse_score(c, pred)
                        r2 = r2_score(c, pred)
                        print(
                            f'{path_in}\t{LANG2FULL_NAME[lang]}\t{r:.4f}\t{mae:.4f}\t'
                            f'{mse:.4f}\t{r2:.4f}'
                            )
                    for fields, p in zip(data, pred):
                        print('\t'.join((
                            *fields[:4], str(p)
                            )), file=fo)

            if fo is not None:
                fo.close()


if __name__ == '__main__':
    main(parse_args())
