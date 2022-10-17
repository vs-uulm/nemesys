"""
Evaluation of field type clustering quality:

Transform the output of the clustering process performed by nemeftr_cluster-segments.py
into a table of cluster quality scores. It expects the input to be named segment-cluster-statistics.csv
as defined in utils.evaluationHelpers.scStatsFile
and outputs scoreTable.csv
"""

import csv, os
from tabulate import tabulate
from typing import Dict, List

from nemere.utils.evaluationHelpers import reportFolder

cols = [
    #   0           1           2               3               4                5           6          7
    'run_title', 'trace', 'conciseness', 'cluster_label', 'most_freq_type', 'precision', 'recall', 'cluster_size'
]

def typedrecallsums(clusterlist: List) -> Dict[str, float]:
    """
    TODO set typedrecallsum to explicit "0.0" for types that are present in trace but not the majority of any cluster.

    :param clusterlist:
    :return:
    """
    # recall for clusters and their most frequent type
    typedrecall = [(e[cols[4]].split(':')[0] if e[cols[3]] != "NOISE" else "NOISE", float(e[cols[6]]))
                   for e in clusterlist]

    # recall sums per group of field type in all clusters
    trsums = dict()
    for t, r in typedrecall:
        if t not in trsums:
            trsums[t] = 0
        trsums[t] += r
    return trsums



if __name__ == '__main__':
    scStatsFile = os.path.join(reportFolder, 'segment-cluster-statistics.csv')
    cstat = dict()
    with open(scStatsFile, 'r') as csvfile:
        cstatr = csv.DictReader(csvfile)

        for colheader in cols:
            if colheader not in cstatr.fieldnames:
                print("incompatible csv format!", colheader, "missing.")
                print(cstatr.fieldnames)
                exit(1)

        for row in cstatr:
            analysis = (row[cols[0]], row[cols[1]])
            if analysis not in cstat:
                cstat[analysis] = list()
            cstat[analysis].append(row)

    # min precision per run - cols[5]: precision, cols[3]: cluster_label
    mppr = {k: min([float(e[cols[5]]) for e in v if e[cols[3]] != "NOISE"]) for k, v in cstat.items()
            if len([c[cols[5]] for c in v if c[cols[3]] != "NOISE"]) > 0}
    smppr = sorted(list(mppr.items()), key=lambda x: x[1])
    print(tabulate(smppr, headers=['Analysis', 'Min precision per run'], tablefmt="pipe"))

    # # recall sums per most frequent type of all clusters in the run with the least type-mixed cluster results
    # leastmixedrecallsums = typedrecallsums(cstat[smppr[-1][0]])

    # scores per run (sorted by min precision per run)
    mcstat = list()
    ftypes = set()
    for k in [l[0] for l in smppr]:
        trs = typedrecallsums(cstat[k])
        mcstat.append({'analysis': k[0], 'atrace': k[1], 'mppr': mppr[k], **trs})
        ftypes.update(trs.keys())

    # make score table from list of dicts
    # scoreheaders = sorted({t for e in mcstat for t in list(e.keys())}, key=lambda x: 'az_' if x in ('mppr', 'NOISE') else x)
    scoreheaders = ['analysis', 'atrace', 'mppr'] + sorted(ftypes, key=lambda x: 'a_' + x if x in ('NOISE', '[unknown]') else x)
    scoretable = [ [ e[h] if h in e else None for h in scoreheaders ] for e in mcstat]
    print(tabulate(scoretable, scoreheaders, tablefmt="pipe"))

    with open(os.path.join(reportFolder, 'scoreTable.csv'), 'w') as scorefile:
        sfw = csv.writer(scorefile)
        sfw.writerow(scoreheaders)
        for line in scoretable:
            sfw.writerow(line)

