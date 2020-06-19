"""
Evaluation of field type clustering quality:

Transform the output of the clustering process performed by characterize_fieldtypes.py
into a table of cluster quality scores. It expects the input to be named clusterStatisticsHDBSCAN.csv
and outputs scoreTable.csv
"""

import csv
from tabulate import tabulate

cols = [
    #   0           1           2               3               4          5           6
    'run_title', 'trace', 'conciseness', 'most_freq_type', 'precision', 'recall', 'cluster_size'
]

def typedrecallsums(clusterlist):
    """
    TODO set typedrecallsum to explicit "0.0" for types that are present in trace but not the majority of any cluster.

    :param clusterlist:
    :return:
    """
    # recall for clusters and their most frequent type
    typedrecall = [(e[cols[3]].split(':')[0], float(e[cols[5]])) for e in clusterlist]

    # recall sums per group of field type in all clusters
    trsums = dict()
    for t, r in typedrecall:
        if t not in trsums:
            trsums[t] = 0
        trsums[t] += r
    return trsums



if __name__ == '__main__':
    cstat = dict()
    with open('clusterStatisticsHDBSCAN.csv', 'r') as csvfile:
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

    # min precision per run
    mppr = {k: min([float(e[cols[4]]) for e in v if e[cols[3]] != "NOISE"]) for k, v in cstat.items()
            if len([c[cols[4]] for c in v if c[cols[3]] != "NOISE"]) > 0}
    smppr = sorted(list(mppr.items()), key=lambda x: x[1])
    print(tabulate(smppr, headers=['Analysis', 'Min precision per run'], tablefmt="pipe"))

    # # recall sums per most frequent type of all clusters in the run with the least type-mixed cluster results
    # leastmixedrecallsums = typedrecallsums(cstat[smppr[-1][0]])

    # scores per run (sorted by min precision per run)
    mcstat = [{'analysis': k[0], 'atrace': k[1], 'mppr': mppr[k], **typedrecallsums(cstat[k])} for k in [l[0] for l in smppr]]

    # make score table from list of dicts
    scoreheaders = sorted({t for e in mcstat for t in list(e.keys())}, key=lambda x: 'az_' if x in ('mppr', 'NOISE') else x)
    scoretable = [ [ e[h] if h in e else None for h in scoreheaders ] for e in mcstat]
    print(tabulate(scoretable, scoreheaders, tablefmt="pipe"))

    with open('scoreTable.csv', 'w') as scorefile:
        sfw = csv.writer(scorefile)
        sfw.writerow(scoreheaders)
        for line in scoretable:
            sfw.writerow(line)

