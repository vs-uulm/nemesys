"""
Evaluation of field type clustering quality:

Transform the output of the clustering process performed by characterize_fieldtypes.py
into a table of cluster quality scores. It expects the input to be named clusterStatisticsHDBSCAN.csv
and outputs scoreTable.csv
"""

import csv
cstat = dict()
with open('clusterStatisticsHDBSCAN.csv', 'r') as csvfile:
    cstatr = csv.reader(csvfile)
    next(cstatr)
    for title, *row in cstatr:
        if title not in cstat:
            cstat[title] = list()
        cstat[title].append(row)

from tabulate import tabulate

# min precision per run
mppr = {k: min([float(e[2]) for e in v]) for k, v in cstat.items()}
smppr = sorted(list(mppr.items()), key=lambda x: x[1])
print(tabulate(smppr, headers=['Analysis','Min precision per run'], tablefmt="pipe"))

def typedrecallsums(clusterlist):
    # recall for clusters and their most frequent type
    typedrecall = [(e[1].split(':')[0], float(e[3])) for e in clusterlist]

    # recall sums per group of field type in all clusters
    trsums = dict()
    for t, r in typedrecall:
        if t not in trsums:
            trsums[t] = 0
        trsums[t] += r
    return trsums

# # recall sums per most frequent type of all clusters in the run with the least type-mixed cluster results
# leastmixedrecallsums = typedrecallsums(cstat[smppr[-1][0]])

# scores per run (sorted by min precision per run)
mcstat = [{'analysis': k, 'mppr': mppr[k], **typedrecallsums(cstat[k])} for k in [l[0] for l in smppr]]

# make score table from list of dicts
scoreheaders = sorted({t for e in mcstat for t in list(e.keys())}, key=lambda x: 'analysis_' if x == 'mppr' else x)
scoretable = [ [ e[h] if h in e else None for h in scoreheaders ] for e in mcstat]
print(tabulate(scoretable, scoreheaders, tablefmt="pipe"))

with open('scoreTable.csv', 'w') as scorefile:
    sfw = csv.writer(scorefile)
    sfw.writerow(scoreheaders)
    for line in scoretable:
        sfw.writerow(line)
