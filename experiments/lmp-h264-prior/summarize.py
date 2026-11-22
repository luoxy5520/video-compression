"""Summarize fixed-budget motion diagnostics and frozen-head mode recall."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean


def read_rows(paths):
    return [json.loads(line) for path in paths
            for line in path.read_text().splitlines() if line.strip()]


def summary(rows, by_video=False):
    groups = defaultdict(list)
    for row in rows:
        groups[row['prompt']].append(row)

    def average(metric):
        if by_video:
            return mean(metric(row) for row in rows)
        return mean(mean(metric(row) for row in group) for group in groups.values())

    common = set.intersection(*(set(row['stats']) for row in rows))
    arms = {}
    for arm in sorted(common):
        def get(row, k, field, index):
            return row['stats'][arm][str(k)][field][index]

        coverage = {
            str(k): [average(lambda row: get(row, k, 'coverage', i))
                     for i in range(4)]
            for k in (1, 2, 4, 8)
        }
        arms[arm] = dict(
            coverage=coverage,
            k2_coverage_all_p=average(
                lambda row: get(row, 2, 'coverage', 1)*row['eligible']/row['p_blocks']),
            k2_centers=average(lambda row: row['stats'][arm]['2']['unique']),
            k2_epe_median=average(lambda row: get(row, 2, 'epe_median_p90', 0)),
            k2_epe_p90=average(lambda row: get(row, 2, 'epe_median_p90', 1)),
            k2_sad=average(lambda row: row['stats'][arm]['sad_k2']['mean']),
            k2_sad_regret=average(lambda row: row['stats'][arm]['sad_k2']['regret_mean']),
        )

    mode = {}
    if all(row['mode_stats'] for row in rows):
        mode = {key: average(lambda row: row['mode_stats'][key])
                for key in rows[0]['mode_stats']}
    return dict(videos=len(rows), prompts=len(groups), arms=arms, mode=mode,
                eligible=sum(row['eligible'] for row in rows),
                p_blocks=sum(row['p_blocks'] for row in rows),
                timing={key: average(lambda row: row['timing'][key])
                        for key in rows[0]['timing']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--by-video', action='store_true')
    args = parser.parse_args()
    print(json.dumps(summary(read_rows(args.paths), by_video=args.by_video), indent=2))
