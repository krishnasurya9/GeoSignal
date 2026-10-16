import sys, os
sys.path.insert(0, '.')
from experiments.case_studies.case_registry import ALL_CASES
from src.geosignal.visualization.propagation_map import draw_propagation_map

for case in ALL_CASES:
    print(f'Drawing {case.case_id}: {case.label} ...', end=' ', flush=True)
    paths = draw_propagation_map(case)
    print(f'saved -> {paths["high"]}')

print('All done.')
