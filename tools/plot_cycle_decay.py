#!/usr/bin/env python3
"""Plot a local eigenmode envelope from saved reference certificates; no ODE."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new figure path; existing evidence is preserved')
    refs = json.loads(args.summary.read_text())['references']
    cycles = np.arange(9)
    colors = {'core': '#227c9d', 'failed-near-edge': '#bb4e32'}
    labels = {'core': 'Core', 'failed-near-edge': 'Near-edge setting'}
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(8, 4.9), layout='constrained')
    for setting in colors:
        rho = refs[setting]['transverse_spectral_radius']
        if refs[setting]['qualified'] is not True or not 0 < rho < 1:
            raise ValueError('Qualified attracting reference required')
        ax.semilogy(cycles, rho**cycles, 'o--', color=colors[setting], linewidth=2,
                    label=f'{labels[setting]}: {rho:.4f} × amplitude per cycle')
    ax.set(xlabel='Completed cycles', ylabel='Fraction of initial eigenmode amplitude',
           xticks=cycles, xlim=(-.1, 8.1), ylim=(1e-10, 2))
    ax.grid(axis='y', alpha=.2)
    ax.legend(frameon=False, loc='lower left')
    ax.set_title('Local transverse disturbances have very different decay rates', pad=32)
    ax.text(.5, 1.035, 'Linear prediction at integer cycles; dashed lines are guides',
            transform=ax.transAxes, ha='center', fontsize=10)
    fig.text(.5, -.015, 'From qualified reference-cycle multipliers. This is not a measured pulse response or bit-accuracy result.',
             ha='center', fontsize=8.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches='tight')
    print(args.output)


if __name__ == '__main__':
    main()
