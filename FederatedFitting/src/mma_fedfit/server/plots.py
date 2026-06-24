import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

import numpy as np
import corner


class Plots:
    PLOT_LAYOUTS = [
        (1, 1), # 0
        (1, 1), # 1
        (1, 2), # 2
        (1, 3), # 3
        (2, 2), # 4
        (2, 3), # 5
        (2, 3), # 6
        (2, 4), # 7
        (2, 4), # 8
        (3, 3), # 9
        (3, 4) # 10
    ]

    def __init__(self, out_dir: Path, params: List[str], samples) -> None:
        self.out_dir = out_dir
        self.params = params
        self.samples = samples
        self.true_values = None
        self.display_true_values = False

    def make_plots(self) -> None:
        self.make_posterior_hists(self.out_dir / 'posterior.png')
        self.make_corner_plots(self.out_dir / 'corner_plots.png')

    def make_posterior_hists(self, path: Path) -> None:
        """
        Create posterior histograms from consensus samples
        """
        medians = np.median(self.samples, axis=0)

        ndim = len(self.params)
        ny, nx = Plots.PLOT_LAYOUTS[ndim]
        fig_sz = (nx * 4 + 4, ny * 4)

        fig, axes = plt.subplots(ny, nx, figsize=fig_sz)

        axes = axes.flatten()

        for i in range(ndim):
            param_samples = np.asarray(self.samples[:, i])
            lower, upper = np.percentile(param_samples, [2.5, 97.5])

            ax = axes[i]
            ax.hist(param_samples, bins=20, color='blue', alpha=0.7, label='Samples')

            # Plot mean value as a vertical line
            ax.axvline(medians[i], color='red', linestyle='--', label=f'Median: {medians[i]:.4f}')
            ax.axvline(lower, color='green', linestyle='--',
                       label=f'lower limit 2.5th percentile: {lower:.4f}')
            ax.axvline(upper, color='green', linestyle='--',
                       label=f'upper limit 97.5th percentile: {upper:.4f}')

            ax.set_title(self.params[i])
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.legend(loc=4)

        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')

    def make_corner_plots(self, path: Path) -> None:
        """
        Create corner plots from consensus samples
        """
        ndim = len(self.params)
        corner.corner(self.samples, labels=self.params, show_titles=True,
            title_fmt='.2f',
            quantiles=[0.025, 0.5, 0.975],  # 95% credible interval
            title_kwargs={'fontsize': 12}, label_kwargs={'fontsize': 14},
            plot_datapoints=True, fill_contours=True,
            levels=(0.68, 0.95,),  # 90% confidence contours
            smooth=1.0, smooth1d=1.0,
            truths=self.true_values[0:ndim] if self.display_true_values else None
        )
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')

