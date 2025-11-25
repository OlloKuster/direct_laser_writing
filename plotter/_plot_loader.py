from plotter.plotting_metalens import metalens_regular_intermediate_plot, metalens_robust_intermediate_plot, \
    metalens_regular_final_plot, metalens_robust_final_plot


def plot_loader(plot: str, *args):
    if plot == "eval_regular":
        return metalens_regular_intermediate_plot()
    if plot == "final_regular":
        return metalens_regular_final_plot()

    if plot == "eval_robust":
        return metalens_robust_intermediate_plot()
    if plot == "final_robust":
        return metalens_robust_final_plot()


