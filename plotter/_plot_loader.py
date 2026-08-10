from plotter.plotting_metalens import metalens_regular_intermediate_plot, metalens_robust_intermediate_plot, \
    metalens_regular_final_plot, metalens_robust_final_plot
from plotter.plotting_power_splitter import power_splitter_regular_intermediate_plot, power_splitter_regular_final_plot, \
    power_splitter_robust_intermediate_plot, power_splitter_robust_final_plot
from plotter.plotting_frequency_filter import frequency_filter_regular_intermediate_plot, frequency_filter_regular_final_plot, frequency_filter_robust_final_plot, frequency_filter_robust_intermediate_plot


def plot_loader(plot: str, *args):
    """
    Loads the respective plotters as a function.
    :param plot: Selects which plotter function will be returned.
                 Modes:
                  "eval_regular" Intermediate plotting for regular optimization.
                  "final_regular" Final plotting for regular optimization.
                  "eval_robust" Intermediate plotting for dlw optimization.
                  "final_robust" Final plotting for dlw optimization.
    :param args:
    :return:
    """
    if plot == "eval_regular":
        return metalens_regular_intermediate_plot()
    if plot == "final_regular":
        return metalens_regular_final_plot()

    if plot == "eval_robust":
        return metalens_robust_intermediate_plot()
    if plot == "final_robust":
        return metalens_robust_final_plot()

    if plot == "power_splitter_regular":
        return power_splitter_regular_intermediate_plot()
    if plot == "power_splitter_final":
        return power_splitter_regular_final_plot()

    if plot == "power_splitter_robust":
        return power_splitter_robust_intermediate_plot()
    if plot == "power_splitter_robust_final":
        return power_splitter_robust_final_plot()

    if plot == "frequency_filter_regular":
        return frequency_filter_regular_intermediate_plot()
    if plot == "frequency_filter_regular_final":
        return frequency_filter_regular_final_plot()
