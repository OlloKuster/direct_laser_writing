from plotter.plotting_metalens import metalens_regular_intermediate_plot, metalens_robust_intermediate_plot, \
    metalens_regular_final_plot, metalens_robust_final_plot
from plotter.plotting_mode_converter import mode_converter_regular_final_plot, mode_converter_regular_intermediate_plot, \
    mode_converter_robust_intermediate_plot, mode_converter_robust_final_plot
from plotter.plotting_multiplexer import multiplexer_regular_intermediate_plot, multiplexer_regular_final_plot
from plotter.plotting_polarization_splitter import polarization_splitter_regular_intermediate_plot, \
    polarization_splitter_regular_final_plot
from plotter.plotting_power_splitter import power_splitter_regular_intermediate_plot, power_splitter_regular_final_plot, \
    power_splitter_robust_intermediate_plot, power_splitter_robust_final_plot


def plot_loader(plot: str, *args):
    """
    Loads the respective plotters as a function.
    :param plot: Selects which plotter function will be returned.
                 Modes:
                  "eval_regular" Intermediate plotting for regular optimization.
                  "final_regular" Final plotting for regular optimization.
                  "eval_robust" Intermediate plotting for robust optimization.
                  "final_robust" Final plotting for robust optimization.
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

    if plot == "mc_eval_regular":
        return mode_converter_regular_intermediate_plot()
    if plot == "mc_final_regular":
        return mode_converter_regular_final_plot()

    if plot == "mc_eval_robust":
        return mode_converter_robust_intermediate_plot()
    if plot == "mc_final_robust":
        return mode_converter_robust_final_plot()

    if plot == "muliplex_eval_regular":
        return multiplexer_regular_intermediate_plot()
    if plot == "muliplex_final_regular":
        return multiplexer_regular_final_plot()

    if plot == "pol_splitter_regular":
        return polarization_splitter_regular_intermediate_plot()
    if plot == "pol_splitter_final":
        return polarization_splitter_regular_final_plot()

    if plot == "power_splitter_regular":
        return power_splitter_regular_intermediate_plot()
    if plot == "power_splitter_final":
        return power_splitter_regular_final_plot()

    if plot == "power_splitter_robust":
        return power_splitter_robust_intermediate_plot()
    if plot == "power_splitter_robust_final":
        return power_splitter_robust_final_plot()
