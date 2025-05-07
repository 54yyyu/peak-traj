from typing import Iterable, Callable, List, Optional # Added List, Optional
from ipywidgets import VBox, HBox, Widget, IntSlider, interactive_output, Output, Label, Button, FloatSlider, TagsInput, \
    Layout, Text # Added Text
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Assuming these are moved or path adjusted
from peak_trajectory.extract_peak_trajectory import extract_peak_trajectory # Renamed
from peak_trajectory.plot.peak_trajectory_plots import plot_peak_trajectory_3d # Renamed

# Renamed class
class ExtractPeakTrajectoryWidget(HBox):
    widgets: dict[str, Widget] = None
    t_list_widgets: list[IntSlider] = None
    dims_widget: IntSlider = None
    k_widget: IntSlider = None
    quantile_widget: FloatSlider = None
    label_peaks_widget: TagsInput = None # Renamed from label_genes

    f: Callable = None
    option_panel: VBox = None
    output: Output = None
    peak_trajectory: pd.DataFrame = None # Renamed
    max_t: int = 20

    @property
    def dims(self) -> int:
        return self.dims_widget.value #

    @property
    def k(self) -> int:
        return self.k_widget.value #

    @property
    def quantile(self) -> float:
        return self.quantile_widget.value #

    @property
    def tlist(self) -> list[int]:
        return [t.value for t in self.t_list_widgets] #

    @property
    def label_peaks(self) -> list[str]: # Renamed
         return self.label_peaks_widget.value # Renamed

    def __init__(
            self,
            peak_embedding: pd.DataFrame, # Renamed
            dist_mat: np.array, # Peak-peak distance matrix
            peak_names: List[str], # Renamed
            t_list: Iterable[int] = (3, 3, 3), #
            label_peaks: Optional[List[str]] = None, # Renamed from label_genes, provide default empty tuple if None
            dims: int = 5, #
            k: int = 10, #
            quantile: float = 0.02, #
            max_t: int = 20, #
    ):
        """
        An interactive widget for optimizing the parameters of extract_peak_trajectory.
        (Adapted from ExtractGeneTrajectoryWidget)

        :param peak_embedding: Peak embedding dataframe # Renamed
        :param dist_mat: Peak-peak Wasserstein distance matrix (symmetric) # Renamed
        :param peak_names: List of peak names/identifiers # Renamed
        :param t_list:  Initial number of diffusion times for each trajectory #
        :param label_peaks: Initial peak labels to plot (default: None) # Renamed
        :param dims: Initial dimensions of peak embedding to use # Renamed
        :param k: Initial adaptive kernel bandwidth #
        :param quantile: Initial thresholding parameter #
        :param max_t: Maximum value for t sliders. Default: 20 #
        :return: A widget
        """
        super().__init__()
        self.max_t = max_t
        # Use empty list if None is passed
        initial_label_peaks = list(label_peaks) if label_peaks is not None else []

        # Renamed build_plot, parameters and function calls
        def build_plot_peaks(dims: int, k: int, quantile: float, label_peaks: list[str], **kwargs):
            tl = [v for n, v in kwargs.items() if n.startswith("_t_")]
            # Use plt.figure() to ensure a new figure context for updates
            fig = plt.figure(figsize=(6, 4))
            # Clear previous plot if exists in output - safer update
            self.output.clear_output(wait=True)
            with self.output:
                try:
                    # Call adapted functions
                    self.peak_trajectory = extract_peak_trajectory(peak_embedding, dist_mat, peak_names=peak_names,
                                                                   t_list=tl, dims=dims, k=k, quantile=quantile) # Renamed
                    # Pass fig's ax to plotting function if it supports it, otherwise plot directly
                    # plot_peak_trajectory_3d(self.peak_trajectory, label_peaks=label_peaks, ax=fig.add_subplot(111, projection='3d')) # Ideal
                    plot_peak_trajectory_3d(self.peak_trajectory, label_peaks=label_peaks) # Current implementation plots itself
                    plt.show() # Show the plot within the output widget
                except Exception as e:
                     print(f"Error generating plot: {e}")
                     plt.close(fig) # Close figure on error


        self.f = build_plot_peaks # Renamed

        self.build_widgets(t_list, dims, k, quantile, initial_label_peaks, peak_names) # Renamed label_peaks
        self.build_ui()

    def build_ui(self):
        def add_trajectory(_):
            # Add a new slider with default value 3
            self.build_widgets(self.tlist + [3], self.dims, self.k, self.quantile, self.label_peaks) # Pass current label_peaks
            self.build_ui() # Rebuild UI to include new slider

        def remove_trajectory(_):
            if len(self.tlist) > 1: # Prevent removing the last slider
                self.build_widgets(self.tlist[:-1], self.dims, self.k, self.quantile, self.label_peaks) # Pass current label_peaks
                self.build_ui() # Rebuild UI

        add_btn = Button(description="Add Trajectory", icon="plus", button_style='info') # Add text
        add_btn.on_click(add_trajectory)
        remove_btn = Button(description="Remove Last", icon="minus", button_style='warning') # Add text
        remove_btn.on_click(remove_trajectory)
        # Disable remove button if only one slider exists
        remove_btn.disabled = len(self.t_list_widgets) <= 1


        self.option_panel = VBox(children=[
            Label("Extract peak trajectories options"), # Renamed
            self.dims_widget, self.k_widget, self.quantile_widget,
            Label("t_list (Diffusion times per trajectory)"), *self.t_list_widgets, HBox([add_btn, remove_btn]), #
            Label("Peaks to Label"), self.label_peaks_widget, # Renamed
        ], layout=Layout(width='auto', padding='10px')) # Adjust layout

        # Link widgets to the function
        self.output = interactive_output(self.f, self.widgets)
        self.output.layout = Layout(height='450px', border='1px solid grey', overflow_y='auto') # Adjust layout

        # Initial call to generate the first plot
        # self.f(**{k: w.value for k, w in self.widgets.items()}) # Trigger initial plot - interactive_output handles this

        self.children = (self.option_panel, self.output) #

    # Renamed label_genes -> label_peaks, gene_names -> peak_names
    def build_widgets(self, t_list: Iterable[int], dims: int, k: int, quantile: float, label_peaks: list[str],
                      peak_names: list[str]):
        self.t_list_widgets = [IntSlider(t, 1, self.max_t, description=f"Traj-{i + 1} t", continuous_update=False) # Shorten desc
                               for i, t in enumerate(t_list)]
        self.dims_widget = IntSlider(dims, 2, 20, description="dims", continuous_update=False) #
        self.k_widget = IntSlider(k, 2, 20, description="k (DM/RW)", continuous_update=False) # Clarify k usage
        self.quantile_widget = FloatSlider(value=quantile, min=0.001, max=0.2, step=0.001, description="quantile", readout_format='.3f', continuous_update=False) # Adjust max/step, format readout
        # Renamed label_genes widget
        self.label_peaks_widget = TagsInput(value=list(label_peaks), # Ensure list
                                            allowed_tags=sorted(peak_names), # Use peak_names for suggestions
                                            allow_duplicates=False, layout=Layout(width='auto')) # Adjust width

        self.widgets = {"dims": self.dims_widget, "k": self.k_widget,
                        "quantile": self.quantile_widget, "label_peaks": self.label_peaks_widget, # Renamed
                        **{f"_t_{i}": t for i, t in enumerate(self.t_list_widgets)}}