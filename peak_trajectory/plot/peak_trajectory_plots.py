import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Iterable, Any, Optional, List # Added Optional, List

# Renamed from plot_gene_trajectory_3d
def plot_peak_trajectory_3d(
        peak_trajectory: pd.DataFrame, # Renamed
        s: int = 1,
        label_peaks: Optional[List[str]] = None, # Renamed from label_genes
        **kwargs: Any
) -> None:
    """
    Generate a 3D plot of a peak-trajectory object.
    (Adapted from plot_gene_trajectory_3d)

    :param peak_trajectory: a peak trajectory result dataframe # Renamed
    :param s: scatterplot point size (default: 1) #
    :param label_peaks: Peak labels (e.g. peak coordinates or associated gene names) to plot (default: None) # Renamed
    :param kwargs: plot arguments that will be passed to Axes.scatter #
    """
    # Check for necessary columns
    for c in 'DM_1', 'DM_2', 'DM_3', 'selected':
        if c not in peak_trajectory.columns: # Renamed
            raise ValueError(f'Column {c} is not present in peak trajectory DataFrame') # Renamed

    ax = plt.figure().add_subplot(projection='3d') # More explicit figure creation
    selections = pd.Categorical(peak_trajectory.selected) # Renamed

    colors = plt.cm.get_cmap('viridis', len(selections.categories)) # Example colormap

    for i, c in enumerate(selections.categories):
        idxs = peak_trajectory['selected'] == c # More direct boolean indexing
        ax.scatter(xs=peak_trajectory.loc[idxs, 'DM_1'],
                   ys=peak_trajectory.loc[idxs, 'DM_2'],
                   zs=peak_trajectory.loc[idxs, 'DM_3'], #
                   s=s,
                   label=c,
                   color=[colors(i)], # Assign distinct color
                   **kwargs)

    if label_peaks: # Renamed
        peaks_to_label = [p for p in label_peaks if p in peak_trajectory.index] # Check if peaks exist
        if not peaks_to_label:
             print(f"Warning: None of the specified label_peaks found in the trajectory data.")
        else:
             for p in peaks_to_label: # Renamed g to p
                 ax.text(x=peak_trajectory.loc[p, 'DM_1'], # Use .loc for label-based indexing
                         y=peak_trajectory.loc[p, 'DM_2'], #
                         z=peak_trajectory.loc[p, 'DM_3'], #
                         s=p) #
    ax.set_xlabel("DM_1") # Add axis labels
    ax.set_ylabel("DM_2")
    ax.set_zlabel("DM_3")
    ax.legend()
    plt.title("Peak Trajectory Embedding (3D)") # Add title
    plt.show()


# Renamed from plot_gene_trajectory_2d
def plot_peak_trajectory_2d(
        peak_trajectory: pd.DataFrame, # Renamed
        s: int = 1,
        label_peaks: Optional[List[str]] = None, # Renamed from label_genes
        **kwargs: Any
) -> None:
    """
    Generate a 2D plot of a peak-trajectory object.
    (Adapted from plot_gene_trajectory_2d)

    :param peak_trajectory: a peak trajectory result dataframe # Renamed
    :param s: scatterplot point size (default: 1) #
    :param label_peaks: Peak labels to plot (default: None) # Renamed
    :param kwargs: plot arguments that will be passed to sns.scatterplot #
    """
    # Check for necessary columns
    for c in 'DM_1', 'DM_2', 'selected':
        if c not in peak_trajectory.columns: # Renamed
            raise ValueError(f'Column {c} is not present in peak trajectory DataFrame') # Renamed

    plt.figure() # Create new figure
    sns.scatterplot(data=peak_trajectory, # Renamed
                    x='DM_1',
                    y='DM_2',
                    hue='selected',
                    s=s,
                    **kwargs)
    if label_peaks: # Renamed
        ax = plt.gca() #
        peaks_to_label = [p for p in label_peaks if p in peak_trajectory.index] # Check if peaks exist
        if not peaks_to_label:
             print(f"Warning: None of the specified label_peaks found in the trajectory data.")
        else:
             for p in peaks_to_label: # Renamed g to p
                 # Add small offset for labels if needed to avoid overlap
                 ax.text(x=peak_trajectory.loc[p, 'DM_1'], # Use .loc
                         y=peak_trajectory.loc[p, 'DM_2'], #
                         s=p) #
    plt.title("Peak Trajectory Embedding (2D)")
    plt.show()


# Renamed from plot_gene_trajectory_umap
def plot_peak_trajectory_umap(
        adata: sc.AnnData, # Expects adata with peak bin scores in .obs
        trajectory: str = 'Trajectory1',
        other_panels: Iterable[str] = (), #
        reverse: bool = False, #
        cmap: str = 'RdYlBu_r', #
        **kwargs: Any,
) -> None:
    """
    Generate a series of UMAP plots for peak trajectory bins.
    (Adapted from plot_gene_trajectory_umap)

    :param adata: a dataset with peak trajectory metadata (e.g., peak bin scores in .obs) # Modified
    :param trajectory: the prefix name of the trajectory (e.g., 'Trajectory1') #
    :param other_panels: other panels (columns in adata.obs) to be added to the UMAP plot #
    :param reverse: reverse the order of the trajectory bin panels #
    :param cmap: colormap to be used for trajectory bins #
    :param kwargs: plot arguments that will be passed to scanpy.pl.umap #
    """
    other_panels = [other_panels] if isinstance(other_panels, str) else list(other_panels) # Ensure list
    # Updated to find peak bins (e.g., "Trajectory1_peaks1")
    trajectory_panels = sorted([k for k in adata.obs_keys() if k.startswith(f"{trajectory}_peaks")]) # Renamed _genes to _peaks
    if not trajectory_panels:
        raise ValueError(f'No peak bin metadata found for {trajectory} (expecting e.g., {trajectory}_peaks1)') # Modified

    if reverse:
        trajectory_panels.reverse() #
    panels = [*trajectory_panels, *other_panels] #
    # Check if panels exist in adata.obs
    missing_panels = [p for p in panels if p not in adata.obs_keys()]
    if missing_panels:
         raise ValueError(f"Panels not found in adata.obs: {missing_panels}")

    sc.pl.umap(adata, color=panels, cmap=cmap, **kwargs) #