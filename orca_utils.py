"""
This module contains the utilities for Orca-based applications,
including a class for structural variants and plotting utilities.
"""

import os
import pathlib
import uuid

from copy import deepcopy
from collections import OrderedDict, namedtuple
from bisect import bisect

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from colormaps import hnh_cmap_ext5, bwcmap

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

ORCA_PATH = str(pathlib.Path(__file__).parent.absolute())

FLY_SEQUENCE_LENGTH = 250000
FLY_WINDOW_RADIUS = FLY_SEQUENCE_LENGTH // 2
FLY_BASE_RESOLUTION = 125
FLY_LEVELS = (8, 4, 2, 1)
FLY_LABELS = ["125bp", "250bp", "500bp", "1kb"]


RefSegment = namedtuple("RefSegment", ["chr", "start", "end", "strand"])


class StructuralChange2:
    """
    Represent structural changes on a chromosome as a list of reference segments.
    Coordinates are 0-based, end-exclusive internally.
    """

    def __init__(self, chr_name, chr_len):
        self.segments = [RefSegment(chr_name, 0, int(chr_len), "+")]
        self._ins_counter = 0
        self._update_coord_points()

    @classmethod
    def _from_segments(cls, segments):
        obj = cls.__new__(cls)
        obj.segments = segments
        obj._ins_counter = 0
        obj._update_coord_points()
        return obj

    def _update_coord_points(self):
        coord_points = [0]
        cur = 0
        for seg in self.segments:
            cur += seg.end - seg.start
            coord_points.append(cur)
        self.coord_points = coord_points

    def _segment_length(self, seg):
        return seg.end - seg.start

    def _slice_segment(self, seg, offset, length):
        if length <= 0:
            return None
        if seg.strand == "+":
            start = seg.start + offset
            end = start + length
        else:
            end = seg.end - offset
            start = end - length
        return RefSegment(seg.chr, start, end, seg.strand)

    def _split_at(self, pos):
        left = []
        right = []
        cur = 0
        for seg in self.segments:
            seglen = self._segment_length(seg)
            next_cur = cur + seglen
            if pos <= cur:
                right.append(seg)
            elif pos >= next_cur:
                left.append(seg)
            else:
                offset = pos - cur
                left_seg = self._slice_segment(seg, 0, offset)
                right_seg = self._slice_segment(seg, offset, seglen - offset)
                if left_seg is not None:
                    left.append(left_seg)
                if right_seg is not None:
                    right.append(right_seg)
            cur = next_cur
        return left, right

    def delete(self, start, end):
        if end < start:
            return
        end_exclusive = end + 1
        left, remainder = self._split_at(start)
        temp = StructuralChange2._from_segments(remainder)
        _, right = temp._split_at(end_exclusive - start)
        self.segments = left + right
        self._update_coord_points()

    def duplicate(self, start, end):
        if end < start:
            return
        end_exclusive = end + 1
        left, remainder = self._split_at(start)
        temp = StructuralChange2._from_segments(remainder)
        mid, right = temp._split_at(end_exclusive - start)
        self.segments = left + mid + mid + right
        self._update_coord_points()

    def invert(self, start, end):
        if end < start:
            return
        end_exclusive = end + 1
        left, remainder = self._split_at(start)
        temp = StructuralChange2._from_segments(remainder)
        mid, right = temp._split_at(end_exclusive - start)
        inverted = [
            RefSegment(seg.chr, seg.start, seg.end, "-" if seg.strand == "+" else "+")
            for seg in reversed(mid)
        ]
        self.segments = left + inverted + right
        self._update_coord_points()

    def insert(self, pos, length, strand="+"):
        left, right = self._split_at(pos)
        ins_name = f"ins{self._ins_counter}"
        self._ins_counter += 1
        ins_seg = RefSegment(ins_name, 0, int(length), strand)
        self.segments = left + [ins_seg] + right
        self._update_coord_points()

    def __add__(self, other):
        return StructuralChange2._from_segments(self.segments + other.segments)

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise TypeError("StructuralChange2 only supports slicing.")
        start = 0 if key.start is None else int(key.start)
        end = self.coord_points[-1] if key.stop is None else int(key.stop)
        if end <= start:
            return []
        start = max(0, start)
        end = min(self.coord_points[-1], end)
        out = []
        cur = 0
        for seg in self.segments:
            seglen = self._segment_length(seg)
            seg_start = cur
            seg_end = cur + seglen
            if seg_end <= start:
                cur = seg_end
                continue
            if seg_start >= end:
                break
            overlap_start = max(start, seg_start)
            overlap_end = min(end, seg_end)
            offset = overlap_start - seg_start
            length = overlap_end - overlap_start
            sub = self._slice_segment(seg, offset, length)
            if sub is not None:
                out.append([sub.chr, sub.start, sub.end, sub.strand])
            cur = seg_end
        return out


def _draw_region(ax, linestart, lineend, color, matlen):
    ax.plot(
        [-matlen / 50, -matlen / 50],
        [matlen * linestart - 0.5 - 0.1, (matlen) * lineend - 0.5 + 0.1],
        solid_capstyle="butt",
        color=color,
        linewidth=8,
        zorder=10,
        clip_on=False,
    )


def _draw_site(ax, linepos, mode, matlen, color="black"):
    if mode == "double":
        ax.plot(
            [-(matlen) / 20, -0.5],
            [(matlen) * linepos - (matlen) / 100.0 - 0.5, (matlen) * linepos - 0.5],
            color=color,
            linewidth=0.2,
            zorder=10,
            clip_on=False,
        )
        ax.plot(
            [-(matlen) / 20, -0.5],
            [(matlen) * linepos + (matlen) / 100.0 - 0.5, (matlen) * linepos - 0.5],
            color=color,
            linewidth=0.2,
            zorder=10,
            clip_on=False,
        )
    elif mode == "single":
        ax.plot(
            [-(matlen) / 20, -0.5],
            [(matlen) * linepos - 0.5, (matlen) * linepos - 0.5],
            color=color,
            linewidth=0.2,
            zorder=10,
            clip_on=False,
        )


def genomeplot(
    output,
    show_genes=False,
    show_tracks=False,
    show_coordinates=True,
    unscaled=False,
    file=None,
    cmap=None,
    unscaled_cmap=None,
    colorbar=True,
    maskpred=False,
    vmin=-1,
    vmax=2,
    model_labels=["NC12", "NC14"],
):
    """
    Plot the multiscale prediction outputs for the Orca-Fly 250kb model.

    Parameters
    ----------
    output : dict
        The result dictionary returned by `genomepredict`.
    show_genes : bool, optional
        Ignored for Orca-Fly (kept for API compatibility).
    show_tracks : bool, optional
        Ignored for Orca-Fly (kept for API compatibility).
    show_coordinates : bool, optional
        Default is True. If True, annotate the generated plot with the
        genome coordinates.
    unscaled : bool, optional
        Default is False. If True, plot the predictions and observations
        without normalizing by distance-based expectation.
    file : str or None, optional
        Default is None. The output file prefix. No output file is generated
        if set to None.
    cmap : str or None, optional
        Default is None. The colormap for plotting scaled interactions (log
        fold over distance-based background). If None, use colormaps.hnh_cmap_ext5.
    unscaled_cmap : str or None, optional
        Default is None. The colormap for plotting unscaled interactions (log
        balanced contact score). If None, use colormaps.hnh_cmap_ext5.
    colorbar : bool, optional
        Default is True. Whether to plot the colorbar.
    maskpred : bool, optional
        Default is True. If True, the prediction heatmaps are masked at positions
        where the observed data have missing values when observed data are provided
        in output dict.
    vmin : int, optional
        Default is -1. The lowerbound value for heatmap colormap.
    vmax : int, optional
        Default is 2. The upperbound value for heatmap colormap.
    model_labels : list(str), optional
        Model labels for plotting. Default is ["NC12", "NC14"].

    Returns
    -------
    None
    """
    if cmap is None:
        cmap = hnh_cmap_ext5
    if unscaled_cmap is None:
        unscaled_cmap = hnh_cmap_ext5

    if output["predictions"][0][0].ndim == 3:
        predictions = []
        for modeli in range(len(output["predictions"])):
            ndims = output["predictions"][modeli][0].shape[0]
            predictions += [
                [
                    output["predictions"][modeli][i][j]
                    for i in range(len(output["predictions"][modeli]))
                ]
                for j in range(ndims)
            ]
        output["predictions"] = predictions

    n_axes = len(output["predictions"])
    if output.get("experiments") is not None:
        n_axes += len(output["predictions"])

    n_cols = len(output["predictions"][0])
    fig, all_axes = plt.subplots(
        figsize=(6 * n_cols, 6 * n_axes), nrows=n_axes, ncols=n_cols
    )

    for row_axes in all_axes:
        for ax in row_axes:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

    labels = (
        FLY_LABELS if n_cols == len(FLY_LABELS) else [f"L{i+1}" for i in range(n_cols)]
    )
    for i, xlabel in enumerate(labels):
        all_axes[-1, i].set_xlabel(xlabel, labelpad=20, fontsize=20, weight="black")

    if output.get("experiments") is not None:
        current_axis = 0
        for label in model_labels:
            for suffix in [" Pred", " Obs"]:
                all_axes[current_axis, 0].set_ylabel(
                    label + suffix,
                    labelpad=20,
                    fontsize=20,
                    weight="black",
                    rotation="horizontal",
                    ha="right",
                    va="center",
                )
                current_axis += 1
    else:
        current_axis = 0
        for label in model_labels:
            for suffix in [" Pred"]:
                all_axes[current_axis, 0].set_ylabel(
                    label + suffix,
                    labelpad=20,
                    fontsize=20,
                    weight="black",
                    rotation="horizontal",
                    ha="right",
                    va="center",
                )
                current_axis += 1

    current_row = 0
    for model_i in range(len(output["predictions"])):
        for ii, ax in enumerate(reversed(all_axes[current_row])):
            s = int(output["start_coords"][ii])
            e = int(output["end_coords"][ii])
            regionstr = output["chr"] + ":" + str(s) + "-" + str(e)
            if show_coordinates:
                ax.set_title(regionstr, fontsize=14, pad=4)
            if unscaled:
                plotmat = output["predictions"][model_i][ii] + np.log(
                    output["normmats"][model_i][ii]
                )
                im = ax.imshow(
                    plotmat,
                    interpolation="none",
                    cmap=unscaled_cmap,
                    vmax=np.nanmax(np.diag(plotmat, k=1)),
                )
            else:
                plotmat = output["predictions"][model_i][ii]
                im = ax.imshow(
                    plotmat, interpolation="none", cmap=cmap, vmin=vmin, vmax=vmax
                )

            if maskpred and output.get("experiments") is not None:
                ax.imshow(
                    np.isnan(output["experiments"][0][ii]),
                    interpolation="none",
                    cmap=bwcmap,
                )

        current_row += 1

        if output.get("experiments") is not None:
            for ii, ax in enumerate(reversed(all_axes[current_row])):
                s = int(output["start_coords"][ii])
                e = int(output["end_coords"][ii])
                regionstr = output["chr"] + ":" + str(s) + "-" + str(e)
                if show_coordinates:
                    ax.set_title(regionstr, fontsize=14, pad=4)
                if unscaled:
                    plotmat = output["experiments"][model_i][ii] + np.log(
                        output["normmats"][model_i][ii]
                    )
                    im = ax.imshow(
                        plotmat,
                        interpolation="none",
                        cmap=unscaled_cmap,
                        vmax=np.nanmax(np.diag(plotmat, k=1)),
                    )
                else:
                    plotmat = output["experiments"][model_i][ii]
                    im = ax.imshow(
                        plotmat, interpolation="none", cmap=cmap, vmin=vmin, vmax=vmax
                    )

            current_row += 1

    if colorbar:
        fig.colorbar(im, ax=all_axes, location="right", fraction=0.015, pad=0.02)

    if file is not None:
        with PdfPages(file) as pdf:
            pdf.savefig(fig, dpi=300)
    return fig, all_axes


def process_anno(anno_scaled, base=0, window_radius=FLY_WINDOW_RADIUS):
    """
    Process annotations to the format used by Orca plotting
    functions such as `genomeplot` and `genomeplot_256Mb`.

    Parameters
    ----------
    anno_scaled : list(list(...))
        List of annotations. Each annotation can be a region specified by
        `[start: int, end: int, info:str]` or a position specified by
        `[pos: int, info:str]`.
        Acceptable info strings for region currently include color names for
        matplotlib. Acceptable info strings for position are currently
        'single' or 'double', which direct whether the annotation is drawn
        by single or double lines.
    base : int
        The starting position of the 32Mb (if window_radius is 16000000)
        or 256Mb (if window_radius is 128000000) region analyzed.
    window_radius : int
        The size of the region analyzed. For Orca-Fly this should be 125000 (250kb window).

    Returns
    -------
    annotation : list
        Processed annotations with coordinates transformed to relative coordinate
        in the range of 0-1.
    """
    annotation = []
    for r in anno_scaled:
        if len(r) == 3:
            annotation.append(
                [
                    (r[0] - base) / (window_radius * 2),
                    (r[1] - base) / (window_radius * 2),
                    r[2],
                ]
            )
        elif len(r) == 2:
            annotation.append([(r[0] - base) / (window_radius * 2), r[1]])
        else:
            raise ValueError
    return annotation


def coord_clip(pos, chrlen, binsize=1000, window_radius=FLY_WINDOW_RADIUS):
    """
    Clip the coordinate to make sure that full window
    centered at the coordinate to stay within chromosome boundaries.
    coord_clip also try to preserve the relative position of the coordinate
    to the grid as specified by binsize whenever possible.

    Parameters
    ----------
    x : int or numpy.ndarray
        Coordinates to round.
    gridsize : int
        The gridsize to round by

    Returns
    -------
    int
        The clipped coordinate
    """
    if pos < binsize or pos > chrlen - binsize:
        return np.clip(pos, window_radius, chrlen - window_radius)
    else:
        if (chrlen - window_radius) % binsize > pos % binsize:
            endclip = (
                chrlen
                - window_radius
                - ((chrlen - window_radius) % binsize - pos % binsize)
            )
        else:
            endclip = (
                chrlen
                - window_radius
                - binsize
                - ((chrlen - window_radius) % binsize - pos % binsize)
            )

        return np.clip(pos, window_radius + pos % binsize, endclip)


def coord_round(x, gridsize=FLY_BASE_RESOLUTION):
    """
    Round coordinate to multiples of gridsize.

    Parameters
    ----------
    x : int or numpy.ndarray
        Coordinates to round.
    gridsize : int
        The gridsize to round by

    Returns
    -------
    int
        The rounded coordinate
    """
    return x - x % gridsize
