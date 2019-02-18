"""
Functions to create interactive 3D brain volume visualizations,
either html/js snippets, stand-alone html or inserted in a notebook.
The visualizations are powered by the brainsprite.js library.
"""

# Author: Pierre Bellec
# License: MIT

import warnings
import numpy as np
import os
from base64 import encodebytes
import json

import nibabel as nb
import numbers
from nilearn.plotting import cm
from matplotlib import cm as mpl_cm
from matplotlib import colors
from nilearn._utils.niimg import _safe_get_data
from nilearn._utils.extmath import fast_abs_percentile
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from nilearn.plotting.img_plotting import _load_anat, _get_colorbar_and_data_ranges
from nilearn.datasets import load_mni152_template
from nilearn.plotting.js_plotting_utils import HTMLDocument
from nilearn._utils import check_niimg_3d
from nilearn import image
from io import BytesIO , StringIO
from nilearn.plotting.find_cuts import find_xyz_cut_coords

def _get_text_data(data_name):
    """Get data from a text file"""
    data_path = os.path.join(
        os.path.dirname(__file__), 'data', datas_name)
    with open(data_path, 'rb') as f:
        return f.read().decode('utf-8')


def _output_sprite(output_sprite, stat_map_img, mask_img, colors, cmap):
    """ Generate a .png sprite of a stat map, or encode it as base64 string.
        Returns: stat_map_base64, if output_sprite is None
    """
    data = _safe_get_data(stat_map_img, ensure_finite=True)
    mask = _safe_get_data(mask_img, ensure_finite=True)
    if output_sprite is None :
        stat_map_sprite = BytesIO()
        _save_sprite(data, stat_map_sprite, colors['vmax'], colors['vmin'],
                    mask, cmap, 'png')
        return _bytesIO_to_base64(stat_map_sprite)
    else :
        _save_sprite(data, stat_map_sprite, colors['vmax'], colors['vmin'],
                    mask, cmap, 'png')
        return ""


def _output_sprite_bg(output_sprite_bg, bg_img, bg_min, bg_max):
    """ Save a .png sprite of a background image, or encode it as base64 string.
        Returns: bg_base64, if output_sprite_bg is None
    """
    bg_data = _safe_get_data(bg_img, ensure_finite=True)
    if output_sprite_bg is None :
        output_sprite_bg = BytesIO()
        _save_sprite(bg_data, output_sprite_bg, bg_max, bg_min, None, 'gray',
                    'png')
        return _bytesIO_to_base64(bg_sprite)
    else :
        _save_sprite(bg_data, output_sprite_bg, bg_max, bg_min, None, 'gray',
                    'png')
        return ""


def _output_cm(output_cm, colors):
    """ Save a .png image of a colormap, or encode it as base64 string.
        Returns: cm_base64, if output_cm is None
    """
    if output_cm is None :
        stat_map_cm = BytesIO()
        _save_cm(stat_map_cm, colors['cmap'], 'png')
        return _bytesIO_to_base64(stat_map_cm)
    else :
        _save_cm(stat_map_cm, colors['cmap'], 'png')
        return ""


def _view_html(bg_img, stat_map_img, mask_img, output_sprite, output_sprite_bg,
              output_cm, id_viewer, id_sprite, id_sprite_bg, id_cm, bg_min,
              bg_max, colors, cmap, colorbar):
    """ Create a html snippet for a brainsprite viewer.
        Retuns: view_html
    """
    # Load and initialize snippet
    html_view = _get_text_data('snippet_view.html')
    view_html.safe_substitute("3Dviewer", id_viewer)

    # Add the stat map data
    stat_map_base64 = _output_sprite(output_sprite, stat_map_img, mask_img,
                                    colors, cmap)
    if output_sprite is None :
        view_html.safe_substitute('$stat_map', 'data:image/png;base64,'
                                 + stat_map_base64)
    else :
        view_html.safe_substitute('$stat_map', output_sprite)

    # Add the background image
    bg_base64 = _output_sprite_bg(output_sprite_bg, bg_img, bg_min, bg_max)
    if output_sprite_bg is None :
        view_html.safe_substitute('$bg', 'data:image/png;base64,' + bg_base64)
    else :
        view_html.safe_substitute('$bg', output_sprite_bg)

    # Add the colormap
    cm_base64 = _output_cm(output_cm, colors)
    if output_cm is None :
        view_html.safe_substitute('$cm', 'data:image/png;base64,' + cm_base64)
    else :
        view_html.safe_substitute('$cm', output_cm)


def sprite_data(stat_map_img,
             bg_img='MNI152',
             output_sprite=None,
             output_sprite_bg=None,
             output_cm=None,
             id_viewer="brainViewer",
             id_sprite='spriteImg',
             id_sprite_bg='spriteBackground',
             id_cm='colormap',
             cut_coords=None,
             flag_value=True,
             colorbar=True,
             title=None,
             onclick=None,
             threshold=1e-6,
             annotate=True,
             draw_cross=True,
             black_bg='auto',
             black_bg='auto',
             cmap=cm.cold_hot,
             symmetric_cmap=True,
             dim='auto',
             vmax=None,
             vmin=None,
             resampling_interpolation='continuous',
             opacity=1,
             **kwargs
             ):
    """
    Generate js and html snippets, as well as sprites, for a brainsprite viewer

    Parameters
    ----------
    stat_map_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The statistical map image. Can be either a 3D volume or a 4D volume
        with exactly one time point.
    bg_img : Niimg-like object (default='MNI152')
        See http://nilearn.github.io/manipulating_images/input_output.html
        The background image that the stat map will be plotted on top of.
        If nothing is specified, the MNI152 template will be used.
        To turn off background image, just pass "bg_img=False".
    output_sprite : string (default=None)
        Save the sprite in the specified (.png) file.
    output_sprite : string (default=None)
        Save the sprite for the background image in the specified (.png) file.
    output_cm : string (default=None)
        Save the colormap in the specified (.png) file.
    id_viewer : string (default='brainViewer')
        The ID of the canvas that holds the brain viewer.
    id_sprite : string (default='spriteImg')
        The ID used for the sprite of the statistical map image.
    id_sprite_bg : string (default='spriteBackground')
        The ID used for the sprite of the background image.
    id_cm : string (default='colormap')
        The ID used for the colormap image.
    cut_coords : None, or a tuple of floats (default None)
        The MNI coordinates of the point where the cut is performed
        as a 3-tuple: (x, y, z). If None is given, the cuts are calculated
        automaticaly.
    flag_value : boolean, optional (default True)
        turn on/off display of the current value.
    colorbar : boolean, optional (default True)
        If True, display a colorbar on top of the plots.
    title : string or None (default=None)
        The title displayed on the figure (or None: no title).
    onclick : string or None (default=None)
        Javascript command to call on click.
    threshold : string, number or None  (default=1e-6)
        If None is given, the image is not thresholded.
        If a string of the form "90%" is given, use the 90-th percentile of
        the absolute value in the image.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        automatically.
    annotate : boolean (default=True)
        If annotate is True, current cuts and value of the map are added to the
        viewer.
    draw_cross : boolean (default=True)
        If draw_cross is True, a cross is drawn on the plot to
        indicate the cuts.
    black_bg : boolean (default='auto')
        If True, the background of the image is set to be black.
        Otherwise, a white background is used.
        If set to auto, an educated guess is made to find if the background
        is white or black.
    cmap : matplotlib colormap, optional
        The colormap for specified image.
    symmetric_cmap : bool, optional (default=True)
        True: make colormap symmetric (ranging from -vmax to vmax).
        False: the colormap will go from the minimum of the volume to vmax.
        Set it to False if you are plotting a positive volume, e.g. an atlas
        or an anatomical image.
    dim : float, 'auto' (default='auto')
        Dimming factor applied to background image. By default, automatic
        heuristics are applied based upon the background image intensity.
        Accepted float values, where a typical scan is between -2 and 2
        (-2 = increase constrast; 2 = decrease contrast), but larger values
        can be used for a more pronounced effect. 0 means no dimming.
    vmax : float, or None (default=None)
        max value for mapping colors.
        If vmax is None and symmetric_cmap is True, vmax is the max
        absolute value of the volume.
        If vmax is None and symmetric_cmap is False, vmax is the max
        value of the volume.
    vmin : float, or None (default=None)
        min value for mapping colors.
        If `symmetric_cmap` is `True`, `vmin` is always equal to `-vmax` and
        cannot be chosen.
        If `symmetric_cmap` is `False`, `vmin` defaults to the min of the
        image, or 0 when a threshold is used.
    resampling_interpolation : string, optional (default continuous)
        The interpolation method for resampling.
        Can be 'continuous', 'linear', or 'nearest'.
        See nilearn.image.resample_img
    opacity : float in [0,1] (default 1)
        The level of opacity of the overlay (0: transparent, 1: opaque)

    Returns
    -------
    view_js : a javascript snippet of the brainsprite viewer.
      This snippet needs to be added at the end of the html document.
    view_html : a html snippet of the brainsprite data.
      This snippet needs to be inserted in a <div> of the html page.
      If some outputs are not generated (output_*), then they are added as
      BASE64 encoded images in the html snippet.
    """

    # Prepare the color map and thresholding
    mask_img, stat_map_img, data, threshold = _mask_stat_map(
        stat_map_img, threshold)
    colors = colorscale(cmap, data.ravel(), threshold=threshold,
                        symmetric_cmap=symmetric_cmap, vmax=vmax,
                        vmin=vmin)

    # Prepare the data for the cuts
    bg_img, bg_min, bg_max, black_bg = _load_bg_img(stat_map_img, bg_img,
                                                    black_bg, dim)
    stat_map_img, mask_img = _resample_stat_map(stat_map_img, bg_img, mask_img,
                                                resampling_interpolation)
    cut_slices = _get_cut_slices(stat_map_img, cut_coords, threshold)

    # Now create a json-like object for the viewer, and converts in html
    json_data = _json_view_data(bg_img, stat_map_img, mask_img, bg_min, bg_max,
                                colors, cmap, colorbar)

    json_view['params'] = _json_view_params(
        stat_map_img.shape, stat_map_img.affine, colors['vmin'],
        colors['vmax'], cut_slices, black_bg, opacity, draw_cross, annotate,
        title, colorbar, value=False)
    html_view = _json_view_to_html(json_view)

    return html_view
