import yaml
from yaml import CLoader as Loader
from varname import nameof
from functools import partial
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_params_yaml(file, Loader=Loader):
    '''Load in input parameters.
    Load a text file containing
    all the parameters needed.

    These will be passed through
    the functions.

    Args:
        File (string): Filepath of parameters file.
        load_mode (string): safe_load, load
    Returns:
        ts_parameter_dict: Dictionary of input parameters.
    '''
    with open(file, 'r') as f:
        ts_parameter_dict = yaml.load(f, Loader)
    return ts_parameter_dict


def default_ts_log_names_dictionary():
    default_ts_log_names = {'ts_sf': 'TS_SF',
                            'ts_xdn': 'TS_XDN',
                            'ts_xsn': 'TS_XSN',
                            'ts_phit_sand_macro': 'TS_PHIT_SAND_MACRO',
                            'ts_phit_sand_micro': 'TS_PHIT_SAND_MICRO',
                            'ts_phit_sand_total': 'TS_PHIT_SAND_TOTAL',
                            'ts_vsh_wet_lam': 'TS_VSH_WET_LAM',
                            'ts_vsh_wet_disp': 'TS_VSH_WET_DISP',
                            'ts_vsh_wet_struct': 'TS_VSH_WET_STRUCT',
                            'ts_vsh_wet_total': 'TS_VSH_WET_TOTAL',
                            'ts_vsh_dry_lam': 'TS_VSH_DRY_LAM',
                            'ts_vsh_dry_disp': 'TS_VSH_DRY_DISP',
                            'ts_vsh_dry_struct': 'TS_VSH_DRY_STRUCT',
                            'ts_vsh_dry_total': 'TS_VSH_DRY_TOTAL',
                            'ts_other': 'TS_OTHER'}
    return default_ts_log_names


def default_ts_colors_dictionary():
    default_ts_colors = {'TS_SF': 'brown',
                         'TS_XDN': 'orange',
                         'TS_XSN': 'brown',
                         'TS_PHIT_SAND_MACRO': 'orange,',
                         'TS_PHIT_SAND_MICRO': 'black,',
                         'TS_PHIT_SAND_TOTAL': 'purple,',
                         'TS_VSH_WET_LAM': 'skyblue',
                         'TS_VSH_WET_DISP': 'aqua',
                         'TS_VSH_WET_STRUCT': 'steelblue',
                         'TS_VSH_WET_TOTAL': 'navy',
                         'TS_VSH_DRY_LAM': 'yellowgreen',
                         'TS_VSH_DRY_DISP': 'olivedrab',
                         'TS_VSH_DRY_STRUCT': 'teal',
                         'TS_VSH_DRY_TOTAL': 'darkgreen',
                         'TS_OTHER': 'lightgrey'}
    return default_ts_colors


def default_log_settings_dictionary():
    default_log_settings = {gr_col: (0.0, 150.0, 'green'),
                            phit_col: (0.0, 0.4, 'red')}
    return default_log_settings


def check_whether_expected_parameters_are_created(ts_parameter_dict):
    '''Checks whether expected parameters have been created (through the
    YAML config file). If not, these are created here with default values).
    As a minimum, parameters 
        gr_col: name of GR column
        phit_col: name of PHIT column 
    ... need to be defined. If this is not done through the YAML file, please 
    ensure that these variables exist before running the analytical code.
    For the semantical and cosmetical parameters, defaults will be used when 
    none are found in the YAML file.

    The YAML file is expected to have the following structure:
        ts_parameters:*
            gr_clean_sand: value*
            gr_lam: value*
            phit_clean_sand: value*
            phit_lam: value*
            (structural_endpoint: paper/alternative)
        col_names:*
            gr_col: value*
            phit_col: value*
            (md_col: value - optional, only used for plotting)
        log_settings:
            *gr_col: !!python/tuple [min_scale, max_scale, color]
            *phit_col: !!python/tuple [min_scale, max_scale, color]
        ts_log_names:
             ts_sf: &ts_sf TS_SF
             ts_xdn: &ts_xdn TS_XDN
             ts_xsn: &ts_xsn TS_XSN
             ts_phit_sand_macro: &ts_phit_sand_macro TS_PHIT_SAND_MACRO
             ts_phit_sand_micro: &ts_phit_sand_micro TS_PHIT_SAND_MICRO
             ts_phit_sand_total: &ts_phit_sand_total TS_PHIT_SAND_TOTAL             
             ts_vsh_wet_lam: &ts_vsh_wet_lam TS_VSH_WET_LAM
             ts_vsh_wet_disp: &ts_vsh_wet_disp TS_VSH_WET_DISP
             ts_vsh_wet_struct: &ts_vsh_wet_struct TS_VSH_WET_STRUCT
             ts_vsh_wet_total: &ts_vsh_wet_total TS_VSH_WET_TOTAL
             ts_vsh_dry_lam: &ts_vsh_dry_lam TS_VSH_DRY_LAM
             ts_vsh_dry_disp: &ts_vsh_dry_disp TS_VSH_DRY_DISP
             ts_vsh_dry_struct: &ts_vsh_dry_struct TS_VSH_DRY_STRUCT
             ts_vsh_dry_total: &ts_vsh_dry_total TS_VSH_DRY_TOTAL
             ts_other: &ts_other TS_OTHER
        ts_colors:
             *ts_sf: !!python/str brown
             *ts_xdn: !!python/str orange
             *ts_xsn: !!python/str brown
             *ts_vsh_wet_lam: !!python/str skyblue
             *ts_vsh_wet_disp: !!python/str aqua
             *ts_vsh_wet_struct: !!python/str steelblue
             *ts_vsh_wet_total: !!python/str navy
             *ts_vsh_dry_lam: !!python/str yellowgreen
             *ts_vsh_dry_disp: !!python/str olivedrab
             *ts_vsh_dry_struct: !!python/str teal
             *ts_vsh_dry_total: !!python/str darkgreen
             *ts_other: !!python/str lightgrey
    (Note that an "&" marks an alias for the node and the * references the aliased
    node with that name)

    Args:
        ts_params: dictionary containing the content/parameters as read from the
        YAML file (derived from function "load_params_yaml").'''
    
    msg = f'Could not find a dictionary "col_names" in dictionary {nameof(ts_params)}'
    assert 'col_names' in ts_params, msg
    globals().update(ts_params['col_names'])
    msg = f'"gr_col" and/or "phit_col" is missing in dictionary "col_names"'
    assert set(['gr_col', 'phit_col']) < set(list(ts_params['col_names'])), msg


    msg = f'Could not find a dictionary "ts_parameters" in dictionary {nameof(ts_params)}'
    try:
        globals().update(ts_params['ts_parameters'])
    except:
        print(f'Could not load parameters from dictionary "ts_parameters"')
        pass

    try:
        gr_clean_sand
    except Exception as e:
        print(e, '- make sure this input is available before running the analysis!')

    try:
        gr_lam
    except Exception as e:
        print(e, '- make sure this input is available before running the analysis!')

    try:
        phit_clean_sand
    except Exception as e:
        print(e, '- make sure this input is available before running the analysis!')

    try:
        phit_lam
    except Exception as e:
        print(e, '- make sure this input is available before running the analysis!')


    default_ts_log_names = default_ts_log_names_dictionary()
    try:
        'ts_log_names' in ts_params
        for k in default_ts_log_names.keys():
            if k not in ts_params['ts_log_names']:
                print(f'log name "{k}" missing from log names dictionary "ts_log_names". Adding default name ("{default_ts_log_names[k]}")')
                ts_params['ts_log_names'][k] = default_ts_log_names[k]
        globals().update(ts_params['ts_log_names'])
    except:
        print(f'No user log names found: using default log names instead.')
        print(f'{default_ts_log_names}')
        globals().update(default_ts_log_names)


    default_ts_colors = default_ts_colors_dictionary()
    try:
        'ts_colors' in ts_params
        for k in default_ts_colors.keys():
            if k not in ts_params['ts_colors']:
                print(f'log name "{k}" missing from log names dictionary "ts_colors". Adding default name ("{default_ts_log_names[k]}")')
                ts_params['ts_colors'][k] = default_ts_colors[k]
        globals()['ts_colors'] = ts_params['ts_colors']
    except:
        print(f'No user preferences for colors found: using default colors instead.')
        print(f'{default_ts_colors}')
        globals()['ts_colors'] = default_ts_colors


    default_log_settings = default_log_settings_dictionary()
    try:
        'log_settings' in ts_params
        for k in default_log_settings.keys():
            if k not in ts_params['log_settings']:
                ts_params['log_settings'][k] = default_ts_colors[k]
        globals()['log_settings'] = ts_params['log_settings']
    except:
        print(f'No user preferences for log settings found: using defaults instead.')
        print(f'{default_log_settings}')
        globals()['log_settings'] = default_log_settings


def calc_disp_shale_endpoint(gr_clean_sand, gr_lam, phit_clean_sand, phit_lam):
    '''Calculate the porosity and GR of the dispersed shale endpoint.

    Args:
        gr_clean_sand: GR of the clean sand endpoint
        gr_lam: GR of the laminated shale endpoint
        phit_clean_sand: (total) porosity of the clean sand enpoint
        phit_lam: (total) porosity of the (laminated) shale endpoint
    Returns:
        a tuple with the GR and (total) porosity of the dispersed endpoint.
    '''
    gr_disp = gr_clean_sand + phit_clean_sand*gr_lam
    phit_disp = phit_lam*phit_clean_sand
    return (gr_disp, phit_disp)


def calc_struct_shale_endpoint(gr_clean_sand, gr_lam, phit_clean_sand, phit_lam):
    '''Calculate the porosity and GR of the structural shale endpoint.
    
    Args:
        gr_clean_sand: GR of the clean sand endpoint
        gr_lam: GR of the laminated shale endpoint
        phit_clean_sand: (total) porosity of the clean sand enpoint
        phit_lam: (total) porosity of the (laminated) shale endpoint
    Returns:
        a tuple with the GR and (total) porosity of the structural endpoint.
    '''
    if  structural_endpoint == 'alternative':
        gr_struct = (1-phit_clean_sand)*gr_lam
    else: # calculate as per paper
        gr_struct = gr_clean_sand*phit_clean_sand + gr_lam*(1-phit_clean_sand)
    phit_struct = phit_clean_sand + (1-phit_clean_sand)*phit_lam
    return (gr_struct, phit_struct)


def calc_gamma(gr, gr_clean_sand, gr_lam):
    '''Calculates the gamma (index) used in T-S calculations*.
    Gamma = (gr_lam-gr)/(gr_lam-gr_clean_sand)
    (* note that gamma = 0 fo rthe laminated shale endpoint, whilst
    the gamma for the clean sand endpoint = 1)

    Args:
        gr: GR to convert to gamma-index
        gr_clean_sand: GR of the clean sand endpoint
        gr_lam: GR of the (laminated) shale endpoint
    Returns:
        gamma
    '''
    gamma = (gr_lam-gr)/(gr_lam-gr_clean_sand)
    return gamma


def calc_gr(gamma, gr_clean_sand, gr_lam):
    '''Calculates the GR from gamma (index).
    
    Args:
        gamma: gamma (index) to convert to GR
        gr_clean_sand: GR of the clean sand endpoint
        gr_lam: GR of the (laminated) shale endpoint
    Returns:
        GR value    
    '''
    gr = gr_lam-gamma*(gr_lam-gr_clean_sand)
    return gr


def get_clean_sand_laminated_shale_line_parameters(gr_clean_sand, gr_lam, phit_clean_sand, phit_lam):
    '''Calculates the equation of the clean sand-laminated shale line.
    
    Args:
        gr_clean_sand: GR of the clean sand endpoint
        gr_lam: GR of the laminated shale endpoint
        phit_clean_sand: (total) porosity of the clean sand enpoint
        phit_lam: (total) porosity of the (laminated) shale endpoint
    Returns:
        a tuples with (gradient, intercept) of the clean sand-laminated
        shale line.
    '''
    gamma_clean_sand = calc_gamma(gr_clean_sand, gr_clean_sand, gr_lam)
    gamma_lam = calc_gamma(gr_lam, gr_clean_sand, gr_lam)
    
    gradient = (phit_lam-phit_clean_sand)/(gamma_lam-gamma_clean_sand)
    intercept = phit_lam
    return (gradient, intercept)


def get_laminated_shale_dispersed_shale_line_parameters(gr_clean_sand, gr_lam, phit_clean_sand, phit_lam):
    '''Calculates the equation of the laminated-dispersed shale line.

    Args:
        gr_clean_sand: GR of the clean sand endpoint
        gr_lam: GR of the laminated shale endpoint
        phit_clean_sand: (total) porosity of the clean sand enpoint
        phit_lam: (total) porosity of the (laminated) shale endpoint
    Returns:
        a tuples with (gradient, intercept) of the laminated shale-dispersed
        shale line.
    '''
    gr_disp, phit_disp = calc_disp_shale_endpoint(gr_clean_sand, gr_lam, phit_clean_sand, phit_lam)
    gamma_lam = calc_gamma(gr_lam, gr_clean_sand, gr_lam)
    gamma_disp = calc_gamma(gr_disp, gr_clean_sand, gr_lam)

    gradient = (phit_lam-phit_disp)/(gamma_lam-gamma_disp)
    intercept = phit_lam
    return (gradient, intercept)


def get_clean_sand_dispersed_shale_line_parameters(gr_clean_sand, gr_lam, phit_clean_sand, phit_lam):
    '''Calculates the equation of the clean sand-dispersed shale line.

    Args:
        gr_clean_sand: GR of the clean sand endpoint
        gr_lam: GR of the laminated shale endpoint
        phit_clean_sand: (total) porosity of the clean sand enpoint
        phit_lam: (total) porosity of the (laminated) shale endpoint
    Returns:
        a tuples with (gradient, intercept) of the clean sand-dispersed shale line.
    '''
    gr_disp, phit_disp = calc_disp_shale_endpoint(gr_clean_sand, gr_lam, phit_clean_sand, phit_lam)
    gamma_clean_sand = calc_gamma(gr_clean_sand, gr_clean_sand, gr_lam)
    gamma_disp = calc_gamma(gr_disp, gr_clean_sand, gr_lam)

    gradient = (phit_disp-phit_clean_sand)/(gamma_disp-gamma_clean_sand)
    intercept = phit_disp-gradient*gamma_disp
    return (gradient, intercept)


def get_clean_sand_structural_shale_line_parameters(gr_clean_sand, gr_lam, phit_clean_sand, phit_lam):
    '''Calculates the equation of the clean sand-structural shale line.

    Args:
        gr_clean_sand: GR of the clean sand endpoint
        gr_lam: GR of the laminated shale endpoint
        phit_clean_sand: (total) porosity of the clean sand enpoint
        phit_lam: (total) porosity of the (laminated) shale endpoint
    Returns:
        a tuples with (gradient, intercept) of the clean sand-structural shale line.
    '''
    gr_struct, phit_struct = calc_struct_shale_endpoint(gr_clean_sand, gr_lam, phit_clean_sand, phit_lam)
    gamma_clean_sand = calc_gamma(gr_clean_sand, gr_clean_sand, gr_lam)
    gamma_struct = calc_gamma(gr_struct, gr_clean_sand, gr_lam)

    gradient = (phit_struct-phit_clean_sand)/(gamma_struct-gamma_clean_sand)
    intercept = phit_struct-gradient*gamma_struct
    return (gradient, intercept)


def get_laminated_shale_structural_shale_line_parameters(gr_clean_sand, gr_lam, phit_clean_sand, phit_lam):
    '''Calculates the equation of the laminated shale-structural shale line.

    Args:
        gr_clean_sand: GR of the clean sand endpoint
        gr_lam: GR of the laminated shale endpoint
        phit_clean_sand: (total) porosity of the clean sand enpoint
        phit_lam: (total) porosity of the (laminated) shale endpoint
    Returns:
        a tuples with (gradient, intercept) of the laminated shale-dispersed
        shale line.
    '''
    gr_struct, phit_struct = calc_struct_shale_endpoint(gr_clean_sand, gr_lam, phit_clean_sand, phit_lam)
    gamma_lam = calc_gamma(gr_lam, gr_clean_sand, gr_lam)
    gamma_struct = calc_gamma(gr_struct, gr_clean_sand, gr_lam)

    gradient = (phit_lam-phit_struct)/(gamma_lam-gamma_struct)
    intercept = phit_lam
    return (gradient, intercept)


def is_in_dispersed_envelope(gr, phit, gr_clean_sand, gr_lam, phit_clean_sand, phit_lam):
    '''Finds whether a point plots within (or on te edge) of the clean sand-laminar-dispersed shale
    envelope or not.
    Args:
        gr: GR value of the point
        por: porosity value of the point
        gr_clean_sand: GR of the clean sand endpoint
        gr_lam: GR of the laminated shale endpoint
        phit_clean_sand: (total) porosity of the clean sand enpoint
        phit_lam: (total) porosity of the (laminated) shale endpoint
    Returns:
        True (in/on the envelope) or False (outside the envelope)
    '''
    n = 4
    grad_cls_lam_line, intc_cls_la_line = get_clean_sand_laminated_shale_line_parameters(gr_clean_sand, \
                                                                                         gr_lam, \
                                                                                         phit_clean_sand,\
                                                                                         phit_lam)
    grad_lam_disp_line, intc_lam_disp_line = get_laminated_shale_dispersed_shale_line_parameters(gr_clean_sand, \
                                                                                                 gr_lam, \
                                                                                                 phit_clean_sand,\
                                                                                                 phit_lam)
    grad_cls_disp_line, intc_cls_disp_line = get_clean_sand_dispersed_shale_line_parameters(gr_clean_sand, \
                                                                                             gr_lam, \
                                                                                             phit_clean_sand,\
                                                                                             phit_lam)
    gamma = calc_gamma(gr, gr_clean_sand, gr_lam)

    if (np.round(phit, n) <= np.round(grad_cls_lam_line*gamma+intc_cls_la_line, n)) \
        and (np.round(phit, n) >= np.round(grad_lam_disp_line*gamma+intc_lam_disp_line, n)) \
        and (np.round(phit, n) >= np.round(grad_cls_disp_line*gamma+intc_cls_disp_line, n)):
        return True
    else:
        return False


def is_in_structural_envelope(gr, phit, gr_clean_sand, gr_lam, phit_clean_sand, phit_lam):
    '''
    Finds whether a point plots within (or on te edge) of the clean sand-laminar-structural shale
    envelope or not.
    Args:
        gr: GR value of the point
        por: porosity value of the point
        gr_clean_sand: GR of the clean sand endpoint
        gr_lam: GR of the laminated shale endpoint
        phit_clean_sand: (total) porosity of the clean sand enpoint
        phit_lam: (total) porosity of the (laminated) shale endpoint
    Returns:
        True (in/on the envelope) or False (outside the envelope)
    '''
    n = 4
    grad_cls_lam_line, intc_cls_la_line = get_clean_sand_laminated_shale_line_parameters(gr_clean_sand, \
                                                                                       gr_lam, \
                                                                                       phit_clean_sand,\
                                                                                       phit_lam)
    grad_cls_struct_line, intc_cls_struct_line = get_clean_sand_structural_shale_line_parameters(gr_clean_sand, \
                                                                                                 gr_lam, \
                                                                                                 phit_clean_sand,\
                                                                                                 phit_lam)
    grad_lam_struct_line, intc_lam_struct_line = get_laminated_shale_structural_shale_line_parameters(gr_clean_sand, \
                                                                                                      gr_lam, \
                                                                                                      phit_clean_sand,\
                                                                                                      phit_lam)
    gamma = calc_gamma(gr, gr_clean_sand, gr_lam)

    if (np.round(phit, n) > np.round(grad_cls_lam_line*gamma+intc_cls_la_line, n)) \
            and (np.round(phit, n) <= np.round(grad_cls_struct_line*gamma+intc_cls_struct_line, n)) \
            and (np.round(phit, n) <= np.round(grad_lam_struct_line*gamma+intc_lam_struct_line, n)):
        return True
    else:
        return False


def calc_sf_xdn_xsn(gr, phit, gr_clean_sand, gr_lam, phit_clean_sand, phit_lam):
    '''Calculates the sand fraction and dispersed shale for a point (defined by GR and por)
    Args:
        gr: GR for the point
        phit: (total) porosity for the point
        gr_clean_sand: GR of the clean sand endpoint
        gr_lam: GR of the (laminated) shale endpoint
        phit_clean_sand: (total) porosity of the clean sand enpoint
        phit_lam: (total) porosity of the (laminated) shale endpoint
    Returns:
        a tupple with (sf, xdn, xsn), in which:
        - sf = sand fraction (i.e. fraction of bulk that is sand laminae)
        - xdn = normalized fraction of dispersed shale (i.e. in fraction of pore space)
        - xsn = normalized fraction of structural shale (i.e. in fraction of matrix)
    '''
    gamma = calc_gamma(gr, gr_clean_sand, gr_lam)

    grad_cls_lam_line, intc_cls_lam_line = get_clean_sand_laminated_shale_line_parameters(gr_clean_sand, \
                                                                                         gr_lam, \
                                                                                         phit_clean_sand,\
                                                                                         phit_lam)
    grad_lam_disp_line, _ = get_laminated_shale_dispersed_shale_line_parameters(gr_clean_sand, \
                                                                                                 gr_lam, \
                                                                                                 phit_clean_sand,\
                                                                                                 phit_lam)
    grad_cls_disp_line, intc_cls_disp_line = get_clean_sand_dispersed_shale_line_parameters(gr_clean_sand, \
                                                                                             gr_lam, \
                                                                                             phit_clean_sand,\
                                                                                             phit_lam)
    grad_cls_struct_line, intc_cls_struct_line = get_clean_sand_structural_shale_line_parameters(gr_clean_sand, \
                                                                                                 gr_lam, \
                                                                                                 phit_clean_sand,\
                                                                                                 phit_lam)

    if is_in_dispersed_envelope(gr, phit, gr_clean_sand, gr_lam, phit_clean_sand, phit_lam):    
        gradient = grad_cls_disp_line
        intercept = phit-gradient*gamma
        sf = (intercept-intc_cls_lam_line)/(grad_cls_lam_line-gradient)
        
        gr_disp, _ = calc_disp_shale_endpoint(gr_clean_sand, gr_lam, phit_clean_sand, phit_lam)
        gamma_disp = calc_gamma(gr_disp, gr_clean_sand, gr_lam)

        gradient = (phit_lam-phit)/(0 - gamma)
        intercept = phit_lam
        xdn = (1-(intercept-intc_cls_disp_line)/(grad_cls_disp_line-gradient))/(1-gamma_disp)
        xsn = 0
        phit_sand_macro = phit_clean_sand*(1-xdn)
        phit_sand_micro = xdn*phit_clean_sand*phit_lam
        phit_sand_total = phit_sand_macro + phit_sand_micro

    elif is_in_structural_envelope(gr, phit, gr_clean_sand, gr_lam, phit_clean_sand, phit_lam):
        gradient = grad_lam_disp_line
        intercept = phit-gradient*gamma
        sf = (intercept-intc_cls_lam_line)/(grad_cls_lam_line-gradient)

        gr_struct, _ = calc_struct_shale_endpoint(gr_clean_sand, gr_lam, phit_clean_sand, phit_lam)
        gamma_struct = calc_gamma(gr_struct, gr_clean_sand, gr_lam)

        gradient = (phit_lam-phit)/(0-gamma)
        intercept = phit_lam
        xdn = 0
        xsn = (1-(intercept-intc_cls_struct_line)/(grad_cls_struct_line-gradient))/(1-gamma_struct)
        phit_sand_macro = phit_clean_sand
        phit_sand_micro = xsn*(1-phit_clean_sand)*phit_lam
        phit_sand_total = phit_sand_macro + phit_sand_micro

    else:
        sf = np.nan
        xdn = np.nan
        xsn = np.nan
        phit_sand_macro = np.nan
        phit_sand_micro = np.nan
        phit_sand_total = np.nan
    
    return (sf, xdn, xsn, phit_sand_macro, phit_sand_micro, phit_sand_total)


def calc_vsh(phit_clean_sand, phit_lam, sf, xdn, xsn):
    '''Calculates Vshale for each shale type - both in the wet and dry domain.
    Args:
        phit_clean_sand: (total) porosity of the clean sand enpoint
        phit_lam: (total) porosity of the (laminated) shale endpoint
        sf: sand fraction (i.e. fraction of bulk that is sand laminae)
        xdn: normalized fraction of dispersed shale (i.e. in fraction of pore space)
        xsn: normalized fraction of structural shale (i.e. in fraction of matrix)
    Returns
        vsh_wet_lam: wet laminated shale volume (in fraction bulk volume)
        vsh_wet_disp: wet dispersed shale volume (in fraction bulk volume)
        vsh_wet_struct: wet structural shale volume (in fraction bulk volume)
        vsh_wet_total: total wet shale volume (in fraction bulk volume)
        vsh_dry_lam: dry laminated shale volume (in fraction bulk volume)
        vsh_dry_disp: dry dispersed shale volume (in fraction bulk volume)
        vsh_dry_struct dry structural shale volume (in fraction bulk volume)
        vsh_dry_total: total dry shale volume (in fraction bulk volume)
    '''
    vsh_wet_lam = (1-sf)
    vsh_wet_disp = phit_clean_sand*xdn
    vsh_wet_struct = (1-phit_clean_sand)*xsn
    vsh_wet_total = (0 if math.isnan(vsh_wet_lam) else vsh_wet_lam) + \
                    (0 if math.isnan(vsh_wet_disp) else vsh_wet_disp) + \
                    (0 if math.isnan(vsh_wet_struct) else vsh_wet_struct) 
    
    vsh_dry_lam = vsh_wet_lam*(1-phit_lam)
    vsh_dry_disp = vsh_wet_disp*(1-phit_lam)
    vsh_dry_struct = vsh_wet_struct*(1-phit_lam)
    vsh_dry_total = (0 if math.isnan(vsh_dry_lam) else vsh_dry_lam) + \
                    (0 if math.isnan(vsh_dry_disp) else vsh_dry_disp) + \
                    (0 if math.isnan(vsh_dry_struct) else vsh_dry_struct) 

    return (vsh_wet_lam, vsh_wet_disp, vsh_wet_struct, vsh_wet_total, vsh_dry_lam, vsh_dry_disp, vsh_dry_struct, vsh_dry_total)


def get_all_endpoint_values(gr_clean_sand, gr_lam, phit_clean_sand, phit_lam, verbose=False):
    '''
    Lists the "coordinates" (GR and phit) of all envelope endpoints 
    Args:
        gr_clean_sand: GR of the zlean sand endpoint
        gr_lam: GR of the laminated shale endpoint
        phit_clean_sand: (total) porosity of the clean sand enpoint
        phit_lam: (total) porosity of the (laminated) shale endpoint
        verbose: prints also the names of the 
    Returns a tuple with (phit, GR and gamma) for each of the endpoints:
        - clean sand
        - laminated shale
        - dispersed shale
        - structural shale
    '''
    gamma_clean_sand = calc_gamma(gr_clean_sand, gr_clean_sand, gr_lam)
    gamma_lam = calc_gamma(gr_lam, gr_clean_sand, gr_lam)
    gr_disp, phit_disp = calc_disp_shale_endpoint(gr_clean_sand, gr_lam, phit_clean_sand, phit_lam)
    gamma_disp = calc_gamma(gr_disp, gr_clean_sand, gr_lam)
    gr_struct, phit_struct = calc_struct_shale_endpoint(gr_clean_sand, gr_lam, phit_clean_sand, phit_lam)
    gamma_struct= calc_gamma(gr_struct, gr_clean_sand, gr_lam)

    return ((phit_clean_sand, gr_clean_sand, gamma_clean_sand), (phit_lam, gr_lam, gamma_lam), (phit_disp, gr_disp, gamma_disp), (phit_struct, gr_struct, gamma_struct))


def ts_xplot(grs, phits, gr_clean_sand, gr_lam, phit_clean_sand, phit_lam, **kwargs):
    '''Plots data on GR3/phit crossplot and addas the T-S envelopes
    Args:
        grs: array with GR values
        phits: array with phit values
        gr_clean_sand: GR of the zlean sand endpoint
        gr_lam: GR of the laminated shale endpoint
        phit_clean_sand: (total) porosity of the clean sand enpoint
        phit_lam: (total) porosity of the (laminated) shale endpoint
        kwargs:
            plot_gr_axis: adds a secondary axis with GR values (in addition to
                default gamma axis). Default=False
            min_gamma: minimum gamma scale on plot. Default=-0.1
            max_gamma: maximum gamma scale on plot. Default=1.1
            min_phit: minimum phit scale on plot. Default=0
            max_phit: maximum phit scale on plot. Default=0.6
            plot_structural_triangle: default=False
            linewidth_envelope: default=0.5
            show_endpoints: default=False
            endpoint_marker_size: default=5
            data_marker_size: default=4
        '''
    plot_gr_axis = kwargs.get('plot_gr_axis', False)
    min_gamma = kwargs.get('min_gamma', -0.1)
    max_gamma = kwargs.get('max_gamma', 1.1)
    min_phit = kwargs.get('min_phit', 0)
    max_phit = kwargs.get('max_phit', 0.6)
    plot_structural_triangle = kwargs.get('plot_structural_triangle', False)
    linewidth_envelope = kwargs.get('linewidth_envelope', 0.5)
    show_endpoints = kwargs.get('show_endpoints', False)
    endpoint_marker_size = kwargs.get('endpoint_marker_size', 5)
    data_marker_size = kwargs.get('data_marker_size', 4)
    gr_disp, phit_disp = calc_disp_shale_endpoint(gr_clean_sand, gr_lam, phit_clean_sand, phit_lam)
    gr_struct, phit_struct = calc_struct_shale_endpoint(gr_clean_sand, gr_lam, phit_clean_sand, phit_lam)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,7))

    if plot_structural_triangle:
        ax.plot([calc_gamma(gr_struct, gr_clean_sand, gr_lam), \
            calc_gamma(gr_clean_sand, gr_clean_sand, gr_lam)], \
            [phit_struct, phit_clean_sand], color='grey', \
            linewidth=linewidth_envelope)
        ax.plot([calc_gamma(gr_struct, gr_clean_sand, gr_lam), \
            calc_gamma(gr_lam, gr_clean_sand, gr_lam)], \
            [phit_struct, phit_lam], color='grey', \
            linewidth=linewidth_envelope)

    ax.plot([calc_gamma(gr_clean_sand, gr_clean_sand, gr_lam), \
        calc_gamma(gr_lam, gr_clean_sand, gr_lam)], \
        [phit_clean_sand, phit_lam], color='black', \
        linewidth=linewidth_envelope)
    ax.plot([calc_gamma(gr_clean_sand, gr_clean_sand, gr_lam), \
        calc_gamma(gr_disp, gr_clean_sand, gr_lam)], \
        [phit_clean_sand, phit_disp], color='black', \
        linewidth=linewidth_envelope)
    ax.plot([calc_gamma(gr_disp, gr_clean_sand, gr_lam), \
        calc_gamma(gr_lam, gr_clean_sand, gr_lam)], \
        [phit_disp, phit_lam], marker=None, color='black', \
        linewidth=linewidth_envelope)

    gammas = list(map(partial(calc_gamma, gr_clean_sand=gr_clean_sand, gr_lam=gr_lam), grs))
    disp_mask = list(map(partial(is_in_dispersed_envelope, gr_clean_sand=gr_clean_sand, \
        gr_lam=gr_lam, phit_clean_sand=phit_clean_sand, phit_lam=phit_lam), grs, phits))
    struct_mask = list(map(partial(is_in_structural_envelope, gr_clean_sand=gr_clean_sand, \
        gr_lam=gr_lam, phit_clean_sand=phit_clean_sand, phit_lam=phit_lam), grs, phits))
    outside_mask = ~(np.array(disp_mask) | np.array(struct_mask))
    key = 'TS_XDN'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])
    ax.plot(np.where(disp_mask,gammas, np.nan), np.where(disp_mask,phits, np.nan), \
        marker='o', markersize=data_marker_size, color=color, linewidth=0)
    key = 'TS_XSN'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])        
    ax.plot(np.where(struct_mask,gammas, np.nan), np.where(struct_mask,phits, np.nan), \
        marker='o', markersize=data_marker_size, color=color, linewidth=0)
    key = 'TS_OTHER'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])        
    ax.plot(np.where(outside_mask,gammas, np.nan), np.where(outside_mask,phits, np.nan), \
        marker='o', markersize=data_marker_size, color=color, linewidth=0)

    ax.set_xlim([min_gamma, max_gamma])
    ax.set_ylim([min_phit, max_phit])
    ax.set_xlabel(r'$\gamma$', fontweight='bold')
    ax.set_ylabel(r'$\phi_t$', fontweight='bold')
    ax.set_title(f'Thomas-Stieber plot\n(LAM: [GR={gr_lam:.1f}, PHIT={phit_lam:.3f}], CLS: [GR={gr_clean_sand:.1f}, PHIT={phit_clean_sand:.3f}])\n', fontweight='bold')

    if show_endpoints:
        gamma_disp = calc_gamma(gr_disp, gr_clean_sand, gr_lam)
        ax.plot(1, phit_clean_sand, marker='o', markersize=endpoint_marker_size, color='k')
        plt.annotate(f'CLS\n(GR, phit)\n({gr_clean_sand:.1f}, {phit_clean_sand:.3f})', \
            (1, phit_clean_sand), (1.02, phit_clean_sand+0.02), ha='center')
        ax.plot(0, phit_lam, marker='o',
                markersize=endpoint_marker_size, color='k')
        plt.annotate(f'LAM\n(GR, phit)\n({gr_lam:.1f}, {phit_lam:.3f})', \
            (0, phit_lam), (-0.02, phit_lam+0.02), ha='center')
        ax.plot(calc_gamma(gr_disp, gr_clean_sand, gr_lam), phit_disp,
                marker='o', markersize=endpoint_marker_size, color='k')
        plt.annotate(f'DISP\n(GR, phit)\n({gr_disp:.1f}, {phit_disp:.3f})', \
            (gamma_disp, phit_disp), (gamma_disp, phit_disp-0.06), ha='center')
    if plot_structural_triangle:
        gamma_struct = calc_gamma(gr_struct, gr_clean_sand, gr_lam)
        ax.plot(gamma_struct, phit_struct, marker='o', markersize=5, color='k')
        plt.annotate(f'STRUCT\n(GR, phit)\n({gr_struct:.1f}, {phit_struct:.3f})', \
            (gamma_struct, phit_struct), (gamma_struct, phit_struct+0.02), ha='center')

    if plot_gr_axis:
        ax_gr = ax.twiny()
        ax_gr.set_xlim([calc_gr(min_gamma, gr_clean_sand, gr_lam), \
            calc_gr(max_gamma, gr_clean_sand, gr_lam)])
        ax_gr.set_xlabel('GR', fontweight='bold')
    
    plt.show()

    
def ts_cpi_plot(df, **kwargs):
    '''Creates a CPI plot with the Thomas-Stieber results and input data (GR + PHIT)
    Track 1: GR
    Track 2: PHIT 
    Track 3: color indicating whether depth increment falls within "dispersed"
                 or "structural" triangle, or outside either of these envelopes
    Track 4: sand fraction (SF) vs fraction laminated shale
    Track 5: (normalized) fraction of dispersed shale (XDN) and (normalized) 
                 struturcal shale
    Track 6: macro and micro porosity in sand fraction 
    Track 7: rock fractions: macro+micro porosity in SF; dry shale volumes 
                 (dispersed + structural + laminated = total) and matrix
    Args:
        df: pandas DataFrame with GR, PHIT and Thomas-Stieber results
        kwargs:
            figsize (tuple): size of figure. Default is (7, 12)
            alpha (float): amount of transparancy (ref matplotlib). Default is 0.4
            legend_fontsize (float): fontsize of legend (ref matplotlib). Default is 6
            ticks_fontsize (float): fontsize of tick marks (ref matplotlib). Default is 8
            label_fontsize (float): fontsize of x-axis labels T-S results (ref matplotlib). Default is 8
            linewidth (float): linewidth of curves (ref matplotlib). Default is 0.5
            plot_left (float): left margin of plot (ref matplotlib). Default is 0.02
            plot_bottom (float):bottom margin of plot (ref matplotlib). Default is 0.35
            plot_right (float): right margin of plot (ref matplotlib). Default is 0.98
            plot_top (float): top margin of plot (ref matplotlib). Default is 0.95
            plot_wspace (float): width of the padding between subplots (ref matplotlib). Default is 0.15
            plot_hspace (float): height of the padding between subplots (ref matplotlib). Default is 0
    '''
    figsize = kwargs.get('figsize', (7,12))
    alpha = kwargs.get('alpha', 0.4)
    legend_fontsize = kwargs.get('legend_fontsize', 6)
    ticks_fontsize = kwargs.get('ticks_fontsize', 8)
    label_fontsize = kwargs.get('label_fontsize', 8)
    linewidth = kwargs.get('linewidth', 0.5)
    plot_left = kwargs.get('plot_left', 0.02)
    plot_bottom = kwargs.get('plot_bottom', 0.35)
    plot_right = kwargs.get('plot_right', 0.98)
    plot_top = kwargs.get('plot_top', 0.95)
    plot_wspace = kwargs.get('plot_wspace', 0.15)
    plot_hspace = kwargs.get('plot_hspace', 0.)

    fig, ax = plt.subplots(nrows=1, ncols=7, figsize=figsize, sharey=True)

    # GR track
    try:
        color = log_settings[gr_col][2]
    except (NameError, KeyError):
        color = 'green'

    try:
        xlim = (log_settings[gr_col][0], log_settings[gr_col][1])
    except (NameError, KeyError):
        xlim = (0, 150)
    ax[0].plot(df[gr_col], df[md_col], color=color, linewidth=linewidth)
    ax[0].set_xlabel(gr_col, fontweight='bold')
    ax[0].set_ylabel(md_col, fontweight='bold')
    ax[0].set_xlim(xlim)
    ax[0].xaxis.grid(visible=True, which='both', color='linen')
    ax[0].minorticks_on()
    ax[0].invert_yaxis()

    # PHIT track
    try:
        color = log_settings[phit_col][2]
    except (NameError, KeyError):
        color = 'red'

    try:
        xlim = (log_settings[phit_col][0], log_settings[phit_col][1])
    except (NameError, KeyError):
        xlim = (0, 0.4)
    ax[1].plot(df[phit_col], df[md_col], color=color, linewidth=linewidth)
    ax[1].set_xlabel(phit_col, fontweight='bold')
    ax[1].fill_betweenx(df[md_col], xlim[0], df[phit_col], color=color, alpha=alpha)
    ax[1].set_xlim(xlim)
    ax[1].minorticks_on()

    # track with shading for dispersed/structural/None:
    key = 'TS_XSN'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])
    ax[2].fill_betweenx(df[md_col], df[ts_xsn]+2, 0, color=color, linewidth=linewidth, label='strc')
    key = 'TS_XDN'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])
    ax[2].fill_betweenx(df[md_col], df[ts_xdn]+2, 0, color=color, label='disp')
    key = 'TS_OTHER'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])
    ax[2].fill_betweenx(df[md_col], np.where(np.logical_and(df[ts_xsn].isna(),
        df[ts_xdn].isna()), 2, np.nan), 0, color=color, label='n/a')
    xlim = [0, 1]
    ax[2].set_xlim(xlim)
    ax[2].set_xticks(xlim)
    ax[2].legend(loc='best', prop={'size': legend_fontsize})

    # SF track
    key = 'TS_SF'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])
    ax[3].plot(df[ts_sf], df[md_col], color=color, linewidth=linewidth)
    ax[3].set_xlabel('SF (fr.b.v)', fontweight='bold', rotation=90, fontsize=label_fontsize)
    ax[3].fill_betweenx(df[md_col], 0, df[ts_sf], color=color, alpha=alpha)
    ax[3].fill_betweenx(df[md_col], df[ts_sf], 1, color='brown', alpha=alpha)
    key = 'TS_OTHER'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])
    ax[3].fill_betweenx(df[md_col], np.where(np.logical_and(df[ts_xsn].isna(), \
        df[ts_xdn].isna()), 2, np.nan), 0, color=color, label='outside limits')
    xlim = [0, 1]
    ax[3].set_xlim(xlim)
    ax[3].set_xticks(xlim)    

    # XDN, XSN track
    key = 'TS_XDN'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])
    ax[4].plot(df[ts_xdn], df[md_col], color=color, linewidth=linewidth, label='XDN')
    key = 'TS_XSN'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])
    ax[4].plot(df[ts_xsn], df[md_col], color=color, linewidth=linewidth, label='XSN')
    key = 'TS_OTHER'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])
    ax[4].fill_betweenx(df[md_col], np.where(np.logical_and(df[ts_xsn].isna(),
        df[ts_xdn].isna()), 2, np.nan), 0, color=color, label='n/a')    
    ax[4].set_xlabel('XDN/XSN (fr.b.v)', fontweight='bold', rotation=90, fontsize=label_fontsize)
    xlim = [0, 1]
    ax[4].set_xlim(xlim)
    ax[4].set_xticks(xlim)    
    ax[4].legend(loc='best', prop={'size': legend_fontsize})

    # SAND POROSITY
    key = 'TS_PHIT_SAND_MACRO'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])
    ax[5].plot(df[ts_phit_sand_macro], df[md_col], color=color, linewidth=linewidth, label=r'macro $\phi$')
    ax[5].fill_betweenx(df[md_col], 0, df[ts_phit_sand_macro], color=color, alpha=alpha/2)
    key = 'TS_PHIT_SAND_MICRO'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])
    ax[5].plot(df[ts_phit_sand_micro]+df[ts_phit_sand_macro], df[md_col], color=color, linewidth=linewidth, label=r'micro $\phi$')
    ax[5].fill_betweenx(df[md_col], df[ts_phit_sand_macro], df[ts_phit_sand_total], color=color, alpha=alpha)
    key = 'TS_OTHER'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])
    ax[5].fill_betweenx(df[md_col], np.where(np.logical_and(df[ts_xsn].isna(),
        df[ts_xdn].isna()), 2, np.nan), 0, color=color, label='n/a')
    ax[5].set_xlabel('SAND POROSITY (fr.b.v)', fontweight='bold',
                     rotation=90, fontsize=label_fontsize)
    try:
        xlim = (log_settings[phit_col][0], log_settings[phit_col][1])
    except (NameError, KeyError):
        xlim = (0, 0.4)
    ax[5].set_xlim(xlim)
    ax[5].set_xticks(xlim)
    ax[5].legend(loc='best', prop={'size': legend_fontsize})

    # VSH
    phi_sd_macro = df[ts_phit_sand_macro]*df[ts_sf]
    phi_sd_micro = phi_sd_macro + df[ts_phit_sand_micro]*df[ts_sf]
    tot_phi = df[phit_col]
    vshdry_disp = tot_phi + df[ts_vsh_dry_disp]
    vshdry_lam = vshdry_disp + df[ts_vsh_dry_lam]
    vshdry_struc = vshdry_lam + df[ts_vsh_dry_struct]
    key = 'TS_PHIT_SAND_MACRO'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])
    ax[6].plot(phi_sd_macro, df[md_col], color=color, linewidth=linewidth)
    ax[6].fill_betweenx(df[md_col], 0, phi_sd_macro, color=color, alpha=alpha/2, label=r'macro $\phi$')
    key = 'TS_PHIT_SAND_MICRO'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])
    ax[6].plot(phi_sd_micro, df[md_col], color=color, linewidth=linewidth)
    ax[6].fill_betweenx(df[md_col], phi_sd_macro, phi_sd_micro, color=color, alpha=alpha, label=r'micro $\phi$')
    ax[6].plot(tot_phi, df[md_col], color='black', linewidth=linewidth)
    ax[6].fill_betweenx(df[md_col], phi_sd_micro, tot_phi, color='black', alpha=alpha, label='tot phi')
    key = 'TS_VSH_DRY_DISP'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])
    ax[6].plot(vshdry_disp, df[md_col], color=color, linewidth=linewidth)
    ax[6].fill_betweenx(df[md_col], tot_phi, vshdry_disp, color=color, alpha=alpha, label=r'Vshdry$_{disp}$')
    key = 'TS_VSH_DRY_LAM'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])
    ax[6].plot(vshdry_lam, df[md_col], color=color, linewidth=linewidth)
    ax[6].fill_betweenx(df[md_col], vshdry_disp, vshdry_lam, color=color, alpha=alpha, label=r'Vshdry$_{lam}$')
    key = 'TS_VSH_DRY_STRUCT'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])
    ax[6].plot(vshdry_struc, df[md_col], color=color,
               linewidth=linewidth)
    ax[6].fill_betweenx(df[md_col], vshdry_lam, vshdry_struc, color=color, alpha=alpha, label=r'Vshdry$_{struct}$')
    ax[6].fill_betweenx(df[md_col], vshdry_struc, 1, facecolor='yellow',edgecolor='black',hatch='..', alpha=alpha, label='matrix')
    key = 'TS_OTHER'
    color = ts_colors.get(key, default_ts_colors_dictionary()[key])
    ax[6].fill_betweenx(df[md_col], np.where(np.logical_and(df[ts_xsn].isna(),
        df[ts_xdn].isna()), 2, np.nan), 0, color=color, label='n/a')
    ax[6].set_xlabel('ROCK fractions (fr.b.v)', fontweight='bold', rotation=90, fontsize=label_fontsize)
    xlim = [0, 1]
    ax[6].set_xlim(xlim)
    ax[6].set_xticks(xlim)
    ax[6].legend(loc='best', prop={'size': legend_fontsize})

    for i, _ax in enumerate(ax):
        if i > 0:
            _ax.set_yticks([])
            _ax.xaxis.grid(visible=True, which='both', color='linen')
    for i, _ax in enumerate(ax):
            _ax.tick_params(labelsize=ticks_fontsize)
    plt.tight_layout()
    plt.subplots_adjust(left=plot_left, bottom=plot_bottom, right=plot_right, top=plot_top, \
        wspace=plot_wspace, hspace=plot_hspace)
    plt.suptitle('Thomas-Stieber', fontweight='bold')

    plt.show()


if __name__ == '__main__':
    ts_params = load_params_yaml(r'../configs/notebook_configs.yaml')
    check_whether_expected_parameters_are_created(ts_params)
    
    data = pd.read_csv('../data/example_data.csv')
    
    ts = pd.DataFrame(data.apply(lambda x: \
        calc_sf_xdn_xsn(x[gr_col], x[phit_col], gr_clean_sand, gr_lam, phit_clean_sand, phit_lam), \
        axis=1).to_list(), columns=[ts_sf, ts_xdn, ts_xsn, ts_phit_sand_macro, ts_phit_sand_micro, ts_phit_sand_total])
    # or alternatively:
    # ts = pd.DataFrame(list(map(partial(calc_sf_xdn_xsn, gr_clean_sand=gr_clean_sand, \
    # gr_lam=gr_lam, phit_clean_sand=phit_clean_sand, phit_lam=phit_lam),data[gr_col], data[phit_col])), \
    # columns=[ts_sf, ts_xdn, ts_xsn, ts_phit_sand_macro, ts_phit_sand_micro, ts_phit_sand_total])
    
    vsh = pd.DataFrame(ts.apply(lambda  x: calc_vsh(phit_clean_sand, phit_lam, x[ts_sf], \
        x[ts_xdn], x[ts_xsn]), axis=1).tolist(), columns=[ts_vsh_wet_lam, ts_vsh_wet_disp, \
        ts_vsh_wet_struct, ts_vsh_wet_total, ts_vsh_dry_lam, ts_vsh_dry_disp, ts_vsh_dry_struct, \
        ts_vsh_dry_total])
    # or alternatively:
    # vsh = pd.DataFrame(list(map(partial(calc_vsh, phit_clean_sand, phit_lam), ts.TS_SF, ts.TS_XDN, ts.TS_XSN)), columns=['TS_VSH_WET_LAM', 'TS_VSH_WET_DISP', 'TS_VSH_WET_STRUCT', 'TS_VSH_WET_TOTAL', 'TS_VSH_DRY_LAM', 'TS_VSH_DRY_DISP', 'TS_VSH_DRY_STRUCT', 'TS_VSH_DRY_TOTAL'])
    
    data = pd.concat([data, ts, vsh], axis=1)
    data

    # ts_xplot(data[gr_col], data[phit_col], gr_clean_sand, gr_lam, phit_clean_sand, phit_lam, \
    #     show_endpoints=True, plot_structural_triangle=True, plot_gr_axis=True, min_phit=-0.25)
    
    ts_cpi_plot(data)