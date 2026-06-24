from collections import namedtuple
from functools import partial
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from omegaconf import OmegaConf


def to_dms(s: str, msu: str) -> str:
    """
    Converts a string in format <u>:<m>:<s> to <u><msu><m>m<s>s.

    :param s: The initial string
    :param msu: The most significant unit, such as 'd' or 'h'.
    :return:
    """
    parts = s.split(':')
    assert len(parts) == 3
    return f'{parts[0]}{msu}{parts[1]}m{parts[2]}s'


Convert = namedtuple('Convert', ['to', 'src_unit', 'dst_unit'])


class DataProcessor:
    ALL_FLAGS = ['time_flag', 'freq_flag', 'name_flag', 'uncertainty_flag', 'RA_Dec_flag',
                 'FD_flag']

    # These will get converted to the specified column
    CONVERSIONS: Dict[str, Tuple[u, str]] = {
        'days': (u.day, 't'),
        'seconds': (u.second, 't'),
        'GHz': (u.gigahertz, 'frequency'),
        'Hz': (u.hertz, 'frequency'),
        'microJy': (u.microJansky, 'flux'),
        'Jy': (u.jansky, 'flux'),
        'mJy': (u.milliJansky, 'flux')
    }

    # These must be set at the end of the day
    COLUMNS: Dict[str, Tuple[u, type]] = {
        't': (u.second, int),
        'frequency': (u.hertz, int),
        'flux': (u.milliJansky, float)
    }

    def __init__(self, data: pd.DataFrame, config: OmegaConf) -> None:
        self.data = data
        self.config = config

        ra = config.dataset.ra
        dec = config.dataset.dec
        self.constraints = SkyCoord(ra, dec)
        self.arcseconds_uncertainty = config.dataset.arcseconds_uncertainty
        self.match_position = config.dataset.exclude_outside_ra_dec_uncertainty

        excluded_flags = set()
        for flag_name in DataProcessor.ALL_FLAGS:
            exc_name = 'exclude_' + flag_name.lower()
            if exc_name in config.dataset and config.dataset[exc_name]:
                excluded_flags.add(flag_name)
        self.excluded_flags = excluded_flags
        self.target_names = config.dataset.target_name
        self.max_acceptable_flagnum = config.dataset.max_acceptable_flagnum

    def matches_position(self, df) -> bool:
        """
        A filter that matches the ra/dec coordinates specified in the config.
        """
        if not self.match_position:
            return True

        if 'RA' not in df or 'Dec' not in df:
            return True

        ra = to_dms(df['RA'], 'h')
        dec = to_dms(df['Dec'], 'd')

        data_coords = SkyCoord(ra, dec)

        sep = data_coords.separation(self.constraints).to(u.arcsec).value

        return sep <= self.arcseconds_uncertainty

    def matches_flags(self, df) -> bool:
        """
        Process all of the flags given in the .csv file into an array indicating which data points we discard.
        """
        for flag_name in self.excluded_flags:
            # if a exclude_x_flag is specified in the config file, then we
            # exclude rows for which that flag is True. For the name flag,
            # we only exclude if both the flag is true and the target is
            # not in the approved target names
            if flag_name in df and df[flag_name]:
                if flag_name == 'name_flag':
                    if df['target'] not in self.target_names:
                        return False
                else:
                    return False

        # if all previous flag checks pass, we exclude when more flags are set
        # than the maximum number of allowed flags
        total_flags = sum(1 for name in DataProcessor.ALL_FLAGS if name in df and df[name])
        return total_flags < self.max_acceptable_flagnum

    def is_limit(self, flux: str) -> bool:
        return '<' in flux or '>' in flux

    def _interpret(self, data: pd.DataFrame, error_col: bool = True) -> pd.DataFrame:
        data = data[data.apply(self.matches_position, axis=1)]
        data = data[data.apply(self.matches_flags, axis=1)]

        cols = ['t', 'frequency', 'flux']
        if error_col:
            cols += ['err']

        rows = []


        # check that all columns are there
        seen_cols = set()
        for col, (unit, dst_col) in DataProcessor.CONVERSIONS.items():
            seen_cols.add(dst_col)
        for col in DataProcessor.COLUMNS.keys():
            if not col in seen_cols:
                raise ValueError('Missing a {col} column in data.')

        for i in range(data.shape[0]):
            row = {}
            rows.append(row)
            for col, (unit, dst_col) in DataProcessor.CONVERSIONS.items():
                if col in data.columns:
                    val = data.iloc[i][col]
                    dst_unit, dst_type = DataProcessor.COLUMNS[dst_col]
                    if dst_col == 'flux':
                        # special handling for error bars and limits
                        # possible formats:
                        #   < upper_limit
                        #   > lower)limit
                        #   val ± err
                        #   val +- err
                        #       we process these
                        #   val
                        #       print a warning that error is missing and ignore
                        assert isinstance(val, str)
                        if val[0] == '<':
                            val = val[1:]
                        elif val[0] == '>':
                            val = val[1:]
                        else:
                            err = None
                            for sep in ['±', '+-']:
                                if sep in val:
                                    parts = val.split(sep)
                                    val = parts[0]
                                    err = dst_type(unit.to(dst_unit, float(parts[1])))
                            if err is None and error_col:
                                print(f'Warning: missing error in column {col}. Ignoring.')
                            row['err'] = err
                    row[dst_col] = dst_type(unit.to(dst_unit, float(val)))

        return pd.DataFrame(rows, columns=cols)

    def find_flux_col(self) -> str:
        for col, (unit, dst_col) in DataProcessor.CONVERSIONS.items():
            if dst_col == 'flux' and col in self.data.columns:
                return col
        raise ValueError('No flux column found.')

    def interpret(self) -> pd.DataFrame:
        flux_col = self.find_flux_col()
        is_not_limit = lambda x: not self.is_limit(x)
        return self._interpret(self.data.loc[self.data[flux_col].map(is_not_limit)])

    def interpret_ULs(self) -> pd.DataFrame:
        """
        This takes in the same datafile as interpret(), but it picks out the upper limits only. 
        We use this for the plotting.
        """
        flux_col = self.find_flux_col()
        return self._interpret(self.data.loc[self.data[flux_col].map(self.is_limit)], error_col=False)

