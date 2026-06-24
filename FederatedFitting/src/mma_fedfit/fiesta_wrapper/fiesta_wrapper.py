import threading
from typing import Optional, Dict

from fiesta.inference.lightcurve_model import AfterglowFlux
from fiesta.conversions import apply_redshift

import math

import numpy as np

from jax.scipy.interpolate import RegularGridInterpolator
from jax.numpy import interp

from afterglowpy import jet


def _get_filters(nu):
    """
    Returns a list of filter names based on the nus in the input data.
    Fiesta requires a list of filters to be supplied when instantiating a model.

    I'm confident I lack understanding of how filters work here. That said, based
    on the data files I've seen, we're talking about the observation frequency
    for radio and this method (as well as Fiesta) seem to assume a monochromatic
    filter when you say "radio-n.mGHz". Given that we're extracting fluxes for specific
    sharp frequencies anyway (i.e., applying a sharp monochromatic filter), this
    seems to both fit and not be particularly relevant except to make Fiesta happy.

    :param nu:
    :return:
    """
    filters = set()
    for freq in nu:
        fghz = freq / 1e9
        if fghz < 1.0 or fghz > 5e9:
            # This isn't wrong, but too constraining since it doesn't account for
            # redshift
            raise RuntimeError(f'Invalid frequency band: {fghz} GHz. Fiesta requires >= 1GHz and <= 5e18 Hz')
        filters.add(f'radio-{fghz}GHz')
    return list(filters)


def _check_range(x: float, l: float, h: float) -> float:
    if x < l or x > h:
        raise RuntimeError(f'Range check error: {x} not in [{l}, {h}]')
    return x


def _params_afpy_to_fiesta(Z):
    """
    Converts a afterglowpy parameter dictionary to one suitable for Fiesta

    :param Z:
        The parameter dictionary. Must contain:

            * thetaObs
            * E0
            * thetaCore
            * thetaWing
            * n0
            * p
            * epsilon_e
            * epsilon_B
    :return:
        An array that can be passed to Fiesta's FluxModel.predict_log_flux.
    """
    return np.array([
        _check_range(Z['thetaObs'], 0, math.pi / 2),
        _check_range(math.log10(Z['E0']), 47, 57),
        _check_range(Z['thetaCore'], 0.01, math.pi / 5),
        _check_range(Z['thetaWing'] / Z['thetaCore'], 0.2, 3.5),
        _check_range(math.log10(Z['n0']), -6, 2),
        _check_range(Z['p'], 2, 3),
        _check_range(math.log10(Z['epsilon_e']), -4, 0),
        _check_range(math.log10(Z['epsilon_B']), -8, 0)
    ])


def _mag_to_flux(mag):
    """Converts a magnitude to flux.

    :param mag:
        Magnitude to convert
    :return:
        Flux in mJys
    """

    # mag = -48.6 + -1 * jnp.log10(mJys) * 2.5 + 26 * 2.5
    # mJys = 10^[26 - (48.6 + mag)/2.5]

    return np.power(10, 26 - (48.6 + mag) / 2.5)


def _mag_abs_from_mag_app(mag_app, Z):
    """Convert an apparent magnitude to an absolute magnitude

    :param mag_app:
        The apparent magnitude.
    :param Z:
        An afterglowpy style parameter dictionary which must contain
        the luminosity distance (with key d_L).
    :return:
        The absolute magnitude.
    """
    # mag_app = mag_abs + 5.0 * jnp.log10(luminosity_distance * 1e6 / 10.0)
    # mag_abs = mag_app - 5.0 * jnp.log10(luminosity_distance * 1e6 / 10.0)
    return mag_app - 5.0 * np.log10(Z['d_L'] * 1e6 / 10.0)


d_L_ref = 3.086e19  # cm


def _apply_redshift(x, nu, log_flux, model, Z):
    """Applies a redshift to a non-redshifted lightcurve.

    Fiesta's models are trained without redshift since, presumably, that's easy
    to calculate. In particular, the `FluxModel.predict_log_flux` method does
    not include a redshift. On the other hand, `FluxMode.convert_to_mag` does
    apply a redshift, but also converts the flux to a magnitude. One way to
    get a redshifted flux is to take the lightcurve from `predict_log_flux`, shift
    the nus and times to observer frame, then interpolate to get a predicted value
    at (x, nu). The other method of getting a flux is to call `convert_to_mag` and
    then convert the magnitude back to a flux. That's done in `_apply_redhsitf_fiesta`
    and seems to produce the same results.

    :param x:
        Observation times array
    :param nu:
        Frequency array
    :param log_flux:
        The log_flux obtained from `predict_log_flux`
    :param model:
        The fiesta model, which contains the non-redshifted times and nus where
        the flux is specified
    :param Z:
        The afterglowpy parameter dictionary
    :return:
        A redshifted flux.
    """
    # This seems somewhat faster on non-gpus than the fiesta's version below.
    # That's perhaps unsurprising, since the other one involves interpolating twice.
    mJys = np.exp(log_flux)
    mJys_obs, times_obs, nus_obs = apply_redshift(mJys, model.times, model.nus, Z['z'])
    interp = RegularGridInterpolator((nus_obs, times_obs), mJys_obs)
    # Fiesta uses days as the time unit
    x_days = x / (24 * 3600)
    interpolated_mJys = interp(np.array([nu, x_days]).transpose())
    # fiesta fixes d_L to 3.086e19; see fiesta.train.AfterglowData.py, line 350

    return interpolated_mJys * ((d_L_ref / Z['d_L']) ** 2)


def _apply_reshift_fiesta(x, nu, log_flux, model, Z):
    """
    See above.
    """

    obs_times, filter_dict = model.convert_to_mag(log_flux,
                                                  {'redshift': Z['z'],
                                                   'luminosity_distance': Z['d_L']})
    r = []
    for i in range(len(nu)):
        nu_GHz = nu[i] / 1e9
        time_days = x[i] / (24 * 3600)
        mags_all = filter_dict[f'radio-{nu_GHz:2.1f}GHz']
        mag_app = interp(np.array([time_days]), obs_times, mags_all)[0]
        mag_abs = _mag_abs_from_mag_app(mag_app, Z)
        flux = _mag_to_flux(mag_abs) * ((d_L_ref / Z['d_L']) ** 2)
        r.append(flux)

    return np.array(r)


class FiestaWrapper:
    """
    A class that wraps an instance of a Fiesta flux model and exposes it using
    the same interface as afterglowpy's fluxDensity.
    """
    _models: Dict[str, AfterglowFlux] = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str, nu: np.array, use_fiesta_redshift: bool = False) -> None:
        self.fmodel = AfterglowFlux(name=model_name, filters=_get_filters(nu))
        self.use_fiesta_redshift = use_fiesta_redshift
        
    def apply(self, x: np.array, nu: np.array, **Z) -> np.array:
        log_flux = self.fmodel.predict_log_flux(_params_afpy_to_fiesta(Z))
        # result is an array log_flux[i, j], where
        #   t = model.times[i], nu = model.nus[j]
        # all for a zero redshift and 10pc luminosity distance

        if self.use_fiesta_redshift or '_alt_redshift' in Z:
            return _apply_reshift_fiesta(x, nu, log_flux, self.fmodel, Z)
        else:
            return _apply_redshift(x, nu, log_flux, self.fmodel, Z)

    @staticmethod
    def flux_density(x: np.array, nu: np.array,
                     model_name: str = 'afgpy_gaussian_CVAE', **Z) -> np.array:
        with FiestaWrapper._lock:
            if model_name not in FiestaWrapper._models:
                FiestaWrapper._models[model_name] = FiestaWrapper(model_name, nu)
        return FiestaWrapper._models[model_name].apply(x, nu, **Z)


def fluxDensity(x: np.array, nu: np.array,
                model_name: str = 'afgpy_gaussian_CVAE', **Z) -> np.array:
    """Computes a flux density.

    :param x:
        Times in observer frame (seconds).
    :param nu:
        Frequency in observer frame for each time (Hz)
    :param model_name:
        A Fiesta model name, such as 'afgpy_tophat_CVAE'
    :param Z:
        A parameter dictionary for the model, with parameters as described
        in https://afterglowpy.readthedocs.io/en/stable/modules/afterglowpy.html#afterglowpy.fluxDensity.
        Fiesta only uses `thetaObs`, `E0`, `thetaCore`, `thetaWing`, `n0`, `p`, `epsilon_e`, and
        `epsilon_B`.
    :return:
        An array with a flux for each pair (x[i], nu[i]).
    """
    return FiestaWrapper.flux_density(x, nu, model_name, **Z)

