import numpy as np
import pandas as pd
import afterglowpy as grb
from astropy.coordinates import SkyCoord
import astropy.units as u
import corner
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# This avoids any GUI-related thread issues and will still allow saving plots via plt.savefig().

#-------------------------------------------------
# Dictionary for filter names (if needed)
#-------------------------------------------------
# band_dict = {
#     2.418e17: "1 keV",
#     1.555e15: "UVW2",
#     1.335e15: "UVM2",
#     1.153e15: "UVW1",
#     8.652e14: "U",
#     8.443e14: "u",
#     6.826e14: "B",
#     6.389e14: "g",
#     5.483e14: "V",
#     4.862e14: "r",
#     4.008e14: "i",
#     3.356e14: "z",
#     2.398e14: "J",
#     1.851e14: "H",
#     1.414e14: "Ks",
#     1.000e10: "10 GHz",
#     6.000e09: "6 GHz",
# }

radio_bands = {
    "L-band": (1e9, 2e9, 1.5e9, 'o'),
    "S-band": (2e9, 4e9, 3e9, 's'),
    "C-band": (4e9, 8e9, 6e9, '^'),
    "X-band": (8e9, 12e9, 10e9, '>'),
    "Ku-band": (12e9, 18e9, 15e9, '*'),
    "K-band": (18e9, 26.5e9, 22e9, 'D'),
    "Ka-band": (26.5e9, 40e9, 33e9, 'P')
}
#-------------------------------------------------
# Data interpretation function (as before)
#-------------------------------------------------
def process_RA_Dec_constraints(data, client_agent_config):
    
    """
    Here we examine the data and reject all which falls outside of the user's specified arcsecond_uncertainty
    """
    
    #Collect the indices we keep:
    #If True, we use the data. If False, we exclude it.
    indices_not_flagged_for_exclusion_RA_Dec = []
    
    exclude_outside_ra_dec_uncertainty = client_agent_config.mcmc_configs.exclude_outside_ra_dec_uncertainty
    if "RA" in data.columns and "Dec" in data.columns and exclude_outside_ra_dec_uncertainty == True:
            
        for i in range(data.shape[0]):
            #These are the RA and Dec given by the data.
            RA_str = data.iloc[i]["RA"]
            Dec_str = data.iloc[i]["Dec"]

            #put the units in:
            RA_str = RA_str.split(':')
            RA = RA_str[0] + 'h' + RA_str[1] + 'm' + RA_str[2] + 's' 

            #put the units in:
            Dec_str = Dec_str.split(':')
            Dec = Dec_str[0] + 'd' + Dec_str[1] + 'm' + Dec_str[2] + 's'

            coordinates_from_data = SkyCoord(RA, Dec)

            ra = client_agent_config.mcmc_configs.ra
            dec = client_agent_config.mcmc_configs.dec
            coordinates_from_user = SkyCoord(ra, dec)

            # Calculate angular separation
            separation = coordinates_from_data.separation(coordinates_from_user)
            separation = separation.to(u.arcsec).value
            arcseconds_uncertainty = client_agent_config.mcmc_configs.arcseconds_uncertainty
            if separation <= arcseconds_uncertainty:
                indices_not_flagged_for_exclusion_RA_Dec.append(True) #we keep it
            if separation > arcseconds_uncertainty:
                indices_not_flagged_for_exclusion_RA_Dec.append(False) #we toss it
        return indices_not_flagged_for_exclusion_RA_Dec

    else:
        #Then we use all the data
        for i in range(data.shape[0]):
            indices_not_flagged_for_exclusion_RA_Dec.append(True)
        return indices_not_flagged_for_exclusion_RA_Dec

def process_flags(data, client_agent_config):
    """
    Process all of the flags given in the .csv file into an array indicating which data points we discard.
    """
    
    #Collect the indices we keep:
    #If True, we use the data. If False, we exclude it.
    indices_not_flagged_for_exclusion = []

    #handle each flag:
    exclude_time_flag = client_agent_config.mcmc_configs.exclude_time_flag
    exclude_ra_dec_flag = client_agent_config.mcmc_configs.exclude_ra_dec_flag
    exclude_name_flag = client_agent_config.mcmc_configs.exclude_name_flag
    exclude_uncertainty_flag = client_agent_config.mcmc_configs.exclude_uncertainty_flag

    if "time_flag" in data.columns and exclude_time_flag == True:
        use_time_flag = True
    else:
        use_time_flag = False

    if "RA_Dec_flag" in data.columns and exclude_ra_dec_flag == True:
        use_RA_Dec_flag = True
    else:
        use_RA_Dec_flag = False

    if "name_flag" in data.columns and exclude_name_flag == True:
        use_name_flag = True
    else:
        use_name_flag = False

    if "uncertainty_flag" in data.columns and exclude_uncertainty_flag == True:
        use_uncertainty_flag = True
    else:
        use_uncertainty_flag = False
        
    for i in range(data.shape[0]):
        flags = []
        if use_time_flag:
            time_flag = data.iloc[i]["time_flag"]
            flags.append(time_flag)
        if use_RA_Dec_flag:
            RA_Dec_flag = data.iloc[i]["RA_Dec_flag"]
            flags.append(RA_Dec_flag)
        if use_uncertainty_flag:
            unc_flag = data.iloc[i]["uncertainty_flag"]
            flags.append(unc_flag)
        if use_name_flag:
            name_flag = data.iloc[i]["name_flag"]
            target = data.iloc[i]["target"]
            target_name = client_agent_config.mcmc_configs.target_name

            if not target in target_name:
                name_flag = 1 #This is the same as throwing the name_flag -- rejects the data point.
                
            flags.append(name_flag)
            
        flags = np.asarray(flags)

        #The total number of flags in this data point (6 possible)
        total_flags = 0
        for flag_name in ['time_flag', 'freq_flag', 'name_flag', 'uncertainty_flag', 'RA_Dec_flag', 'FD_flag']:
            if flag_name in data.iloc[i]:
                total_flags += data.iloc[i][flag_name]


        #This means it contains a nonzero flag, which the user has set to exclude
        #Or the total number of flags is worryingly high to the user
        max_acceptable_flagnum = client_agent_config.mcmc_configs.max_acceptable_flagnum
        
        if np.sum(flags) > 0 or total_flags > max_acceptable_flagnum:
            indices_not_flagged_for_exclusion.append(False)
        else:
            indices_not_flagged_for_exclusion.append(True)
    
    return indices_not_flagged_for_exclusion



def interpret(data, client_agent_config):
    print('interpret data')
    if "days" in data.columns:
        data["t"] = data["days"] * 86400
    elif "seconds" in data.columns:
        data["t"] = data["seconds"]
    else:
        raise RuntimeError("Error: no time column in data. Need either seconds or days.")
     
    # if "Filter" in data.columns:
    #     data["filter"] = data["Filter"]
    # elif "Band" in data.columns:
    #     data["filter"] = data["Band"]
    # elif "band" in data.columns:
    #     data["filter"] = data["band"]
    
    if "GHz" in data.columns:
        data["frequency"] = data["GHz"]
        freq_correct = 1e9
    elif "Hz" in data.columns:
        data["frequency"] = data["Hz"]
        freq_correct = 1
    else:
        raise RuntimeError("Error: no frequency column in data. Need either Hz or GHz.")
    
    if "microJy" in data.columns:
        data["flux"] = data["microJy"]
        flux_correct = 1e-3
    elif "Jy" in data.columns:
        data["flux"] = data["Jy"]
        flux_correct = 1e3
    elif "mJy" in data.columns:
        data["flux"] = data["mJy"]
        flux_correct = 1
    else:
        raise RuntimeError("Error: no flux density column in data. Need mJy, microJy, or Jy.")

    #use the RA and Dec radius:
    indices_not_flagged_for_exclusion_RA_Dec = process_RA_Dec_constraints(data, client_agent_config)
    
    #use the flags:
    indices_not_flagged_for_exclusion = process_flags(data, client_agent_config)
    
    freq, new_flux, err, tvals = [], [], [], []

    for i in range(data.shape[0]):
        if indices_not_flagged_for_exclusion[i] == True and indices_not_flagged_for_exclusion_RA_Dec[i] == True:
            try:
                this_freq = float(data.iloc[i]["frequency"]) * freq_correct
            except:
                continue  # skip this row entirely if frequency can't be parsed
    
            # if "err" in data.columns:
            #     flux = data.iloc[i]["flux"]
            #     error = float(data.iloc[i]["err"])
            #     if "<" in flux:
            #         print('skipping UL')
            #         #new_flux.append("UL")
            #         #err.append(0)
            #     elif ">" in flux:
            #         print('skipping UL')
            #         #new_flux.append("UL")
            #         #err.append(0)
            #     else:
            #         flux = float(flux)
            #         new_flux.append(flux)
            #         err.append(float(error))
            #         freq.append(this_freq)
            #         tvals.append(data.iloc[i]["t"])
            # else:
            flux = data.iloc[i]["flux"]

            if "<" in flux:
                continue
                #new_flux.append("UL")
                #err.append(0)
            elif ">" in flux:
                continue
                #new_flux.append("UL")
                #err.append(0)
            
            if "±" in flux:
                splt = flux.split("±")
                new_flux.append(float(splt[0]))
                err.append(float(splt[1]))
                freq.append(this_freq)
                tvals.append(data.iloc[i]["t"])
            elif "+-" in flux:
                splt = flux.split("+-")
                new_flux.append(float(splt[0]))
                err.append(float(splt[1]))
                tvals.append(data.iloc[i]["t"])
                freq.append(this_freq)
            # elif "+" in flux and "-" in flux:
            #     splt = flux.split("+")
            #     new_flux.append(float(splt[0]))
            #     err_splt = splt[1].split("-")
            #     err.append(max([float(splt[0]), float(err_splt[1])]))
            #     freq.append(this_freq)
            #     tvals.append(data.iloc[i]["t"])
            # else:
            #     new_flux.append(float(flux))
            #     err.append(0)
            #     freq.append(this_freq)
            #     tvals.append(data.iloc[i]["t"])
            else:
                #If it does not have a defined uncertainty, it cannot be included in the fit.
                print('A detection is indicated but does not have an associated uncertainty. Please manually enter one')
                continue

    for i in range(len(new_flux)):
        if new_flux[i] != "UL":
            # if flux_correct != "mag":
            new_flux[i] = new_flux[i] * flux_correct
            err[i] = err[i] * flux_correct

    # Build a clean DataFrame only from valid, parsed values

    print(len(tvals))
    print(len(freq))
    print(len(new_flux))
    print(len(err))
    
    valid_data = pd.DataFrame({
    "t": tvals,
    "frequency": freq,
    "flux": new_flux,
    "err": err})
    
    print(f'number of detections to fit after Flags:{len(tvals)}')

    # Return just the columns we want, fully cleaned
    valid_data = valid_data[["t", "frequency", "flux", "err"]].astype(np.float64)
    return valid_data


def interpret_ULs(data, client_agent_config):
    print('interpret UL data')
    """
    This takes in the same datafile as interpret(), but it picks out the upper limits only. 
    We use this for the plotting.
    """
    if "days" in data.columns:
        data["t"] = data["days"] * 86400
    elif "seconds" in data.columns:
        data["t"] = data["seconds"]
    # elif "t_delta" in data.columns:
    #     data["t"] = data["t_delta"]
    else:
        sys.exit("Error: no time column in data. Need either seconds or days.")
    
    
    # if "Filter" in data.columns:
    #     data["filter"] = data["Filter"]
    # elif "Band" in data.columns:
    #     data["filter"] = data["Band"]
    # elif "band" in data.columns:
    #     data["filter"] = data["band"]
    
    if "GHz" in data.columns:
        data["frequency"] = data["GHz"]
        freq_correct = 1e9
    elif "Hz" in data.columns:
        data["frequency"] = data["Hz"]
        freq_correct = 1
    else:
        sys.exit("Error: no frequency column in data. Need either Hz or GHz.")
    
    if "microJy" in data.columns:
        data["flux"] = data["microJy"]
        flux_correct = 1e-3
    elif "Jy" in data.columns:
        data["flux"] = data["Jy"]
        flux_correct = 1e3
    elif "mJy" in data.columns:
        data["flux"] = data["mJy"]
        flux_correct = 1
    # elif "mag" in data.columns:
    #     data["flux"] = data["mag"]
    #     flux_correct = "mag"
    else:
        sys.exit("Error: no flux density column in data. Need mJy, microJy, or Jy.")


    #use the RA and Dec radius:
    indices_not_flagged_for_exclusion_RA_Dec = process_RA_Dec_constraints(data, client_agent_config)
    
    #use the flags:
    indices_not_flagged_for_exclusion = process_flags(data, client_agent_config)

    freq, new_flux, tvals = [], [], []

    for i in range(data.shape[0]):
        if indices_not_flagged_for_exclusion[i] == True and indices_not_flagged_for_exclusion_RA_Dec[i] == True:
            try:
                this_freq = float(data.iloc[i]["frequency"]) * freq_correct
            except:
                continue  # skip this row entirely if frequency can't be parsed
    
            # if "err" in data.columns:
            #     flux = data.iloc[i]["flux"]
            #     error = float(data.iloc[i]["err"])
            #     if "<" in flux:
            #         flux_str = flux.split('<')
            #         flux = flux_str[1]
            #         flux = float(flux)
            #         new_flux.append(flux)
            #         freq.append(this_freq)
            #         tvals.append(data.iloc[i]["t"])
            #     elif ">" in flux:
            #         flux_str = flux.split('>')
            #         flux = flux_str[1]
            #         flux = float(flux)
            #         new_flux.append(flux)
            #         freq.append(this_freq)
            #         tvals.append(data.iloc[i]["t"])
    
            # else:
            flux = data.iloc[i]["flux"]

            if "<" in flux:
                flux_str = flux.split('<')
                flux = flux_str[1]
                flux = float(flux)
                new_flux.append(flux)
                freq.append(this_freq)
                tvals.append(data.iloc[i]["t"])
            elif ">" in flux:
                flux_str = flux.split('>')
                flux = flux_str[1]
                flux = float(flux)
                new_flux.append(flux)
                freq.append(this_freq)
                tvals.append(data.iloc[i]["t"])

    for i in range(len(new_flux)):
        if new_flux[i] != "UL":
            # if flux_correct != "mag":
            new_flux[i] = new_flux[i] * flux_correct
                #err[i] = err[i] * flux_correct

    print(len(tvals))
    print(len(freq))
    print(len(new_flux))
    
    valid_data = pd.DataFrame({
    "t": tvals,
    "frequency": freq,
    "flux": new_flux})
    
    print(f'number of ULs after Flags:{len(tvals)}')

    # Return just the columns we want, fully cleaned
    valid_data = valid_data[["t", "frequency", "flux"]].astype(np.float64)
    return valid_data


# --- Function to Plot Global Light Curve Locally ---
def plot_global_light_curve(data, theta, global_min, global_max, frequencies, output_filename):
    
    times = np.geomspace(global_min, global_max, 100)
    fig, ax = plt.subplots()
    E0, thetaObs, thetaCore, n0, epsilon_e, epsilon_B, p = theta
    Z = {
        "jetType": grb.jet.Gaussian,
        "specType": 0,
        "thetaObs": thetaObs,
        "E0": E0,
        "thetaCore": thetaCore,
        "thetaWing": 0.6,
        "n0": n0,
        "p": p,
        "epsilon_e": epsilon_e,
        "epsilon_B": epsilon_B,
        "xi_N": 1.0,
        "d_L": 1.2344e26,
        "z": 0.00897,
    }

    colors = plt.cm.viridis(np.linspace(0, 1, len(frequencies)))
    for i, freq in enumerate(frequencies):
        nus = np.full(times.shape, freq)
        model = grb.fluxDensity(times, nus, **Z)
        ax.plot(times, model, color=colors[i], label=f"Freq: {freq:.2e} Hz")
    ax.set(xlabel="Time since detection (s)", ylabel="Flux density (mJy)",
           xscale="log", yscale="log")
    ax.legend()
    plt.savefig(output_filename)
    plt.close()
    print(f"Global light curve saved to {output_filename}", flush=True)


    # times = np.geomspace(global_min, global_max, 100)
    # fig, ax = plt.subplots()
    # E0, thetaObs, thetaCore, n0, epsilon_e, epsilon_B, p = theta
    # Z = {
    #     "jetType": grb.jet.Gaussian,
    #     "specType": 0,
    #     "thetaObs": thetaObs,
    #     "E0": E0,
    #     "thetaCore": thetaCore,
    #     "thetaWing": 0.6,
    #     "n0": n0,
    #     "p": p,
    #     "epsilon_e": epsilon_e,
    #     "epsilon_B": epsilon_B,
    #     "xi_N": 1.0,
    #     "d_L": 1.2344e26,
    #     "z": 0.00897,
    # }

    # colors = plt.cm.viridis(np.linspace(0, 1, len(frequencies)))
    
    # filters = sorted(data["frequency"].unique())

    # for i in range(len(filters)):
    #     to_plot = data.loc[data["frequency"] == filters[i]]
    #     nus = np.full(times.shape, filters[i])
    #     model = grb.fluxDensity(times, nus, **Z)
    #     ax.plot(times, model, color=colors[i], linewidth=0.5)
    #     ax.errorbar(
    #         to_plot["t"],
    #         to_plot["flux"],
    #         yerr=to_plot["err"],
    #         capsize=0,
    #         fmt=".",
    #         label=band_dict[filters[i]],
    #         color=colors[i],
    #     )
    """
    File "/u/parthpatel7173/MMA_RadioWave/FederatedFitting/src/mma_fedfit/generator/inference_utils.py", line 170, in plot_global_light_curve
    label=band_dict[filters[i]],
    KeyError: 670000000.0

    """
    # ax.set(
    #     xlabel="Time since detection (s)",
    #     ylabel="Flux density (mJy)",
    #     xscale="log",
    #     yscale="log",
    # )
    # ax.legend(frameon=True)
    # plt.savefig(output_filename)
    # plt.close()
    # print(f"Global light curve saved to {output_filename}", flush=True)

def make_Log_Likelihood_plot(log_prob, burnin, nwalkers, plot_names, save_path):
    """
    Modified to take save path as a parameter
    """
    plt.figure(figsize=(10, 6))
    for i in range(nwalkers):
        plt.plot(log_prob[:, i], alpha=0.3)
    
    plt.xlabel("Steps")
    plt.ylabel("Log Likelihood")
    plt.title("Log Likelihood Progression " + plot_names + " walkers = " + str(nwalkers))
    plt.ylim(-500, 120)
    plt.axvline(burnin, label='burnin = ' + str(burnin), color='g', lw='.5')
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')

def make_corner_plots(samples, burnin, nwalkers, ndim, params, true_values, display_truths_on_corner, save_path):
    """
    Modified to take parameters as arguments
    """
    figure = corner.corner(
        #we've already applied burnin
        #samples[burnin * nwalkers:],
        samples,
        labels=params,
        show_titles=True,
        title_fmt=".2f",
        quantiles=[0.05, 0.5, 0.95],  # 90% credible interval
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
        plot_datapoints=True,
        fill_contours=True,
        levels=(0.50, 0.90,),  # 90% confidence contours
        smooth=1.0,
        smooth1d=1.0,
        truths=true_values[0:ndim] if display_truths_on_corner else None
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')

def make_posterior_hists(samples, burnin, nwalkers, ndim, params, save_path):
    """
    Modified to take parameters as arguments
    """
    medians = np.median(samples[burnin * nwalkers:], axis=0)

    # print('median parameter values after burnin:')
    # for i in range(ndim):
    #     print(f"{params[i]}: {medians[i]:.4f}")
    
    # Create subplots
    if ndim == 8:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows, 4 columns
    if ndim == 9:
        fig, axes = plt.subplots(3, 3, figsize=(16, 10))  # 3 rows, 3 columns
    if ndim == 10:
        fig, axes = plt.subplots(2, 5, figsize=(16, 10))  # 3 rows, 3 columns
        
    axes = axes.flatten()
    
    # Loop over each dimension and create a histogram
    theta = []
    
    for i in range(ndim):
        # theta_component = np.asarray(samples[burnin * nwalkers:, i])
        theta_component = np.asarray(samples[:, i])
        lower, upper = np.percentile(theta_component, [15.865, 84.135])
    
        theta.append(theta_component)
        ax = axes[i]
        # ax.hist(samples[burnin * nwalkers:, i], bins=20, color="blue", alpha=0.7, label="Samples")
        ax.hist(samples[:, i], bins=20, color="blue", alpha=0.7, label="Samples")
    
        # Plot mean value as a vertical line
        ax.axvline(medians[i], color="red", linestyle="--", label=f"Median: {medians[i]:.4f}")
        ax.axvline(lower, color="green", linestyle="--", label=f"lower limit: {lower:.4f}")
        ax.axvline(upper, color="green", linestyle="--", label=f"upper limit: {upper:.4f}")
        
        ax.set_title(params[i])
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend(loc=4)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')

def plot_lc_noUL(data, theta, site_id):
    """
    Plot light curve based on local data and consensus parameters
    """
    data = data.sort_values(by=["frequency"], ascending=False)
    
    # Extract and transform model parameters
    if z_known:
        E0, thetaObs, thetaCore, n0, epsilon_e, epsilon_B, p, thetaWing = theta
        z = z_fixed
    else:
        E0, thetaObs, thetaCore, n0, epsilon_e, epsilon_B, p, thetaWing, z = theta

    E0 = 10**E0
    n0 = 10**n0
    epsilon_e = 10**epsilon_e
    epsilon_B = 10**epsilon_B

    DL = cosmo.luminosity_distance(z).to(u.cm)
    DL = DL.value

    Z = {
        "jetType": grb.jet.Gaussian,
        "specType": 0,
        "thetaObs": thetaObs,
        "E0": E0,
        "thetaCore": thetaCore,
        "thetaWing": thetaWing,
        "n0": n0,
        "p": p,
        "epsilon_e": epsilon_e,
        "epsilon_B": epsilon_B,
        "xi_N": 1.0,
        "d_L": DL,
        "z": z,
    }

    times = np.geomspace(min(data["t"]), max(data["t"]), 100)
    fig, ax = plt.subplots(figsize=(10, 7))

    band_colors_list = ['black', 'red', 'blue', 'green', 'magenta']
    legend_handles = []

    # First plot model curves for all bands
    for i, (band, (fmin, fmax, fcen, marker)) in enumerate(radio_bands.items()):
        # Filter data for this band
        band_data = data[(data["frequency"] >= fmin) & (data["frequency"] < fmax)]
        
        if not band_data.empty:
            color = band_colors_list[i % len(band_colors_list)]
            # Plot model curve for this band
            model = grb.fluxDensity(times, np.full(times.shape, fcen), **Z)
            ax.plot(times / (24 * 60.0 * 60), model, color=color, linewidth=1.2)

            # Create legend handle for band
            legend_label = f"{band} ({fcen/1e9:.1f} GHz)"
            legend_handles.append(Line2D(
                [0], [0],
                marker=None,
                color=color,
                linestyle='-',
                linewidth=1.2,
                label=legend_label
            ))

            # Plot data points for this band
            band_t = np.array(band_data["t"]) / (24 * 60.0 * 60)
            band_fnu = np.array(band_data["flux"])
            band_err = np.array(band_data["err"])

            ax.errorbar(
                band_t,
                band_fnu,
                yerr=band_err,
                fmt='o',
                ecolor=color,
                elinewidth=0.6,
                capsize=2,
                color=color,
                alpha=0.8
            )

    ax.set(
        xlabel="Time since detection (days)",
        ylabel="Flux density (mJy)",
        xscale="log",
        yscale="log",
        title=f"Site {site_id} - Consensus Model"
    )

    ax.legend(handles=legend_handles, loc="lower left", frameon=False)
    plt.tight_layout()
    plt.savefig(f"./site{site_id}/lightcurve_consensus.png", bbox_inches='tight')

def plot_lc_wUL(data, data_UL, theta, site_id):
    """
    Plot light curve with upper limits based on local data and consensus parameters
    """
    data = data.sort_values(by=["frequency"], ascending=False)
    data_UL = data_UL.sort_values(by=["frequency"], ascending=False)
    
    # Extract and transform model parameters
    if z_known:
        E0, thetaObs, thetaCore, n0, epsilon_e, epsilon_B, p, thetaWing = theta
        z = z_fixed
    else:
        E0, thetaObs, thetaCore, n0, epsilon_e, epsilon_B, p, thetaWing, z = theta

    E0 = 10**E0
    n0 = 10**n0
    epsilon_e = 10**epsilon_e
    epsilon_B = 10**epsilon_B

    DL = cosmo.luminosity_distance(z).to(u.cm)
    DL = DL.value

    Z = {
        "jetType": grb.jet.Gaussian,
        "specType": 0,
        "thetaObs": thetaObs,
        "E0": E0,
        "thetaCore": thetaCore,
        "thetaWing": thetaWing,
        "n0": n0,
        "p": p,
        "epsilon_e": epsilon_e,
        "epsilon_B": epsilon_B,
        "xi_N": 1.0,
        "d_L": DL,
        "z": z,
    }

    times = np.geomspace(min(data["t"]), max(data["t"]), 100)
    fig, ax = plt.subplots(figsize=(10, 7))

    band_colors_list = ['black', 'red', 'blue', 'green', 'magenta']
    legend_handles = []

    # First plot model curves for all bands
    for i, (band, (fmin, fmax, fcen, marker)) in enumerate(radio_bands.items()):
        # Filter data for this band
        band_data = data[(data["frequency"] >= fmin) & (data["frequency"] < fmax)]
        band_data_UL = data_UL[(data_UL["frequency"] >= fmin) & (data_UL["frequency"] < fmax)]
        
        if not band_data.empty or not band_data_UL.empty:
            color = band_colors_list[i % len(band_colors_list)]
            # Plot model curve for this band
            model = grb.fluxDensity(times, np.full(times.shape, fcen), **Z)
            ax.plot(times / (24 * 60.0 * 60), model, color=color, linewidth=1.2)

            # Create legend handle for band
            legend_label = f"{band} ({fcen/1e9:.1f} GHz)"
            legend_handles.append(Line2D(
                [0], [0],
                marker=None,
                color=color,
                linestyle='-',
                linewidth=1.2,
                label=legend_label
            ))

            # Plot regular data points for this band
            if not band_data.empty:
                band_t = np.array(band_data["t"]) / (24 * 60.0 * 60)
                band_fnu = np.array(band_data["flux"])
                band_err = np.array(band_data["err"])

                ax.errorbar(
                    band_t,
                    band_fnu,
                    yerr=band_err,
                    fmt='o',
                    ecolor=color,
                    elinewidth=0.6,
                    capsize=2,
                    color=color,
                    alpha=0.8
                )
            
            # Plot upper limits for this band
            if not band_data_UL.empty:
                band_t_UL = np.array(band_data_UL["t"]) / (24 * 60.0 * 60)
                band_fnu_UL = np.array(band_data_UL["flux"])
                
                ax.scatter(
                    band_t_UL,
                    band_fnu_UL,
                    marker='v',
                    facecolors='none',
                    edgecolors=color,
                    s=40,
                    alpha=0.9
                )

    # Add upper limit to legend
    legend_handles.append(Line2D(
        [0], [0],
        marker='v',
        color='black',
        markerfacecolor='none',
        markeredgecolor='black',
        linewidth=0,
        markersize=7,
        label="Upper limit"
    ))

    ax.set(
        xlabel="Time since detection (days)",
        ylabel="Flux density (mJy)",
        xscale="log",
        yscale="log",
        title=f"Site {site_id} - Consensus Model"
    )

    ax.legend(handles=legend_handles, loc="lower left", frameon=False)
    plt.tight_layout()
    plt.savefig(f"./site{site_id}/lightcurve_consensus.png", bbox_inches='tight')
