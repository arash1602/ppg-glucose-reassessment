import pmcx
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

import tissue_model.tissue_params as params


Nx, Ny, Nz = params.vox_x, params.vox_y, params.vox_z

vol = np.zeros((Nx, Ny, Nz), dtype = np.uint8)
vol[:,
    :,
    0 : params.bot_glass_depth_vox
    ] = 1

vol[:,
    :,
    params.bot_glass_depth_vox : params.bot_epi_depth_vox
    ] = 2

vol[:,
    :,
    params.bot_epi_depth_vox : params.bot_derm_depth_vox
    ] = 3

vol[:,
    :,
    params.bot_derm_depth_vox : params.bot_sub_depth_vox
    ] = 4

background_optic_params = [0.0, 0.0, 1.0, 1.0]


def run_simulation(
        glass_optic_params, 
        epi_optic_params, 
        derm_optic_params, 
        sub_optic_params
        )-> dict:    
    # =========================================
    # MCeXtreme Simulation
    # =========================================

    # Format: [mu_a, mu_s, g, n]


    # Define simulation values
    prop = [
        background_optic_params,
        glass_optic_params,
        epi_optic_params,
        derm_optic_params,
        sub_optic_params
    ]

    photon_count = 1e8

    sim_config = {
        "nphoton": photon_count, # 1e9: 57 seconds
        "vol": vol,
        "prop": prop,
        "srcpos": [params.src_x_vox, params.src_y_vox, params.src_z_vox],
        "srcdir": [0, 0, 1],
        "detpos": [
            [params.det_x_vox, params.det_y_vox, params.det_z_vox, params.det_r_vox], 
            ],
        "tstart": 0,
        "tend": 10e-9,
        "tstep": 10e-9,
        "unitinmm": params.vox_size_mm,
        "autopilot": 1,
        "gpuid": 1,
        "issrcfrom0": 1,
        "issavedet": 1,
        "issaveseed": 0,
        "issave2pt": 0,
        "maxdetphoton": photon_count,
        "seed": -1,
        "savedetflag" : "dspx"
    }


    # Run simulation
    sim_out = pmcx.mcxlab(sim_config)

    detp = sim_out["detp"]
    positions = np.asarray(detp["p"]) * params.vox_size_mm
    paths = np.asarray(detp["ppath"]) * params.vox_size_mm

    dx = params.src_x_mm - positions[:, 0]
    dy = params.src_y_mm - positions[:, 1]
    r = np.sqrt(dx**2 + dy**2)
    weights = pmcx.detweight(detp)
    nscatter = np.asarray(detp["nscat"])


    
    ring_1 = (r >= params.ring_1_inner_rad) & (r <= params.ring_1_outer_rad)
    ring_2 = (r >= params.ring_2_inner_rad) & (r <= params.ring_2_outer_rad)
    ring_3 = (r >= params.ring_3_inner_rad) & (r <= params.ring_3_outer_rad)
    ring_4 = (r >= params.ring_4_inner_rad) & (r <= params.ring_4_outer_rad)
    ring_5 = (r >= params.ring_5_inner_rad) & (r <= params.ring_5_outer_rad)

    rings = [ring_1, ring_2, ring_3, ring_4, ring_5]
    metrics = {
        "absorbance": np.zeros(5),
        "mean_path": 
            {
                "epi_mean_path": np.zeros(5),
                "derm_mean_path": np.zeros(5),
                "sub_mean_path": np.zeros(5),
            },
        "dA_dmu_s":
            {
                "epi_dA_dmu_s": np.zeros(5),
                "derm_dA_dmu_s": np.zeros(5),
                "sub_dA_dmu_s": np.zeros(5),
            },
        "sensitivity": 
            {
                "epi_sensitivity": np.zeros(5),
                "derm_sensitivity": np.zeros(5),
                "sub_sensitivity": np.zeros(5),
            },
        "power_dB": np.zeros(5)
    }

    SDS_idx = 0
    for ring in rings:
        R = weights[ring].sum() / sim_config["nphoton"]
        power_dB = 10 * np.log10(R)
        A = -np.log(R)

        epi_mean_path = (weights[ring] * paths[ring, 1]).sum() / weights[ring].sum()
        derm_mean_path = (weights[ring] * paths[ring, 2]).sum() / weights[ring].sum()
        sub_mean_path = (weights[ring] * paths[ring, 3]).sum() / weights[ring].sum()
        total_path = epi_mean_path + derm_mean_path + sub_mean_path

        epi_dA_dmu_s = epi_mean_path - ((weights[ring] * nscatter[ring, 1]).sum() / weights[ring].sum()) / epi_optic_params[1]
        derm_dA_dmu_s = derm_mean_path - ((weights[ring] * nscatter[ring, 2]).sum() / weights[ring].sum()) / derm_optic_params[1]
        sub_dA_dmu_s = sub_mean_path - ((weights[ring] * nscatter[ring, 3]).sum() / weights[ring].sum()) / sub_optic_params[1]

        epi_sensitivity = epi_mean_path / total_path
        derm_sensitivity = derm_mean_path / total_path
        sub_sensitivity = sub_mean_path / total_path

        metrics["absorbance"][SDS_idx] = A
        metrics["power_dB"][SDS_idx] = power_dB

        metrics["mean_path"]["epi_mean_path"][SDS_idx] = epi_mean_path
        metrics["mean_path"]["derm_mean_path"][SDS_idx] = derm_mean_path
        metrics["mean_path"]["sub_mean_path"][SDS_idx] = sub_mean_path

        metrics["dA_dmu_s"]["epi_dA_dmu_s"][SDS_idx] = epi_dA_dmu_s
        metrics["dA_dmu_s"]["derm_dA_dmu_s"][SDS_idx] = derm_dA_dmu_s
        metrics["dA_dmu_s"]["sub_dA_dmu_s"][SDS_idx] = sub_dA_dmu_s

        metrics["sensitivity"]["epi_sensitivity"][SDS_idx] = epi_sensitivity
        metrics["sensitivity"]["derm_sensitivity"][SDS_idx] = derm_sensitivity
        metrics["sensitivity"]["sub_sensitivity"][SDS_idx] = sub_sensitivity


        SDS_idx += 1



    # replay_config = sim_config.copy()
    # replay_config["outputtype"] = "jacobian"
    # replay_config["detphotons"] = sim_out["detp"]
    # replay_config["seed"] = sim_out["seeds"]

    # replay_out = pmcx.run(replay_config)

    # jacobian_PMDF = replay_out["flux"]



    # flux = sim_out["flux"]

    # fig_flux, axes_flux = plt.subplots(1, 2, layout = "constrained")

    # # select one time gate
    # flux_1 = flux[:, :, :, 0]
    # flux_1_yslice = flux_1[:, 30, :]
    # axes_flux[0].set_title("Fluence Map")
    # axes_flux[0].set_xlabel("x-axis (voxels)")
    # axes_flux[0].set_ylabel("z-axis (voxels)")
    # axes_flux[0].axhline(params.bot_epi_depth_vox, linestyle = "--", color = "black")
    # axes_flux[0].axhline(params.bot_derm_depth_vox, linestyle = "--", color = "black")
    # axes_flux[0].text(1, params.bot_epi_depth_vox / 2, "epidermis")
    # axes_flux[0].text(1, params.bot_derm_depth_vox / 2, "dermis")
    # axes_flux[0].text(1, params.bot_sub_depth_vox / 2, "subcutaneous")
    # axes_flux[0].arrow(sim_config["srcpos"][0], sim_config["srcpos"][2] - 1, 0, 2, head_width = 1, head_length = 1, color = "red", label = "source")
    # axes_flux[0].arrow(sim_config["detpos"][0][0], sim_config["detpos"][0][2] - 1, 0, 2, head_width = 1, head_length = 1, color = "blue", label = "detector" )

    # axes_flux[0].legend()
    # axes_flux[0].imshow(np.log10(np.abs(flux_1_yslice.T) + 1e-20))

    # jacobian_1 = np.sum(jacobian_PMDF, axis = 3)
    # jacobian_1_slice = jacobian_1[:, 30, :]
    # axes_flux[1].set_title("Spatial Absorption Sensitivity")
    # axes_flux[1].set_xlabel("x-axis (voxels)")
    # axes_flux[1].set_ylabel("z-axis (voxels)")
    # axes_flux[1].axhline(3, linestyle = "--", color = "black")
    # axes_flux[1].axhline(13, linestyle = "--", color = "black")
    # axes_flux[1].text(1, 1.5, "epidermis")
    # axes_flux[1].text(1, 8, "dermis")
    # axes_flux[1].text(1, 30, "subcutaneous")
    # axes_flux[1].arrow(sim_config["srcpos"][0], sim_config["srcpos"][2] - 1, 0, 2, head_width = 1, head_length = 1, color = "red", label = "source")
    # axes_flux[1].arrow(sim_config["detpos"][0][0], sim_config["detpos"][0][2] - 1, 0, 2, head_width = 1, head_length = 1, color = "blue", label = "detector" )

    # axes_flux[1].legend()
    # image = axes_flux[1].imshow(np.log10(np.abs(jacobian_1_slice.T) + 1e-20))

    # # fig_flux.colorbar(
    # #     image,
    # #     ax = axes_flux[0],
    # # )

    # fig_flux.colorbar(
    #     image,
    #     ax = axes_flux[1],
    # )

    # PMDF = np.sum(jacobian_PMDF, axis = 3)
    # PMDF_epi = np.sum(PMDF[vol == 1])
    # PMDF_derm = np.sum(PMDF[vol == 2])
    # PMDF_sub = np.sum(PMDF[vol == 3])

    # PMDF_total = PMDF_epi + PMDF_derm + PMDF_sub

    # PMDF_epi_per = 100 * (PMDF_epi / PMDF_total)
    # PMDF_derm_per = 100 * (PMDF_derm / PMDF_total)
    # PMDF_sub_per = 100 * (PMDF_sub / PMDF_total)

    # print(f"ES(%): {PMDF_epi_per:.2f} %")
    # print(f"DS(%): {PMDF_derm_per:.2f} %")
    # print(f"SS(%): {PMDF_sub_per:.2f} %")

    return metrics