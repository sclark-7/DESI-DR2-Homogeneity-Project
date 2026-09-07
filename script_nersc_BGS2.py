#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# measurement of BGS mocks only

import numpy as np
import sys
import time
import astropy.units as u
import h5py
import hdf5plugin  
from astropy.table import Table
#from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from sklearn.neighbors import KDTree
from multiprocessing import Pool, current_process

nrand = 3


def Nr(begin, end, tree_main, tree_rand, coords_main, weights_main, weights_rand, radii, radius_bins, rand_density):

    print(current_process(), begin, end)
    # Ignore divide-by-zero warnings (happens when random counts = 0)
    with np.errstate(divide="ignore"):
        inds, dists = tree_main.query_radius(coords_main[begin:end], r=radii[-1], return_distance=True)

        # For each galaxy: histogram distances into bins, weight them,
        # then take cumulative sum to get N(<R)

        weighted_counts_main = np.array(
            [
                np.cumsum(np.histogram(dist, radius_bins, weights=weights_main[ind])[0])
                for (ind, dist) in zip(inds, dists)
            ]
        )
                # 2. Same thing for RANDOM neighbours

        inds, dists = tree_rand.query_radius(coords_main[begin:end], r=radii[-1], return_distance=True)
        weighted_counts_rand = np.array(
            [
                np.cumsum(np.histogram(dist, radius_bins, weights=weights_rand[ind])[0])
                for (ind, dist) in zip(inds, dists)
            ]
        )


        # 3. Compute ratio = (data / random) × normalisation
        #    Transpose so shape becomes (Nradii, Ngalaxies_in_batch)
        ratio = (weighted_counts_main * rand_density / weighted_counts_rand).T
       
        # Multiply each galaxy’s ratio by its weight and sum over galaxies
        # Mask invalid values (inf, nan)
        MSC_data = np.ma.masked_invalid(ratio * weights_main[begin:end]).sum(axis=1)
    
        # Count how many galaxies contributed finite values at each radius
        count_data = [np.sum(weights_main[begin:end][np.isfinite(rat)]) for rat in ratio]

    return count_data, MSC_data


def read_lss_table(fname):
    with h5py.File(fname, "r") as f:
        g = f["LSS"]
        return Table({k: np.array(g[k]) for k in g.keys()})


# MAIN PROGRAM
if __name__ == "__main__":


    # Cosmology used to convert redshift → comoving distance
    cosmo = FlatLambdaCDM(H0=100.0 * u.km / u.s / u.Mpc, Om0=0.3151917236644108)


    # Read command-line arguments

    # Parameters from command line
    tracer = sys.argv[1]
    skyarea = sys.argv[2]
    mock = int(sys.argv[3])
    sys_weight = int(sys.argv[4])

    # Dictionary describing each tracer type

    # Tracer configurations
    tracers = {
        "BGS1": ["AbacusSummitBGS_v2", "BGS_BRIGHT-02", [0.1, 0.25]],
        "BGS2": ["AbacusSummitBGS_v2", "BGS_BRIGHT-02", [0.25, 0.4]],
        #"LRG1": ["AbacusSummit", "LRG", [0.4, 0.6]],
        #"LRG2": ["AbacusSummit", "LRG", [0.6, 0.8]],
        #"LRG3": ["AbacusSummit", "LRG", [0.8, 1.1]],
        #"ELG2": ["AbacusSummit", "ELG_LOPnotqso", [1.1, 1.6]],
        #"QSO": ["AbacusSummit", "QSO", [0.8, 2.1]],
    }

    # Unpack tracer info
    sim, name, (zlow, zhigh) = tracers[tracer]
        # Base path to mock files
    path_start = (
        f"/global/cfs/cdirs/desi/survey/catalogs/DA2/mocks/SecondGenMocks/{sim}/altmtl{mock}/kibo-v1/mock{mock}/LSScats"
    )

    # LOAD DATA AND RANDOMS
    # Load data
    data_rand_all = []


    if skyarea == "GCcomb":
               # Load NGC + SGC and combine them
        data_main_NGC = read_lss_table(f"{path_start}/{name}_NGC_clustering.dat.h5")
        data_main_SGC = read_lss_table(f"{path_start}/{name}_SGC_clustering.dat.h5")
                # Apply redshift cut and concatenate
        data_main = np.concatenate(
            [
                data_main_NGC[(data_main_NGC["Z"] > zlow) & (data_main_NGC["Z"] < zhigh)],
                data_main_SGC[(data_main_SGC["Z"] > zlow) & (data_main_SGC["Z"] < zhigh)],
            ]
        )
                # Load 3 random catalogues for both NGC and SGC
        for r in range(nrand):
            data_rand_NGC = read_lss_table(f"{path_start}/{name}_NGC_{r}_clustering.ran.h5")
            data_rand_SGC = read_lss_table(f"{path_start}/{name}_SGC_{r}_clustering.ran.h5")
            data_rand_all.append(
                np.concatenate(
                    [
                        data_rand_NGC[(data_rand_NGC["Z"] > zlow) & (data_rand_NGC["Z"] < zhigh)],
                        data_rand_SGC[(data_rand_SGC["Z"] > zlow) & (data_rand_SGC["Z"] < zhigh)],
                    ]
                )
            )
    else:
                # Single region (NGC or SGC)
        data_main = read_lss_table(f"{path_start}/{name}_{skyarea}_clustering.dat.h5")
        data_main = data_main[(data_main["Z"] > zlow) & (data_main["Z"] < zhigh)]
        for r in range(nrand):
            data_rand = read_lss_table(f"{path_start}/{name}_{skyarea}_{r}_clustering.ran.h5")
            data_rand_all.append(data_rand[(data_rand["Z"] > zlow) & (data_rand["Z"] < zhigh)])

    # Combine all randoms
    data_rand = np.concatenate(data_rand_all)

    # CONVERT RA/DEC/Z → 3D CARTESIAN COORDINATES
    # Calculate 3D comoving coordinates - FIXED: Convert to radians!
    ra_main_rad = np.radians(data_main["RA"])
    dec_main_rad = np.radians(data_main["DEC"])
    d = cosmo.comoving_distance(data_main["Z"])

        # Standard spherical → Cartesian conversion
    xx = d * np.cos(ra_main_rad) * np.cos(dec_main_rad)
    yy = d * np.sin(ra_main_rad) * np.cos(dec_main_rad)
    zz = d * np.sin(dec_main_rad)
    coords_main = np.stack((xx, yy, zz), axis=1)

       # Choose weights depending on sys_weight flag
    weights_main = data_main["WEIGHT"] * data_main["WEIGHT_FKP"] if sys_weight else data_main["WEIGHT_FKP"]

 # SUBSAMPLE RANDOMS ---------------- * Bug Fix * -------------------------------------------
    # Subsample randoms
    if sys_weight:
                # Remove randoms with WEIGHT=0 
        data_rand = data_rand[data_rand["WEIGHT"] > 0]
    # ----------------------------------------------------------------

# N_rand <= 10x N_data
    subsamp_index = np.random.choice(
        len(data_rand), 
        size=min(len(data_rand), int(10 * len(coords_main))), replace=False
        )
    print(f"Data: {len(data_main)}, Randoms: {len(subsamp_index)}")

    # Convert randoms to 3D coordinates
    ra_rand_rad = np.radians(data_rand["RA"][subsamp_index])
    dec_rand_rad = np.radians(data_rand["DEC"][subsamp_index])
    d = cosmo.comoving_distance(data_rand["Z"][subsamp_index])
    xx = d * np.cos(ra_rand_rad) * np.cos(dec_rand_rad)
    yy = d * np.sin(ra_rand_rad) * np.cos(dec_rand_rad)
    zz = d * np.sin(dec_rand_rad)
    coords_rand = np.stack((xx, yy, zz), axis=1)
    weights_rand = (
        data_rand["WEIGHT"][subsamp_index] * data_rand["WEIGHT_FKP"][subsamp_index]
        if sys_weight
        else data_rand["WEIGHT_FKP"][subsamp_index]
    )

    # BUILD KD-TREES FOR FAST NEIGHBOUR SEARCHES
    # Build KD trees
    tree_main = KDTree(coords_main)
    tree_rand = KDTree(coords_rand)

    # DEFINE RADII AND NORMALISATION
    # Define bins
    radii = np.arange(20, 250, 0.5)
    radius_bins = np.concatenate([[0.0], radii])

    # Ratio of total random weight to total data weight
    rand_density = np.sum(weights_rand) / np.sum(weights_main)


    print(f"Random density: {rand_density:.4f}")

    # SPLIT INTO BATCHES FOR PARALLEL PROCESSING
    # Set up batches - LARGER for better performance
    batch_size = 1000
    numbatches = int(len(coords_main) / batch_size) + 1
    begin_array = np.arange(0, len(coords_main), batch_size)
    end_array = begin_array + batch_size
    end_array[-1] = len(coords_main)

    start = time.time()

    # RUN ALL BATCHES IN PARALLEL
    # Process in parallel - use fully vectorized version
    with Pool(12) as pool:
        processed = pool.starmap(
            Nr,  # Use the fully vectorized version
            [
                [
                    begin,
                    end,
                    tree_main,
                    tree_rand,
                    coords_main,
                    weights_main,
                    weights_rand,
                    radii,
                    radius_bins,
                    rand_density,
                ]
                for (begin, end) in zip(begin_array, end_array)
            ],
        )
    pool.close()
    pool.join()

    # COMBINE RESULTS FROM ALL BATCHES
    # Combine results
    count_data = np.sum(np.vstack([process[0] for process in processed]), axis=0)
    MSC_data = np.sum(np.vstack([process[1] for process in processed]), axis=0)

    # Final MSC = weighted sum / number of contributing galaxies
    MSC_data /= count_data

    print(f"Computation time: {(time.time() - start)/60.0:.2f} minutes")

    # SAVE RESULTS
    # Save with single-line header containing metadata (pandas-compatible)
    header = f"rand_density={rand_density:.16f} tracer={tracer} skyarea={skyarea} mock={mock} weighted={sys_weight} n_data={len(coords_main)} n_random={len(coords_rand)}"

    # Save results
    extension = "weighted" if sys_weight else "unweighted"
    output_dir = f"/global/homes/s/sclark/DR2/desi/homogeneity/{tracer}/Abacus_DA2"
    msc_fname = f"{output_dir}/MSC_{tracer}_{skyarea}_mock{mock}_{extension}.txt"
    np.savetxt(msc_fname, np.c_[radii, MSC_data, count_data], fmt="%.16f", delimiter=" ", header=header, comments="# ")

    print(f"Saved MSC to {msc_fname}")
