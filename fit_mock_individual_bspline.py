# Compare homogeneity scale fits using different bin widths -- bspline
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from fit_mock_average_desilike import setup_fitting
from python_routines import get_mocks, run_bestfit, run_bestfit_spline, run_emcee, plot_poly_samples, TRACER_COLORS, lighten, get_abacus_dir
from matplotlib.patches import FancyBboxPatch

if "__main__" in __name__:

    # Parameters from command line

    for mock in range(25):
   # mock = None #average
        
        tracers, names, colors, templates, templates_real, rmins, rmaxs, model, rebin = setup_fitting()
        #model = "poly6"
        model = "bspline"
        #rmins = {"BGS1": 85.0, "BGS2": 85.0, "LRG1": 85.0, "LRG2": 85.0, "LRG3": 85.0, "ELG2": 65.0, "QSO": 100.0}
        #rmaxs = {"BGS1": 180.0, "BGS2": 180.0, "LRG1": 180.0, "LRG2": 180.0, "LRG3": 180.0, "ELG2": 160.0, "QSO": 195.0}
    
        # Fit the mock average of each tracer with a 3rd, 4th, 5th, 6th rm order polynomial
        for t, tracer in enumerate(["BGS1", "BGS2", "LRG1", "LRG2", "LRG3", "ELG2", "QSO"]):
    
            abacus_dir = get_abacus_dir(tracer)
            base = TRACER_COLORS[tracer]
            #SGC_color = lighten(base, 1.2)
            SGC_color=tuple(0.25 * np.array(to_rgb(base)) + 0.75 * np.array([1, 1, 1]))
    
            # Create ONE figure for this tracer
            fig = plt.figure()
            ax = fig.add_axes((0.13, 0.13, 0.85, 0.85))
            
            
            for skyarea in ["NGC", "SGC"]:
                Abacus, Holi = get_mocks(tracer, skyarea, "weighted", rebin=rebin)
    
                if skyarea == "NGC":
                    marker_face = base
                    marker_edge = "black"
                    error_color = "black"
                    alpha = 0.9
            
                else:
                    marker_face = SGC_color
                    marker_edge = base
                    error_color = base
                    alpha = 0.75
    
                samples = {}
                samples_filename = f"desi/homogeneity/{tracer}/CD_{abacus_dir}_{skyarea}_{mock}_singlecov_{model}_rebin{rebin}_samples_emcee.txt"
    
    
                if os.path.isfile(samples_filename):
    
                    samples[model] = pd.read_csv(
                        samples_filename,
                        sep=r"\s+",
                    )
                
                else:
                
                    # Get the best-fit B-spline
                    spline_params, chi2, splinefit, R_H = run_bestfit_spline(
                        Abacus,
                        rmin=rmins[tracer],
                        rmax=rmaxs[tracer],
                        datatype="D2",
                        plot=False,
                        single_cov=True,
                        mock=mock,
                    )
                    # extra test
                    print( tracer, skyarea, "n_coeffs =", len(spline_params[1]), "n_basis =", len(spline_params[0]) - spline_params[2] - 1)
                
                    print(tracer, skyarea, "Chi2 =", chi2, "R_H =", R_H)
                
                    try:
       
                
                        # Run emcee using the best-fit spline parameters
                        samples[model] = run_emcee(
                            Abacus,
                            spline_params,
                            model="bspline",
                            rmin=rmins[tracer],
                            rmax=rmaxs[tracer],
                            datatype="D2",
                            single_cov=True,
                            mock=mock,
                        )
                
                        # Save INCLUDING column names
                        samples[model].to_csv(
                            samples_filename,
                            sep=" ",
                            index=False,
                        )
                
                    except RuntimeError as e:
                        print(f"\nFAILED: {tracer} {skyarea} mock {mock}")
                        print(e)
                        continue
                        
                
                #plot_filename = f"desi/homogeneity/{tracer}/CD_{abacus_dir}_{skyarea}_{mock}_singlecov_{model}_rebin{rebin}.png"
                plot_filename = f"desi/homogeneity/{tracer}/CD_{abacus_dir}_{mock}_singlecov_{model}_rebin{rebin}_NGC_SGC.png"
                
                #plot_poly_samples(
                #    plot_filename,
                #    tracer,
                #    color=base if skyarea == "NGC" else SGC_color,
                #    name=names[t],
                #    data=Abacus,
                #    samples=samples[model],
                #    rmin=rmins[tracer],
                #    rmax=rmaxs[tracer],
                #    model=model,
                #    mock=mock,
                #    marker_face=marker_face,
                #    marker_edge=marker_edge,
                #    ax=ax,
                #    label=skyarea,
                #    alpha=alpha
                #)
    
            #box = FancyBboxPatch(
            #    (0.50, 0.01),       # x, y
            #    0.50,               # width
            #    0.30,               # height
            #    transform=ax.transAxes,
            #    boxstyle="round,pad=0.01,rounding_size=0.02",
            #    facecolor="white",
            #    edgecolor="black",
            #    linewidth=1.0,
            #    alpha=0.9,
            #    zorder=4,
            #)
            
            #ax.add_patch(box)
    
            
            # Put all text above the box
            #for txt in ax.texts:
            #    txt.set_zorder(5)
    
                # Save the COMPLETE combined figure
    #         fig.savefig(plot_filename, dpi=300, bbox_inches="tight")
    #         plt.close(fig)
