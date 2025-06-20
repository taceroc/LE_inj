# Inject LE simulation on DC2 images

using the `lsst_stack = v27.0.0` on NERSC.

## How to create the DC2 images?
* `job_step1step2.sh` : creates the calexp by running step1 and step2 from the pipeline for a specific year
* `job_coadd_measure.sh`: creates the coadds by calling the `run_coadd.py` that executes some parts of the step3 of the pipeline


## How to inject the LE simulations on the coadds?

### First, generate the LE image, and save it as a fits file 

The [repo](https://github.com/taceroc/lightecho_modeling_oop) on branch plane_simple_ners, contains the code to generate the LE.
(this doesn't need the lsst_stack)
```
python pipe_le.py SimulateLEInfPlane -file_to_parameters /LE_inj/params_le_for_4026_15_2022_y.yml --bool_save --no-bool_show_plots --no-bool_show_initial_object -file_to_parameters_surface /LE_inj/name_surface.yml -loc_to_fits /LE_inj/fits/4026_15/2ndplane_closer
```

#### Arguments:
* `-file_to_parameters`: the yml file that specifies the geometry of the plane, location of the source.
* `-file_to_parameters_surface`: just a temporary file to store the name of the surface's values, after the simulation
* `-loc_to_fits`: where to save the .fits file

### Second, inject the LE on the coadds

`python inject_diff_save.py`: This is still very manual, define which fits file to use, which coadd to use by editing the code

You have to load first the `lsst_stack`
```
export STACKCVMFS=/cvmfs/sw.lsst.eu/linux-x86_64/lsst_distrib
export LSST_STACK_VERSION=v27.0.0

module load cpu

source $STACKCVMFS/$LSST_STACK_VERSION/loadLSST-ext.bash
setup lsst_distrib
```

This saves the numpy array of the two injected images and the difference, and also a .jpg image with the three images.

