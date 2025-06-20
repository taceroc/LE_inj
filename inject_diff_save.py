import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle
from astropy.io import fits
import astropy.units as u
from astropy.table import Table, vstack
import astropy
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord
import getpass

import lsst.afw.display as afwDisplay
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
from lsst.daf.butler import Butler
import lsst.geom as geom

from lsst.source.injection import ingest_injection_catalog, generate_injection_catalog
from lsst.source.injection import VisitInjectConfig, VisitInjectTask

from lsst.ip.diffim.subtractImages import AlardLuptonSubtractTask, AlardLuptonSubtractConfig

from lsst.meas.algorithms.detection import SourceDetectionTask, SourceDetectionConfig
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig
import lsst.afw.table as afwTable

from lsst.meas.deblender import SourceDeblendTask, SourceDeblendConfig
from lsst.meas.base import SingleFrameMeasurementTask, SingleFrameMeasurementConfig
from lsst.meas.base import ForcedMeasurementTask

# ----- Load collection with step1 step2: the calexps
my_collection_identifier = 'test_512025_1_t4026p15y2022'
print(my_collection_identifier)

user = os.getenv("USER")
print(user)

my_outputCollection = f"u/{user}_pm/{my_collection_identifier}" 
print('Name of new butler collection for my output: ', my_outputCollection)

#load collection with calexp 
my_collection_identifier_step1step2 = 'test_512025_1_t4026p15y2022'
print(my_collection_identifier_step1step2)

collections_step1step2 = f"u/{user}_pm/{my_collection_identifier_step1step2}" 
print('Name of new butler collection for my output: ', collections_step1step2)

# load butler
repo = '/global/cfs/cdirs/lsst/production/gen3/DC2/Run2.2i/repo'
butler = Butler(repo, collections=collections_step1step2)

#list calexps
calexp_list = list(butler.registry.queryDataIds(
    ["tract", "patch", "visit", "detector"],
    instrument="LSSTCam-imSim",
    datasets="calexp",
    collections=collections_step1step2,))

#list visit table
visitTableRef = list(butler.registry.queryDatasets('visitTable'))

# load visit table
visitTable = pd.DataFrame()
for v in visitTableRef:
    visitTable_uni = butler.get(v)
    # visitTable_uni = visitTable_uni[visitTable_uni['band'] == 'g']
    visitTable = pd.concat([visitTable,visitTable_uni])

# load day_obs calexps
day_obds = [x['day_obs'] for x in calexp_list]

#organize dates
dates_calexps = pd.DataFrame()
dates_calexps.loc[:, 'year'] = [str(x)[:4] for x in day_obds]
dates_calexps.loc[:, 'month'] = [str(x)[4:6] for x in day_obds]
dates_calexps.loc[:, 'day'] = [str(x)[6:] for x in day_obds]
dates_calexps.loc[:, 'time'] = dates_calexps.loc[:, 'year']+'-'+dates_calexps.loc[:, 'month']+'-'+dates_calexps.loc[:, 'day']
dates_calexps.loc[:, 'time'] = dates_calexps.loc[:, 'time'].map(lambda x: Time(x))
dates_calexps.loc[:, 'mjd'] = dates_calexps.loc[:, 'time'].map(lambda x: x.mjd)
dates_calexps.loc[:, 'mjd'] = dates_calexps.loc[:, 'mjd'].map(lambda x: round(x,0))

# group by year and month, we want a coadd per month
dates_calexps_ym = dates_calexps.groupby(["year", "month"]).agg({'mjd':'first', 'time':'count'})
dates_calexps_ym.reset_index(inplace=True)


# -----Load coadds for each month pair given by the visit summary, same used to built the coadd
coadd_to_use = []
collection_to_use = []
for i in range(len(dates_calexps_ym)-1):
    print(dates_calexps_ym['month'][i], dates_calexps_ym['month'][i+1])
    list_collections = butler.registry.queryCollections(f"u/taceroc_pm/*_t4026p15y2022_coadd*{dates_calexps_ym['month'][i]}-{dates_calexps_ym['month'][i+1]}*")
    my_outputCollection = list_collections[-1]
    collection_to_use.append(my_outputCollection)
    butler = Butler(repo, collections=my_outputCollection)
    coadd_list = list(butler.registry.queryDatasets('deepCoadd', where="tract = 4026 AND patch = 15 AND skymap = 'DC2' AND band = 'r' "))
    if len(coadd_list) >= 1:
        coadd_to_use.append(coadd_list[0])


# ----- Load LEs simulations fits files
# list_fits = sorted(glob.glob("fits/4026_15/*52000.0pc.fits"))
# list_fits = sorted(glob.glob("fits/4026_15/ring/*52000.0pc.fits"))
# list_fits = sorted(glob.glob("fits/4026_15/ring_company/*52000.0pc.fits"))

# list_fits = sorted(glob.glob("fits/4026_15/*52000.0pc.fits"))
# list_fits = sorted(glob.glob("fits/4026_15/*52000.0pc.fits"))

def generate_injection(list_fits, ids_to_use_time, ids_to_use_time_fits = 1):
    # load metadata of LE
    list_metadata = []
    for lls in list_fits: 
        edits = lls.replace(list_fits[0].split('surface')[0], 'lightecho_modeling_oop/OOP/results/LC_infplane_test_multi/arrays/')
        edits = edits.replace('surface_values', 'meta_info')
        edits = edits.replace('fits', 'pkl')
        list_metadata.append(edits)
    
    
    # ---- Define injection catalog for 1st image
     # I still dont know how to 'remove' the coadds with empty
    coadd = butler.get(coadd_to_use[ids_to_use_time])
    boxcen = coadd.getBBox().getCenter()
    wcs_b = coadd.getWcs()
    cen = wcs_b.pixelToSky(boxcen)
    radec = SkyCoord(ra=cen[0].asDegrees()*u.deg, dec=cen[1].asDegrees()*u.deg)
    
    print(radec)
    
    imsize = coadd.getBBox().getDimensions()[0]*wcs_b.getPixelScale().asDegrees()
    print('Size of calexp in degrees: ', imsize)
    
    inject_size = imsize/2
    
    # ----- select fits for 1st image
    # ids_to_use_time_fits = 1 #ids_to_use_time
    which_fits = list_fits[ids_to_use_time_fits]
    fits_img = fits.open(which_fits)[0].data
    mags_plot = -2.5*np.log10(fits_img)-48.6
    mags_plot = np.nan_to_num(mags_plot, nan=0.0, posinf=0.0, neginf=0.0)
    ns = imsize/((mags_plot.shape[0])*wcs_b.getPixelScale().asDegrees())
    Ns = int((ns*ns) - 10)
    if Ns > 20:
        Ns = 20
    print("mean surface", np.mean(mags_plot[fits_img>0]))
    
    # ---- generate injection of LEs for 1st LE
    my_injection_catalog_LEs = generate_injection_catalog(
        ra_lim=[radec.ra.value-inject_size, radec.ra.value+inject_size],
        dec_lim=[radec.dec.value-inject_size, radec.dec.value+inject_size],
        number=Ns,
        seed='3210',
        source_type= "Stamp",
        mag=[np.mean(mags_plot[fits_img>0])],
        stamp= [which_fits],
    )
    
    first_ct = my_injection_catalog_LEs[0]['stamp'].split('ct')[1].split('_')[0]

    return my_injection_catalog_LEs, first_ct, list_metadata, coadd, wcs_b, radec, inject_size, Ns, np.mean(mags_plot[fits_img>0])-1

# list_fits = [sorted(glob.glob("fits/4026_15/ring/*52000.0pc.fits")), sorted(glob.glob("fits/4026_15/ring_company/*52000.0pc.fits"))]
list_fits = [sorted(glob.glob("fits/4026_15/2ndplane_closer/*52000.0pc.fits"))]
my_injection_catalog_LEs = []
ids_to_use_time = 3
ids_to_use_time_fits = 2
for llix in list_fits:
    # print(llix)
    my_injection_catalog_LEsix, first_ct, list_metadata, coadd, wcs_b, radec, inject_size, Ns, mag_source = generate_injection(llix, ids_to_use_time, ids_to_use_time_fits)
    my_injection_catalog_LEs.append(my_injection_catalog_LEsix)
    
# load metadata 1st image
with open(list_metadata[ids_to_use_time_fits], 'rb') as f:
    meta_data = pickle.load(f)

# find location of source to inject also a star there
resx = coadd.getBBox().getDimensions()[0]*wcs_b.getPixelScale().asDegrees()/coadd.getBBox().getDimensions()[0]
resy= coadd.getBBox().getDimensions()[1]*wcs_b.getPixelScale().asDegrees()/coadd.getBBox().getDimensions()[1]
center_shift = [resx*(meta_data[0]['act (arcsec)']/0.2), resy*(meta_data[0]['bct (arcsec)']/0.2)]

# ---- generate injection of source
my_injection_catalog_source = generate_injection_catalog(
    ra_lim=[radec.ra.value-inject_size, radec.ra.value+inject_size],
    dec_lim=[radec.dec.value-inject_size, radec.dec.value+inject_size],
    number=Ns,
    seed='3210',
    source_type= "Star",
    mag=[mag_source], #the mag could be extracted 
)

my_injection_catalog_source['ra'] = my_injection_catalog_source['ra']-center_shift[0]
my_injection_catalog_source['dec'] = my_injection_catalog_source['dec']+center_shift[1]


def do_injections(coadd, injection_catalog):#my_injection_catalog_LEs, my_injection_catalog_source):
    inject_config = VisitInjectConfig()
    inject_task = VisitInjectTask(config=inject_config)

    psf = coadd.getPsf()
    photo_calib = coadd.getPhotoCalib()
    wcs_b = coadd.getWcs()
    
    injected_output = inject_task.run(
        injection_catalogs=[*injection_catalog],#[my_injection_catalog_LEs, my_injection_catalog_source],
        input_exposure=coadd.clone(),
        psf=psf,
        photo_calib=photo_calib,
        wcs=wcs_b,
    )
    injected_coadd_1st = injected_output.output_exposure
    injected_catalog = injected_output.output_catalog


    return injected_coadd_1st
    
injected_coadd_1st = do_injections(coadd, [*my_injection_catalog_LEs, my_injection_catalog_source])

# ---- Define injection catalog for 2nd image

ids_to_use_time = 4
# ----- select fits for 2nd image
list_fits = [sorted(glob.glob("fits/4026_15/2ndplane_closer_otherarc/*52000.0pc.fits"))]
my_injection_catalog_LEs = []
for llix in list_fits:
    my_injection_catalog_LEsix, second_ct, _, coadd, _, _, _, _, _ = generate_injection(llix, ids_to_use_time, ids_to_use_time_fits = 3)
    my_injection_catalog_LEs.append(my_injection_catalog_LEsix)


# use same catalog for source
injected_coadd_2nd = do_injections(coadd, [*my_injection_catalog_LEs, my_injection_catalog_source])



# ----- Do source detection on the injected images

def do_source_detection_injections(injected_coadd_2nd):

    schema = afwTable.SourceTable.makeMinimalSchema()
    print(schema)
    raerr = schema.addField("coord_raErr", type="F")
    decerr = schema.addField("coord_decErr", type="F")
    
    config = CharacterizeImageConfig()
    config.psfIterations = 3
    charImageTask = CharacterizeImageTask(config=config)
    del config
    
    config = SourceDetectionConfig()
    config.thresholdValue = 10
    sourceDetectionTask = SourceDetectionTask(schema=schema, config=config)
    
    config = SourceDeblendConfig()
    sourceDeblendTask = SourceDeblendTask(schema=schema, config=config)
    
    config = SingleFrameMeasurementConfig()
    sourceMeasurementTask = SingleFrameMeasurementTask(schema=schema,
                                                       config=config)
    
    tab = afwTable.SourceTable.make(schema)
    result = charImageTask.run(injected_coadd_2nd)
    result = sourceDetectionTask.run(tab, injected_coadd_2nd)
    sources = result.sources
    sourceDeblendTask.run(injected_coadd_2nd, sources)
    sourceMeasurementTask.run(measCat=sources, exposure=injected_coadd_2nd)
    sources = sources.copy(True)

    return sources

sources = do_source_detection_injections(injected_coadd_2nd)


def image_subtraction(injected_coadd_1st, injected_coadd_2nd, sources):
    config = AlardLuptonSubtractConfig()
    
    alTask = AlardLuptonSubtractTask(config=config)
    
    # result = alTask.run(injected_coadd_1st, injected_coadd_2nd, src_catalog[0])
    result = alTask.run(injected_coadd_1st, injected_coadd_2nd, sources)
    return result

subtraction_outputs = image_subtraction(injected_coadd_1st, injected_coadd_2nd, sources)

# ----- save cutouts

import astropy.visualization as aviz
import matplotlib
matplotlib.use("AGG")
# Force matplotlib defaults
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import matplotlib.pyplot as plt
from matplotlib import cm
import io

def plot_one_image(ax, data, size, scale, name=None):
    """Plot a normalized image on an axis."""
    if name == "Difference":
        norm = aviz.ImageNormalize(
            # focus on a rect of dim 15 at the center of the image.
            data[data.shape[0] // 2 - 7:data.shape[0] // 2 + 8,
                 data.shape[1] // 2 - 7:data.shape[1] // 2 + 8],
            interval=aviz.MinMaxInterval(),
            stretch=aviz.AsinhStretch(a=0.1),
        )
    else:
        norm = aviz.ImageNormalize(
            data,
            interval=aviz.MinMaxInterval(),
            stretch=aviz.AsinhStretch(a=0.1),
        )
    ax.imshow(data, cmap=cm.bone, interpolation="none", norm=norm,
              extent=(0, size, 0, size), origin="lower", aspect="equal")
    x_line = 1
    y_line = 1
    ax.plot((x_line, x_line + 1.0/scale), (y_line, y_line), color="blue", lw=6)
    ax.plot((x_line, x_line + 1.0/scale), (y_line, y_line), color="yellow", lw=2)
    ax.axis("off")
    if name is not None:
        ax.set_title(name)

path_name_ids = f"{coadd_to_use[ids_to_use_time].dataId['tract']}_{coadd_to_use[ids_to_use_time].dataId['patch']}_{coadd_to_use[ids_to_use_time].dataId['band']}"
path_name = 'loc'+list_fits[-1][-1].split('loc')[1].replace('.fits', '')

os.makedirs(f"numpy/{path_name_ids}", exist_ok=True)
os.makedirs(f"numpy/{path_name_ids}/2ndplane_closer_otherarc/{path_name}", exist_ok=True)

path_to_numpy = f"numpy/{path_name_ids}/2ndplane_closer_otherarc/{path_name}"

os.makedirs(f"images/{path_name_ids}", exist_ok=True)
os.makedirs(f"images/{path_name_ids}/2ndplane_closer_otherarc/{path_name}", exist_ok=True)

path_to_images = f"images/{path_name_ids}/2ndplane_closer_otherarc/{path_name}"

numpy_cutouts = {}
for row in my_injection_catalog_LEs[-1]:
    try:
        center = wcs_b.skyToPixel(geom.SpherePoint(row['ra']*geom.degrees, row['dec']*geom.degrees))
        s = 200
        extent = geom.Extent2I(s, s)
        science_cutout = subtraction_outputs.matchedScience.getCutout(center, extent)
        template_cutout = injected_coadd_1st.getCutout(center, extent)
        difference_cutout = subtraction_outputs.difference.getCutout(center, extent)
        dia_source_id = row['injection_id']
        # self.numpy_path.mkdir(dia_source_id)
        numpy_cutouts[f"sci_{s}"] = science_cutout.image.array
        numpy_cutouts[f"temp_{s}"] = template_cutout.image.array
        numpy_cutouts[f"diff_{s}"] = difference_cutout.image.array
    
        scale = science_cutout.wcs.getPixelScale(science_cutout.getBBox().getCenter()).asArcseconds()
        fig, axs = plt.subplots(1, 3, constrained_layout=True)
        plot_one_image(axs[0], template_cutout.image.array, s, scale, "1st inj")
        plot_one_image(axs[1], science_cutout.image.array, s, scale, "2nd inj")
        plot_one_image(axs[2], difference_cutout.image.array, s, scale, "Difference")
        
        output = io.BytesIO()
        # plt.show()
        outfile_img = f'{path_to_images}/{dia_source_id}_{s}_{first_ct}-{second_ct}.png'
        plt.savefig(outfile_img, bbox_inches="tight", format="png")
        output.seek(0)
    
    
        for cutout_type, cutout in numpy_cutouts.items():
            outfile = f'{path_to_numpy}/{dia_source_id}_{cutout_type}_{first_ct}-{second_ct}.npy'
            np.save(outfile, np.expand_dims(cutout, axis=0))
    except Exception as ex:
        print(ex)
        continue



