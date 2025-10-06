import lsst.drp.tasks.make_direct_warp
assert type(config) is lsst.drp.tasks.make_direct_warp.MakeDirectWarpConfig, f"config is of type {type(config).__module__}.{type(config).__name__} instead of lsst.drp.tasks.make_direct_warp.MakeDirectWarpConfig"

import lsst.afw.math._warper
import lsst.meas.algorithms
import lsst.meas.algorithms.cloughTocher2DInterpolator
import lsst.meas.base._id_generator
import lsst.obs.lsst._packer
import lsst.pipe.base._observation_dimension_packer
import lsst.pipe.base.config
import lsst.pipe.tasks.coaddInputRecorder
import lsst.pipe.tasks.selectImages
# Flag to enable/disable saving of log output for a task, enabled by default.
config.saveLogOutput=True

# Number of noise realizations to simulate and persist.
config.numberOfNoiseRealizations=0

# Offset to the seed used for the noise realization. This can be used to create a different noise realization if the default ones are catastrophic, or for testing sensitivity to the noise.
config.seedOffset=0

# Use the median of variance plane in the input calexp to generate noise realizations? If False, per-pixel variance will be used.
config.useMedianVariance=True

# Revert the old backgrounds from the `background_revert_list` connection?
config.doRevertOldBackground=False

# Apply the new backgrounds from the `background_apply_list` connection?
config.doApplyNewBackground=False

# Apply flat background ratio prior to background adjustments? Should be True if processing was done with an illumination correction.
config.doApplyFlatBackgroundRatio=True

# If True, use the PSF model and aperture corrections from the 'visit_summary' connection to make the warp. If False, use the PSF model and aperture corrections from the 'calexp' connection.
config.useVisitSummaryPsf=False

# If True, use the WCS from the 'visit_summary' connection to make the warp. If False, use the WCS from the 'calexp' connection.
config.useVisitSummaryWcs=False

# If True, use the photometric calibration from the 'visit_summary' connection to make the warp. If False, use the photometric calibration from the 'calexp' connection.
config.useVisitSummaryPhotoCalib=False

# Select ccds before warping?
config.doSelectPreWarp=True

# Flag to enable/disable saving of log output for a task, enabled by default.
config.select.saveLogOutput=True

# Maximum median ellipticity residual
config.select.maxEllipResidual=0.0055

# Maximum scatter in the size residuals
config.select.maxSizeScatter=None

# Maximum scatter in the size residuals, scaled by the median size
config.select.maxScaledSizeScatter=0.022

# Maximum delta (max - min) of model PSF trace radius values evaluated on a grid on the unmasked detector pixels (pixel).
config.select.maxPsfTraceRadiusDelta=4.4

# Maximum delta (max - min) of model PSF aperture flux (with aperture radius of max(2, 3*psfSigma)) values evaluated on a grid on the unmasked detector pixels (based on a normalized-to-one flux).
config.select.maxPsfApFluxDelta=1.6

# Maximum delta (max - min) of model PSF aperture correction values evaluated on a grid on the unmasked detector pixels scaled (divided) by the measured model psfSigma.
config.select.maxPsfApCorrSigmaScaledDelta=0.13

# Minimum number of PSF stars for the final PSF model to be considered well-constrained and suitible for inclusion in the coadd.  This number should take into consideration the spatial order used for the PSF model.  If the current band for the exposure is not included as a key in this dict, the value associated with the "fallback" key will be used.
config.select.minNPsfStarPerBand={'u': 6.0, 'g': 15.0, 'r': 15.0, 'i': 15.0, 'z': 15.0, 'y': 15.0, 'fallback': 6.0}

# Template parameter used to format corresponding field template parameter
config.select.connections.coaddName='deep'

# Interpolate over bad pixels before warping?
config.doPreWarpInterpolation=False

# List of mask planes to interpolate over.
config.preWarpInterpolation.badMaskPlanes=['BAD', 'SAT', 'CR']

# Constant value to fill outside of the convex hull of the good pixels. A long (longer than twice the ``interpLength``) streak of bad pixels at an edge will be set to this value.
config.preWarpInterpolation.fillValue=0.0

# Maximum number of pixels away from a bad pixel to include in building the interpolant. Must be greater than or equal to 1.
config.preWarpInterpolation.interpLength=4

# Whether to flip the x and y coordinates before constructing the Delaunay triangulation. This may produce a slightly different result since the triangulation is not guaranteed to be invariant under coordinate flips.
config.preWarpInterpolation.flipXY=True

# Add records for CCDs we iterated over but did not add a coaddTempExp due to a lack of unmasked pixels in the coadd footprint.
config.inputRecorder.saveEmptyCcds=False

# Add records for CCDs we iterated over but did not add a coaddTempExp due to an exception (often due to the calexp not being found on disk).
config.inputRecorder.saveErrorCcds=False

# Save the total number of good pixels in each coaddTempExp (redundant with a sum of good pixels in associated CCDs)
config.inputRecorder.saveVisitGoodPix=True

# Save weights in the CCDs table as well as the visits table? (This is necessary for easy construction of CoaddPsf, but otherwise duplicate information.)
config.inputRecorder.saveCcdWeights=True

# Pad the patch boundary of the warp by these many pixels, so as to allow for PSF-matching later
config.border=256

# Warping kernel
config.warper.warpingKernelName='lanczos3'

# Warping kernel for mask (use ``warpingKernelName`` if '')
config.warper.maskWarpingKernelName='bilinear'

# ``interpLength`` argument to `lsst.afw.math.warpExposure`
config.warper.interpLength=10

# ``cacheSize`` argument to `lsst.afw.math.SeparableKernel.computeCache`
config.warper.cacheSize=0

# mask bits to grow to full width of image/variance kernel,
config.warper.growFullMask=16

# Warp the masked fraction image?
config.doWarpMaskedFraction=False

# Warping kernel
config.maskedFractionWarper.warpingKernelName='bilinear'

# Warping kernel for mask (use ``warpingKernelName`` if '')
config.maskedFractionWarper.maskWarpingKernelName='bilinear'

# ``interpLength`` argument to `lsst.afw.math.warpExposure`
config.maskedFractionWarper.interpLength=10

# ``cacheSize`` argument to `lsst.afw.math.SeparableKernel.computeCache`
config.maskedFractionWarper.cacheSize=1000000

# mask bits to grow to full width of image/variance kernel,
config.maskedFractionWarper.growFullMask=16

# Warping kernel cache size
config.coaddPsf.cacheSize=10000

# Name of warping kernel; choices: lanczos3,lanczos4,lanczos5,bilinear,nearest
config.coaddPsf.warpingKernelName='lanczos3'

# Identifier for a data release or other version to embed in generated IDs. Zero is reserved for IDs with no embedded release identifier.
config.idGenerator.release_id=4

# Number of (contiguous, starting from zero) `release_id` values to reserve space for. One (not zero) is used to reserve no space.
config.idGenerator.n_releases=64

# Number of detectors, or, more precisely, one greater than the maximum detector ID, for this instrument. Default (None) obtains this value from the instrument dimension record. This should rarely need to be overridden outside of tests.
config.idGenerator.packer['observation'].n_detectors=None

# Number of observations (visits or exposures, as per 'is_exposure`) expected, or, more precisely, one greater than the maximum visit/exposure ID. Default (None) obtains this value from the instrument dimension record. This should rarely need to be overridden outside of tests.
config.idGenerator.packer['observation'].n_observations=None

# Mapping from controller code to integer.
config.idGenerator.packer['rubin'].controllers={'O': 0}

# Reserved number of controller codes.  May be larger than `len(controllers)`.
config.idGenerator.packer['rubin'].n_controllers=1

# Reserved number of visit definitions a single exposure may belong to.
config.idGenerator.packer['rubin'].n_visit_definitions=2

# Reserved number of distinct valid-date day_obs values, starting from `day_obs_begin`.
config.idGenerator.packer['rubin'].n_days=16384

# Reserved number of seq_num values, starting from 0.
config.idGenerator.packer['rubin'].n_seq_nums=32768

# Reserved number of detectors, starting from 0.
config.idGenerator.packer['rubin'].n_detectors=256

# Inclusive lower bound on day_obs.
config.idGenerator.packer['rubin'].day_obs_begin=20100101

config.idGenerator.packer.name=None
# name for connection calexp_list
config.connections.calexp_list='visit_image'

# name for connection background_revert_list
config.connections.background_revert_list='calexpBackground'

# name for connection background_apply_list
config.connections.background_apply_list='skyCorr'

# name for connection background_to_photometric_ratio_list
config.connections.background_to_photometric_ratio_list='background_to_photometric_ratio'

# name for connection visit_summary
config.connections.visit_summary='visit_summary'

# name for connection sky_map
config.connections.sky_map='skyMap'

# name for connection warp
config.connections.warp='direct_warp'

# name for connection masked_fraction_warp
config.connections.masked_fraction_warp='direct_warp_masked_fraction'

# Template parameter used to format corresponding field template parameter
config.connections.coaddName='deep'

