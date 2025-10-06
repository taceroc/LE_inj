import lsst.drp.tasks.assemble_coadd
assert type(config) is lsst.drp.tasks.assemble_coadd.CompareWarpAssembleCoaddConfig, f"config is of type {type(config).__module__}.{type(config).__name__} instead of lsst.drp.tasks.assemble_coadd.CompareWarpAssembleCoaddConfig"

import lsst.meas.algorithms
import lsst.meas.algorithms.detection
import lsst.meas.algorithms.gaussianPsfFactory
import lsst.meas.algorithms.maskStreaks
import lsst.meas.algorithms.scaleVariance
import lsst.meas.algorithms.subtractBackground
import lsst.pipe.base.config
import lsst.pipe.tasks.coaddInputRecorder
import lsst.pipe.tasks.healSparseMapping
import lsst.pipe.tasks.interpImage
import lsst.pipe.tasks.scaleZeroPoint
import lsst.pipe.tasks.selectImages
# Flag to enable/disable saving of log output for a task, enabled by default.
config.saveLogOutput=True

# Coadd name: typically one of deep or goodSeeing.
config.coaddName='deep'

# Flag to enable/disable saving of log output for a task, enabled by default.
config.select.saveLogOutput=True

# Maximum median ellipticity residual
config.select.maxEllipResidual=0.007

# Maximum scatter in the size residuals
config.select.maxSizeScatter=None

# Maximum scatter in the size residuals, scaled by the median size
config.select.maxScaledSizeScatter=0.019

# Maximum delta (max - min) of model PSF trace radius values evaluated on a grid on the unmasked detector pixels (pixel).
config.select.maxPsfTraceRadiusDelta=0.7

# Maximum delta (max - min) of model PSF aperture flux (with aperture radius of max(2, 3*psfSigma)) values evaluated on a grid on the unmasked detector pixels (based on a normalized-to-one flux).
config.select.maxPsfApFluxDelta=0.24

# Maximum delta (max - min) of model PSF aperture correction values evaluated on a grid on the unmasked detector pixels scaled (divided) by the measured model psfSigma.
config.select.maxPsfApCorrSigmaScaledDelta=0.22

# Minimum number of PSF stars for the final PSF model to be considered well-constrained and suitible for inclusion in the coadd.  This number should take into consideration the spatial order used for the PSF model.  If the current band for the exposure is not included as a key in this dict, the value associated with the "fallback" key will be used.
config.select.minNPsfStarPerBand={'u': 6.0, 'g': 15.0, 'r': 15.0, 'i': 15.0, 'z': 15.0, 'y': 15.0, 'fallback': 6.0}

# Template parameter used to format corresponding field template parameter
config.select.connections.coaddName='deep'

# Mask planes that, if set, the associated pixel should not be included in the coaddTempExp.
config.badMaskPlanes=['NO_DATA', 'BAD', 'SAT', 'SUSPECT']

# Add records for CCDs we iterated over but did not add a coaddTempExp due to a lack of unmasked pixels in the coadd footprint.
config.inputRecorder.saveEmptyCcds=False

# Add records for CCDs we iterated over but did not add a coaddTempExp due to an exception (often due to the calexp not being found on disk).
config.inputRecorder.saveErrorCcds=False

# Save the total number of good pixels in each coaddTempExp (redundant with a sum of good pixels in associated CCDs)
config.inputRecorder.saveVisitGoodPix=True

# Save weights in the CCDs table as well as the visits table? (This is necessary for easy construction of CoaddPsf, but otherwise duplicate information.)
config.inputRecorder.saveCcdWeights=True

# Warp name: one of 'direct' or 'psfMatched'
config.warpType='direct'

# Width, height of stack subregion size; make small enough that a full stack of images will fit into memory  at once.
config.subregionSize=[10000, 100]

# Main stacking statistic for aggregating over the epochs.
config.statistic='MEAN'

# Perform online coaddition when statistic="MEAN" to save memory?
config.doOnlineForMean=False

# Sigma for outlier rejection; ignored if non-clipping statistic selected.
config.sigmaClip=3.0

# Number of iterations of outlier rejection; ignored if non-clipping statistic selected.
config.clipIter=2

# Calculate coadd variance from input variance by stacking statistic. Passed to StatisticsControl.setCalcErrorFromInputVariance()
config.calcErrorFromInputVariance=True

# desired photometric zero point
config.scaleZeroPoint.zeroPoint=27.0

# Interpolate over NaN pixels? Also extrapolate, if necessary, but the results are ugly.
config.doInterp=True

# Kernel size (width and height) (pixels); if None then sizeFactor is used
config.interpImage.modelPsf.size=None

# Kernel size as a factor of fwhm (dimensionless); size = sizeFactor * fwhm; ignored if size is not None
config.interpImage.modelPsf.sizeFactor=3.0

# Minimum kernel size if using sizeFactor (pixels); ignored if size is not None
config.interpImage.modelPsf.minSize=5

# Maximum kernel size if using sizeFactor (pixels); ignored if size is not None
config.interpImage.modelPsf.maxSize=None

# Default FWHM of Gaussian model of core of star (pixels)
config.interpImage.modelPsf.defaultFwhm=3.0

# Add a Gaussian to represent wings?
config.interpImage.modelPsf.addWing=True

# wing width, as a multiple of core width (dimensionless); ignored if addWing false
config.interpImage.modelPsf.wingFwhmFactor=2.5

# wing amplitude, as a multiple of core amplitude (dimensionless); ignored if addWing false
config.interpImage.modelPsf.wingAmplitude=0.1

# Smoothly taper to the fallback value at the edge of the image?
config.interpImage.useFallbackValueAtEdge=True

# Type of statistic to calculate edge fallbackValue for interpolation
config.interpImage.fallbackValueType='MEDIAN'

# If fallbackValueType is 'USER' then use this as the fallbackValue; ignored otherwise
config.interpImage.fallbackUserValue=0.0

# Allow negative values for egde interpolation fallbackValue?  If False, set fallbackValue to max(fallbackValue, 0.0)
config.interpImage.negativeFallbackAllowed=False

# Transpose image before interpolating? This allows the interpolation to act over columns instead of rows.
config.interpImage.transpose=0

# Persist coadd?
config.doWrite=True

# Persist artifact masks? Should be True for CompareWarp only.
config.doWriteArtifactMasks=True

# Create image of number of contributing exposures for each pixel
config.doNImage=True

# Threshold (in fractional weight) of rejection at which we propagate a mask plane to the coadd; that is, we set the mask bit on the coadd if the fraction the rejected frames would have contributed exceeds this value.
config.maskPropagationThresholds={'SAT': 0.1}

# Mask planes to remove before coadding
config.removeMaskPlanes=['NOT_DEBLENDED', 'EDGE', 'CROSSTALK']

# Set mask and flag bits for bright objects?
config.doMaskBrightObjects=False

# Name of mask bit used for bright objects
config.brightObjectMaskName='BRIGHT_OBJECT'

# Warping kernel cache size
config.coaddPsf.cacheSize=0

# Name of warping kernel; choices: lanczos3,lanczos4,lanczos5,bilinear,nearest
config.coaddPsf.warpingKernelName='lanczos3'

# Attach a piecewise TransmissionCurve for the coadd? (requires all input Exposures to have TransmissionCurves).
config.doAttachTransmissionCurve=False

# Should be set to True if fake sources have been inserted into the input data.
config.hasFakes=False

# Coadd only visits selected by a SelectVisitsTask
config.doSelectVisits=True

# Create a bitwise map of coadd inputs
config.doInputMap=True

# Mapping healpix nside.  Must be power of 2.
config.inputMapper.nside=32768

# HealSparse coverage map nside.  Must be power of 2.
config.inputMapper.nside_coverage=256

# Minimum area fraction of a map healpixel pixel that must be covered by bad pixels to be removed from the input map. This is approximate.
config.inputMapper.bad_mask_min_coverage=0.5

# name for connection inputWarps
config.connections.inputWarps='direct_warp'

# name for connection skyMap
config.connections.skyMap='skyMap'

# name for connection selectedVisits
config.connections.selectedVisits='deep_coadd_visit_selection'

# name for connection brightObjectMask
config.connections.brightObjectMask='brightObjectMask'

# name for connection coaddExposure
config.connections.coaddExposure='deep_coadd_predetection'

# name for connection nImage
config.connections.nImage='deep_coadd_n_image'

# name for connection inputMap
config.connections.inputMap='deep_coadd_input_map'

# name for connection psfMatchedWarps
config.connections.psfMatchedWarps='psf_matched_warp'

# name for connection templateCoadd
config.connections.templateCoadd='deep_coadd_compare_template'

# name for connection artifactMasks
config.connections.artifactMasks='compare_warp_artifact_mask'

# Template parameter used to format corresponding field template parameter
config.connections.inputCoaddName='deep'

# Template parameter used to format corresponding field template parameter
config.connections.outputCoaddName='deep'

# Template parameter used to format corresponding field template parameter
config.connections.warpType='direct'

# Template parameter used to format corresponding field template parameter
config.connections.warpTypeSuffix=''

# Flag to enable/disable saving of log output for a task, enabled by default.
config.assembleStaticSkyModel.saveLogOutput=True

# Coadd name: typically one of deep or goodSeeing.
config.assembleStaticSkyModel.coaddName='deep'

# Flag to enable/disable saving of log output for a task, enabled by default.
config.assembleStaticSkyModel.select.saveLogOutput=True

# Maximum median ellipticity residual
config.assembleStaticSkyModel.select.maxEllipResidual=0.007

# Maximum scatter in the size residuals
config.assembleStaticSkyModel.select.maxSizeScatter=None

# Maximum scatter in the size residuals, scaled by the median size
config.assembleStaticSkyModel.select.maxScaledSizeScatter=0.019

# Maximum delta (max - min) of model PSF trace radius values evaluated on a grid on the unmasked detector pixels (pixel).
config.assembleStaticSkyModel.select.maxPsfTraceRadiusDelta=0.7

# Maximum delta (max - min) of model PSF aperture flux (with aperture radius of max(2, 3*psfSigma)) values evaluated on a grid on the unmasked detector pixels (based on a normalized-to-one flux).
config.assembleStaticSkyModel.select.maxPsfApFluxDelta=0.24

# Maximum delta (max - min) of model PSF aperture correction values evaluated on a grid on the unmasked detector pixels scaled (divided) by the measured model psfSigma.
config.assembleStaticSkyModel.select.maxPsfApCorrSigmaScaledDelta=0.22

# Minimum number of PSF stars for the final PSF model to be considered well-constrained and suitible for inclusion in the coadd.  This number should take into consideration the spatial order used for the PSF model.  If the current band for the exposure is not included as a key in this dict, the value associated with the "fallback" key will be used.
config.assembleStaticSkyModel.select.minNPsfStarPerBand={'u': 6.0, 'g': 15.0, 'r': 15.0, 'i': 15.0, 'z': 15.0, 'y': 15.0, 'fallback': 6.0}

# Template parameter used to format corresponding field template parameter
config.assembleStaticSkyModel.select.connections.coaddName='deep'

# Mask planes that, if set, the associated pixel should not be included in the coaddTempExp.
config.assembleStaticSkyModel.badMaskPlanes=['NO_DATA']

# Add records for CCDs we iterated over but did not add a coaddTempExp due to a lack of unmasked pixels in the coadd footprint.
config.assembleStaticSkyModel.inputRecorder.saveEmptyCcds=False

# Add records for CCDs we iterated over but did not add a coaddTempExp due to an exception (often due to the calexp not being found on disk).
config.assembleStaticSkyModel.inputRecorder.saveErrorCcds=False

# Save the total number of good pixels in each coaddTempExp (redundant with a sum of good pixels in associated CCDs)
config.assembleStaticSkyModel.inputRecorder.saveVisitGoodPix=True

# Save weights in the CCDs table as well as the visits table? (This is necessary for easy construction of CoaddPsf, but otherwise duplicate information.)
config.assembleStaticSkyModel.inputRecorder.saveCcdWeights=True

# Warp name: one of 'direct' or 'psfMatched'
config.assembleStaticSkyModel.warpType='psfMatched'

# Width, height of stack subregion size; make small enough that a full stack of images will fit into memory  at once.
config.assembleStaticSkyModel.subregionSize=[10000, 100]

# Main stacking statistic for aggregating over the epochs.
config.assembleStaticSkyModel.statistic='MEANCLIP'

# Perform online coaddition when statistic="MEAN" to save memory?
config.assembleStaticSkyModel.doOnlineForMean=False

# Sigma for outlier rejection; ignored if non-clipping statistic selected.
config.assembleStaticSkyModel.sigmaClip=2.5

# Number of iterations of outlier rejection; ignored if non-clipping statistic selected.
config.assembleStaticSkyModel.clipIter=3

# Calculate coadd variance from input variance by stacking statistic. Passed to StatisticsControl.setCalcErrorFromInputVariance()
config.assembleStaticSkyModel.calcErrorFromInputVariance=False

# desired photometric zero point
config.assembleStaticSkyModel.scaleZeroPoint.zeroPoint=27.0

# Interpolate over NaN pixels? Also extrapolate, if necessary, but the results are ugly.
config.assembleStaticSkyModel.doInterp=True

# Kernel size (width and height) (pixels); if None then sizeFactor is used
config.assembleStaticSkyModel.interpImage.modelPsf.size=None

# Kernel size as a factor of fwhm (dimensionless); size = sizeFactor * fwhm; ignored if size is not None
config.assembleStaticSkyModel.interpImage.modelPsf.sizeFactor=3.0

# Minimum kernel size if using sizeFactor (pixels); ignored if size is not None
config.assembleStaticSkyModel.interpImage.modelPsf.minSize=5

# Maximum kernel size if using sizeFactor (pixels); ignored if size is not None
config.assembleStaticSkyModel.interpImage.modelPsf.maxSize=None

# Default FWHM of Gaussian model of core of star (pixels)
config.assembleStaticSkyModel.interpImage.modelPsf.defaultFwhm=3.0

# Add a Gaussian to represent wings?
config.assembleStaticSkyModel.interpImage.modelPsf.addWing=True

# wing width, as a multiple of core width (dimensionless); ignored if addWing false
config.assembleStaticSkyModel.interpImage.modelPsf.wingFwhmFactor=2.5

# wing amplitude, as a multiple of core amplitude (dimensionless); ignored if addWing false
config.assembleStaticSkyModel.interpImage.modelPsf.wingAmplitude=0.1

# Smoothly taper to the fallback value at the edge of the image?
config.assembleStaticSkyModel.interpImage.useFallbackValueAtEdge=True

# Type of statistic to calculate edge fallbackValue for interpolation
config.assembleStaticSkyModel.interpImage.fallbackValueType='MEDIAN'

# If fallbackValueType is 'USER' then use this as the fallbackValue; ignored otherwise
config.assembleStaticSkyModel.interpImage.fallbackUserValue=0.0

# Allow negative values for egde interpolation fallbackValue?  If False, set fallbackValue to max(fallbackValue, 0.0)
config.assembleStaticSkyModel.interpImage.negativeFallbackAllowed=False

# Transpose image before interpolating? This allows the interpolation to act over columns instead of rows.
config.assembleStaticSkyModel.interpImage.transpose=0

# Persist coadd?
config.assembleStaticSkyModel.doWrite=False

# Persist artifact masks? Should be True for CompareWarp only.
config.assembleStaticSkyModel.doWriteArtifactMasks=False

# Create image of number of contributing exposures for each pixel
config.assembleStaticSkyModel.doNImage=False

# Threshold (in fractional weight) of rejection at which we propagate a mask plane to the coadd; that is, we set the mask bit on the coadd if the fraction the rejected frames would have contributed exceeds this value.
config.assembleStaticSkyModel.maskPropagationThresholds={'SAT': 0.1}

# Mask planes to remove before coadding
config.assembleStaticSkyModel.removeMaskPlanes=['NOT_DEBLENDED']

# Set mask and flag bits for bright objects?
config.assembleStaticSkyModel.doMaskBrightObjects=False

# Name of mask bit used for bright objects
config.assembleStaticSkyModel.brightObjectMaskName='BRIGHT_OBJECT'

# Warping kernel cache size
config.assembleStaticSkyModel.coaddPsf.cacheSize=0

# Name of warping kernel; choices: lanczos3,lanczos4,lanczos5,bilinear,nearest
config.assembleStaticSkyModel.coaddPsf.warpingKernelName='lanczos3'

# Attach a piecewise TransmissionCurve for the coadd? (requires all input Exposures to have TransmissionCurves).
config.assembleStaticSkyModel.doAttachTransmissionCurve=False

# Should be set to True if fake sources have been inserted into the input data.
config.assembleStaticSkyModel.hasFakes=False

# Coadd only visits selected by a SelectVisitsTask
config.assembleStaticSkyModel.doSelectVisits=True

# Create a bitwise map of coadd inputs
config.assembleStaticSkyModel.doInputMap=False

# Mapping healpix nside.  Must be power of 2.
config.assembleStaticSkyModel.inputMapper.nside=32768

# HealSparse coverage map nside.  Must be power of 2.
config.assembleStaticSkyModel.inputMapper.nside_coverage=256

# Minimum area fraction of a map healpixel pixel that must be covered by bad pixels to be removed from the input map. This is approximate.
config.assembleStaticSkyModel.inputMapper.bad_mask_min_coverage=0.5

# name for connection inputWarps
config.assembleStaticSkyModel.connections.inputWarps='{inputCoaddName}Coadd_{warpType}Warp'

# name for connection skyMap
config.assembleStaticSkyModel.connections.skyMap='skyMap'

# name for connection selectedVisits
config.assembleStaticSkyModel.connections.selectedVisits='{outputCoaddName}Visits'

# name for connection brightObjectMask
config.assembleStaticSkyModel.connections.brightObjectMask='brightObjectMask'

# name for connection coaddExposure
config.assembleStaticSkyModel.connections.coaddExposure='{outputCoaddName}Coadd{warpTypeSuffix}'

# name for connection nImage
config.assembleStaticSkyModel.connections.nImage='{outputCoaddName}Coadd_nImage'

# name for connection inputMap
config.assembleStaticSkyModel.connections.inputMap='{outputCoaddName}Coadd_inputMap'

# Template parameter used to format corresponding field template parameter
config.assembleStaticSkyModel.connections.inputCoaddName='deep'

# Template parameter used to format corresponding field template parameter
config.assembleStaticSkyModel.connections.outputCoaddName='deep'

# Template parameter used to format corresponding field template parameter
config.assembleStaticSkyModel.connections.warpType='psfMatched'

# Template parameter used to format corresponding field template parameter
config.assembleStaticSkyModel.connections.warpTypeSuffix=''

# detected sources with fewer than the specified number of pixels will be ignored
config.detect.minPixels=4

# Grow pixels as isotropically as possible? If False, use a Manhattan metric instead.
config.detect.isotropicGrow=True

# Grow all footprints at the same time? This allows disconnected footprints to merge.
config.detect.combinedGrow=True

# Grow detections by nSigmaToGrow * [PSF RMS width]; if 0 then do not grow
config.detect.nSigmaToGrow=0.4

# Grow detections to set the image mask bits, but return the original (not-grown) footprints
config.detect.returnOriginalFootprints=False

# Threshold for detecting footprints; exact meaning and units depend on thresholdType.
config.detect.thresholdValue=5.0

# Multiplier on thresholdValue for whether a source is included in the output catalog. For example, thresholdValue=5, includeThresholdMultiplier=10, thresholdType='pixel_stdev' results in a catalog of sources at >50 sigma with the detection mask and footprints including pixels >5 sigma.
config.detect.includeThresholdMultiplier=1.0

# Specifies the meaning of thresholdValue.
config.detect.thresholdType='pixel_stdev'

# Specifies whether to detect positive, or negative sources, or both.
config.detect.thresholdPolarity='both'

# Fiddle factor to add to the background; debugging only
config.detect.adjustBackground=0.0

# Estimate the background again after final source detection?
config.detect.reEstimateBackground=False

# Convert from a photometrically flat image to one suitable for background subtraction? Only used if reEstimateBackground is True.If True, then a backgroundToPhotometricRatio must be supplied to the task run method.
config.detect.doApplyFlatBackgroundRatio=False

# type of statistic to use for grid points
config.detect.background.statisticsProperty='MEANCLIP'

# behaviour if there are too few points in grid for requested interpolation style
config.detect.background.undersampleStyle='REDUCE_INTERP_ORDER'

# how large a region of the sky should be used for each background point
config.detect.background.binSize=128

# Sky region size to be used for each background point in X direction. If 0, the binSize config is used.
config.detect.background.binSizeX=0

# Sky region size to be used for each background point in Y direction. If 0, the binSize config is used.
config.detect.background.binSizeY=0

# how to interpolate the background values. This maps to an enum; see afw::math::Background
config.detect.background.algorithm='AKIMA_SPLINE'

# Names of mask planes to ignore while estimating the background
config.detect.background.ignoredPixelMask=['BAD', 'EDGE', 'DETECTED', 'DETECTED_NEGATIVE', 'NO_DATA']

# Ignore NaNs when estimating the background
config.detect.background.isNanSafe=False

# Use Approximate (Chebyshev) to model background.
config.detect.background.useApprox=True

# Approximation order in X for background Chebyshev (valid only with useApprox=True)
config.detect.background.approxOrderX=6

# Approximation order in Y for background Chebyshev (valid only with useApprox=True)
config.detect.background.approxOrderY=-1

# Use inverse variance weighting in calculation (valid only with useApprox=True)
config.detect.background.weighting=True

# Convert from a photometrically flat image to one suitable to background subtraction? If True, then a backgroundToPhotometricRatio must be supplied to the task run method.
config.detect.background.doApplyFlatBackgroundRatio=False

# type of statistic to use for grid points
config.detect.tempLocalBackground.statisticsProperty='MEANCLIP'

# behaviour if there are too few points in grid for requested interpolation style
config.detect.tempLocalBackground.undersampleStyle='REDUCE_INTERP_ORDER'

# how large a region of the sky should be used for each background point
config.detect.tempLocalBackground.binSize=64

# Sky region size to be used for each background point in X direction. If 0, the binSize config is used.
config.detect.tempLocalBackground.binSizeX=0

# Sky region size to be used for each background point in Y direction. If 0, the binSize config is used.
config.detect.tempLocalBackground.binSizeY=0

# how to interpolate the background values. This maps to an enum; see afw::math::Background
config.detect.tempLocalBackground.algorithm='AKIMA_SPLINE'

# Names of mask planes to ignore while estimating the background
config.detect.tempLocalBackground.ignoredPixelMask=['BAD', 'EDGE', 'DETECTED', 'DETECTED_NEGATIVE', 'NO_DATA']

# Ignore NaNs when estimating the background
config.detect.tempLocalBackground.isNanSafe=False

# Use Approximate (Chebyshev) to model background.
config.detect.tempLocalBackground.useApprox=False

# Approximation order in X for background Chebyshev (valid only with useApprox=True)
config.detect.tempLocalBackground.approxOrderX=6

# Approximation order in Y for background Chebyshev (valid only with useApprox=True)
config.detect.tempLocalBackground.approxOrderY=-1

# Use inverse variance weighting in calculation (valid only with useApprox=True)
config.detect.tempLocalBackground.weighting=True

# Convert from a photometrically flat image to one suitable to background subtraction? If True, then a backgroundToPhotometricRatio must be supplied to the task run method.
config.detect.tempLocalBackground.doApplyFlatBackgroundRatio=False

# Enable temporary local background subtraction? (see tempLocalBackground)
config.detect.doTempLocalBackground=False

# type of statistic to use for grid points
config.detect.tempWideBackground.statisticsProperty='MEANCLIP'

# behaviour if there are too few points in grid for requested interpolation style
config.detect.tempWideBackground.undersampleStyle='REDUCE_INTERP_ORDER'

# how large a region of the sky should be used for each background point
config.detect.tempWideBackground.binSize=512

# Sky region size to be used for each background point in X direction. If 0, the binSize config is used.
config.detect.tempWideBackground.binSizeX=0

# Sky region size to be used for each background point in Y direction. If 0, the binSize config is used.
config.detect.tempWideBackground.binSizeY=0

# how to interpolate the background values. This maps to an enum; see afw::math::Background
config.detect.tempWideBackground.algorithm='AKIMA_SPLINE'

# Names of mask planes to ignore while estimating the background
config.detect.tempWideBackground.ignoredPixelMask=['BAD', 'EDGE', 'NO_DATA']

# Ignore NaNs when estimating the background
config.detect.tempWideBackground.isNanSafe=False

# Use Approximate (Chebyshev) to model background.
config.detect.tempWideBackground.useApprox=False

# Approximation order in X for background Chebyshev (valid only with useApprox=True)
config.detect.tempWideBackground.approxOrderX=6

# Approximation order in Y for background Chebyshev (valid only with useApprox=True)
config.detect.tempWideBackground.approxOrderY=-1

# Use inverse variance weighting in calculation (valid only with useApprox=True)
config.detect.tempWideBackground.weighting=True

# Convert from a photometrically flat image to one suitable to background subtraction? If True, then a backgroundToPhotometricRatio must be supplied to the task run method.
config.detect.tempWideBackground.doApplyFlatBackgroundRatio=False

# Do temporary wide (large-scale) background subtraction before footprint detection?
config.detect.doTempWideBackground=False

# The maximum number of peaks in a Footprint before trying to replace its peaks using the temporary local background
config.detect.nPeaksMaxSimple=1

# Multiple of PSF RMS size to use for convolution kernel bounding box size; note that this is not a half-size. The size will be rounded up to the nearest odd integer
config.detect.nSigmaForKernel=7.0

# Mask planes to ignore when calculating statistics of image (for thresholdType=stdev)
config.detect.statsMask=['BAD', 'SAT', 'EDGE', 'NO_DATA']

# Mask planes to exclude when detecting sources.
config.detect.excludeMaskPlanes=[]

# detected sources with fewer than the specified number of pixels will be ignored
config.detectTemplate.minPixels=1

# Grow pixels as isotropically as possible? If False, use a Manhattan metric instead.
config.detectTemplate.isotropicGrow=True

# Grow all footprints at the same time? This allows disconnected footprints to merge.
config.detectTemplate.combinedGrow=True

# Grow detections by nSigmaToGrow * [PSF RMS width]; if 0 then do not grow
config.detectTemplate.nSigmaToGrow=2.4

# Grow detections to set the image mask bits, but return the original (not-grown) footprints
config.detectTemplate.returnOriginalFootprints=False

# Threshold for detecting footprints; exact meaning and units depend on thresholdType.
config.detectTemplate.thresholdValue=50.0

# Multiplier on thresholdValue for whether a source is included in the output catalog. For example, thresholdValue=5, includeThresholdMultiplier=10, thresholdType='pixel_stdev' results in a catalog of sources at >50 sigma with the detection mask and footprints including pixels >5 sigma.
config.detectTemplate.includeThresholdMultiplier=1.0

# Specifies the meaning of thresholdValue.
config.detectTemplate.thresholdType='pixel_stdev'

# Specifies whether to detect positive, or negative sources, or both.
config.detectTemplate.thresholdPolarity='positive'

# Fiddle factor to add to the background; debugging only
config.detectTemplate.adjustBackground=0.0

# Estimate the background again after final source detection?
config.detectTemplate.reEstimateBackground=False

# Convert from a photometrically flat image to one suitable for background subtraction? Only used if reEstimateBackground is True.If True, then a backgroundToPhotometricRatio must be supplied to the task run method.
config.detectTemplate.doApplyFlatBackgroundRatio=False

# type of statistic to use for grid points
config.detectTemplate.background.statisticsProperty='MEANCLIP'

# behaviour if there are too few points in grid for requested interpolation style
config.detectTemplate.background.undersampleStyle='REDUCE_INTERP_ORDER'

# how large a region of the sky should be used for each background point
config.detectTemplate.background.binSize=128

# Sky region size to be used for each background point in X direction. If 0, the binSize config is used.
config.detectTemplate.background.binSizeX=0

# Sky region size to be used for each background point in Y direction. If 0, the binSize config is used.
config.detectTemplate.background.binSizeY=0

# how to interpolate the background values. This maps to an enum; see afw::math::Background
config.detectTemplate.background.algorithm='AKIMA_SPLINE'

# Names of mask planes to ignore while estimating the background
config.detectTemplate.background.ignoredPixelMask=['BAD', 'EDGE', 'DETECTED', 'DETECTED_NEGATIVE', 'NO_DATA']

# Ignore NaNs when estimating the background
config.detectTemplate.background.isNanSafe=False

# Use Approximate (Chebyshev) to model background.
config.detectTemplate.background.useApprox=True

# Approximation order in X for background Chebyshev (valid only with useApprox=True)
config.detectTemplate.background.approxOrderX=6

# Approximation order in Y for background Chebyshev (valid only with useApprox=True)
config.detectTemplate.background.approxOrderY=-1

# Use inverse variance weighting in calculation (valid only with useApprox=True)
config.detectTemplate.background.weighting=True

# Convert from a photometrically flat image to one suitable to background subtraction? If True, then a backgroundToPhotometricRatio must be supplied to the task run method.
config.detectTemplate.background.doApplyFlatBackgroundRatio=False

# type of statistic to use for grid points
config.detectTemplate.tempLocalBackground.statisticsProperty='MEANCLIP'

# behaviour if there are too few points in grid for requested interpolation style
config.detectTemplate.tempLocalBackground.undersampleStyle='REDUCE_INTERP_ORDER'

# how large a region of the sky should be used for each background point
config.detectTemplate.tempLocalBackground.binSize=64

# Sky region size to be used for each background point in X direction. If 0, the binSize config is used.
config.detectTemplate.tempLocalBackground.binSizeX=0

# Sky region size to be used for each background point in Y direction. If 0, the binSize config is used.
config.detectTemplate.tempLocalBackground.binSizeY=0

# how to interpolate the background values. This maps to an enum; see afw::math::Background
config.detectTemplate.tempLocalBackground.algorithm='AKIMA_SPLINE'

# Names of mask planes to ignore while estimating the background
config.detectTemplate.tempLocalBackground.ignoredPixelMask=['BAD', 'EDGE', 'DETECTED', 'DETECTED_NEGATIVE', 'NO_DATA']

# Ignore NaNs when estimating the background
config.detectTemplate.tempLocalBackground.isNanSafe=False

# Use Approximate (Chebyshev) to model background.
config.detectTemplate.tempLocalBackground.useApprox=False

# Approximation order in X for background Chebyshev (valid only with useApprox=True)
config.detectTemplate.tempLocalBackground.approxOrderX=6

# Approximation order in Y for background Chebyshev (valid only with useApprox=True)
config.detectTemplate.tempLocalBackground.approxOrderY=-1

# Use inverse variance weighting in calculation (valid only with useApprox=True)
config.detectTemplate.tempLocalBackground.weighting=True

# Convert from a photometrically flat image to one suitable to background subtraction? If True, then a backgroundToPhotometricRatio must be supplied to the task run method.
config.detectTemplate.tempLocalBackground.doApplyFlatBackgroundRatio=False

# Enable temporary local background subtraction? (see tempLocalBackground)
config.detectTemplate.doTempLocalBackground=False

# type of statistic to use for grid points
config.detectTemplate.tempWideBackground.statisticsProperty='MEANCLIP'

# behaviour if there are too few points in grid for requested interpolation style
config.detectTemplate.tempWideBackground.undersampleStyle='REDUCE_INTERP_ORDER'

# how large a region of the sky should be used for each background point
config.detectTemplate.tempWideBackground.binSize=512

# Sky region size to be used for each background point in X direction. If 0, the binSize config is used.
config.detectTemplate.tempWideBackground.binSizeX=0

# Sky region size to be used for each background point in Y direction. If 0, the binSize config is used.
config.detectTemplate.tempWideBackground.binSizeY=0

# how to interpolate the background values. This maps to an enum; see afw::math::Background
config.detectTemplate.tempWideBackground.algorithm='AKIMA_SPLINE'

# Names of mask planes to ignore while estimating the background
config.detectTemplate.tempWideBackground.ignoredPixelMask=['BAD', 'EDGE', 'NO_DATA']

# Ignore NaNs when estimating the background
config.detectTemplate.tempWideBackground.isNanSafe=False

# Use Approximate (Chebyshev) to model background.
config.detectTemplate.tempWideBackground.useApprox=False

# Approximation order in X for background Chebyshev (valid only with useApprox=True)
config.detectTemplate.tempWideBackground.approxOrderX=6

# Approximation order in Y for background Chebyshev (valid only with useApprox=True)
config.detectTemplate.tempWideBackground.approxOrderY=-1

# Use inverse variance weighting in calculation (valid only with useApprox=True)
config.detectTemplate.tempWideBackground.weighting=True

# Convert from a photometrically flat image to one suitable to background subtraction? If True, then a backgroundToPhotometricRatio must be supplied to the task run method.
config.detectTemplate.tempWideBackground.doApplyFlatBackgroundRatio=False

# Do temporary wide (large-scale) background subtraction before footprint detection?
config.detectTemplate.doTempWideBackground=False

# The maximum number of peaks in a Footprint before trying to replace its peaks using the temporary local background
config.detectTemplate.nPeaksMaxSimple=1

# Multiple of PSF RMS size to use for convolution kernel bounding box size; note that this is not a half-size. The size will be rounded up to the nearest odd integer
config.detectTemplate.nSigmaForKernel=7.0

# Mask planes to ignore when calculating statistics of image (for thresholdType=stdev)
config.detectTemplate.statsMask=['BAD', 'SAT', 'EDGE', 'NO_DATA']

# Mask planes to exclude when detecting sources.
config.detectTemplate.excludeMaskPlanes=[]

# Minimum height of the streak-finding kernel relative to the tallest kernel
config.maskStreaks.minimumKernelHeight=0.0

# Minimum absolute height of the streak-finding kernel
config.maskStreaks.absMinimumKernelHeight=5.0

# Minimum size in pixels of detected clusters
config.maskStreaks.clusterMinimumSize=50

# Allowed deviation (in pixels) from a straight line for a detected line
config.maskStreaks.clusterMinimumDeviation=2

# Stepsize in angle-radius parameter space
config.maskStreaks.delta=0.2

# Number of sigmas from center of kernel to include in voting procedure
config.maskStreaks.nSigma=2.0

# Number of sigmas from center of kernel to mask
config.maskStreaks.nSigmaMask=5.0

# Binsize in pixels for position parameter rho when finding clusters of detected lines
config.maskStreaks.rhoBinSize=30.0

# Binsize in degrees for angle parameter theta when finding clusters of detected lines
config.maskStreaks.thetaBinSize=2.0

# Inverse of the Moffat sigma parameter (in units of pixels)describing the profile of the streak
config.maskStreaks.invSigma=0.1

# Threshold at which to determine edge of line, in units of nanoJanskys
config.maskStreaks.footprintThreshold=0.01

# Absolute difference in Chi2 between iterations of line profilefitting that is acceptable for convergence
config.maskStreaks.dChi2Tolerance=0.1

# Maximum number of line profile fitting iterations that is acceptable for convergence
config.maskStreaks.maxFitIter=100

# Name of mask with pixels above detection threshold, used for firstestimate of streak locations
config.maskStreaks.detectedMaskPlane='DETECTED'

# If true, only propagate the part of the streak mask that overlaps with the detection mask.
config.maskStreaks.onlyMaskDetected=True

# Name of mask plane holding detected streaks
config.maskStreaks.streaksMaskPlane='STREAK'

# Names of mask plane regions to ignore entirely when doing streak detection
config.maskStreaks.badMaskPlanes=['NO_DATA', 'INTRP', 'BAD', 'SAT', 'EDGE']

# Maximum width in pixels to allow for masking a streak.The fit streak parameters will not be modified, and a warning will be issued if the fitted width is larger than this value.Set to 0 to disable.
config.maskStreaks.maxStreakWidth=0.0

# Name of mask bit used for streaks
config.streakMaskName='STREAK'

# Charactistic maximum local number of epochs/visits in which an artifact candidate can appear  and still be masked.  The effective maxNumEpochs is a broken linear function of local number of epochs (N): min(maxFractionEpochsLow*N, maxNumEpochs + maxFractionEpochsHigh*N). For each footprint detected on the image difference between the psfMatched warp and static sky model, if a significant fraction of pixels (defined by spatialThreshold) are residuals in more than the computed effective maxNumEpochs, the artifact candidate is deemed persistant rather than transient and not masked.
config.maxNumEpochs=2

# Fraction of local number of epochs (N) to use as effective maxNumEpochs for low N. Effective maxNumEpochs = min(maxFractionEpochsLow * N, maxNumEpochs + maxFractionEpochsHigh * N)
config.maxFractionEpochsLow=0.4

# Fraction of local number of epochs (N) to use as effective maxNumEpochs for high N. Effective maxNumEpochs = min(maxFractionEpochsLow * N, maxNumEpochs + maxFractionEpochsHigh * N)
config.maxFractionEpochsHigh=0.03

# Unitless fraction of pixels defining how much of the outlier region has to meet the temporal criteria. If 0, clip all. If 1, clip none.
config.spatialThreshold=0.5

# Rescale Warp variance plane using empirical noise?
config.doScaleWarpVariance=True

# type of statistic to use for grid points
config.scaleWarpVariance.background.statisticsProperty='MEANCLIP'

# behaviour if there are too few points in grid for requested interpolation style
config.scaleWarpVariance.background.undersampleStyle='REDUCE_INTERP_ORDER'

# how large a region of the sky should be used for each background point
config.scaleWarpVariance.background.binSize=32

# Sky region size to be used for each background point in X direction. If 0, the binSize config is used.
config.scaleWarpVariance.background.binSizeX=0

# Sky region size to be used for each background point in Y direction. If 0, the binSize config is used.
config.scaleWarpVariance.background.binSizeY=0

# how to interpolate the background values. This maps to an enum; see afw::math::Background
config.scaleWarpVariance.background.algorithm='AKIMA_SPLINE'

# Names of mask planes to ignore while estimating the background
config.scaleWarpVariance.background.ignoredPixelMask=['DETECTED', 'DETECTED_NEGATIVE', 'BAD', 'SAT', 'NO_DATA', 'INTRP']

# Ignore NaNs when estimating the background
config.scaleWarpVariance.background.isNanSafe=False

# Use Approximate (Chebyshev) to model background.
config.scaleWarpVariance.background.useApprox=False

# Approximation order in X for background Chebyshev (valid only with useApprox=True)
config.scaleWarpVariance.background.approxOrderX=6

# Approximation order in Y for background Chebyshev (valid only with useApprox=True)
config.scaleWarpVariance.background.approxOrderY=-1

# Use inverse variance weighting in calculation (valid only with useApprox=True)
config.scaleWarpVariance.background.weighting=True

# Convert from a photometrically flat image to one suitable to background subtraction? If True, then a backgroundToPhotometricRatio must be supplied to the task run method.
config.scaleWarpVariance.background.doApplyFlatBackgroundRatio=False

# Mask planes for pixels to ignore when scaling variance
config.scaleWarpVariance.maskPlanes=['DETECTED', 'DETECTED_NEGATIVE', 'BAD', 'SAT', 'NO_DATA', 'INTRP']

# Maximum variance scaling value to permit
config.scaleWarpVariance.limit=10.0

# Rescue artifacts from clipping that completely lie within a footprint detectedon the PsfMatched Template Coadd. Replicates a behavior of SafeClip.
config.doPreserveContainedBySource=True

# Ignore artifact candidates that are mostly covered by the bad pixel mask, because they will be excluded anyway. This prevents them from contributing to the outlier epoch count image and potentially being labeled as persistant.'Mostly' is defined by the config 'prefilterArtifactsRatio'.
config.doPrefilterArtifacts=True

# Prefilter artifact candidates that are mostly covered by these bad mask planes.
config.prefilterArtifactsMaskPlanes=['NO_DATA', 'BAD', 'SAT', 'SUSPECT']

# Prefilter artifact candidates with less than this fraction overlapping good pixels
config.prefilterArtifactsRatio=0.05

# Filter artifact candidates based on morphological criteria, i.g. those that appear to be streaks.
config.doFilterMorphological=False

# Grow streak footprints by this number multiplied by the PSF width
config.growStreakFp=5.0

