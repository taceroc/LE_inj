import lsst.drp.tasks.make_psf_matched_warp
assert type(config) is lsst.drp.tasks.make_psf_matched_warp.MakePsfMatchedWarpConfig, f"config is of type {type(config).__module__}.{type(config).__name__} instead of lsst.drp.tasks.make_psf_matched_warp.MakePsfMatchedWarpConfig"

import lsst.afw.math._warper
import lsst.ip.diffim.modelPsfMatch
import lsst.ip.diffim.psfMatch
import lsst.meas.algorithms.gaussianPsfFactory
import lsst.meas.algorithms.subtractBackground
import lsst.pipe.base.config
# Flag to enable/disable saving of log output for a task, enabled by default.
config.saveLogOutput=True

# Kernel size (width and height) (pixels); if None then sizeFactor is used
config.modelPsf.size=None

# Kernel size as a factor of fwhm (dimensionless); size = sizeFactor * fwhm; ignored if size is not None
config.modelPsf.sizeFactor=3.0

# Minimum kernel size if using sizeFactor (pixels); ignored if size is not None
config.modelPsf.minSize=5

# Maximum kernel size if using sizeFactor (pixels); ignored if size is not None
config.modelPsf.maxSize=None

# Default FWHM of Gaussian model of core of star (pixels)
config.modelPsf.defaultFwhm=9.0

# Add a Gaussian to represent wings?
config.modelPsf.addWing=True

# wing width, as a multiple of core width (dimensionless); ignored if addWing false
config.modelPsf.wingFwhmFactor=2.5

# wing amplitude, as a multiple of core amplitude (dimensionless); ignored if addWing false
config.modelPsf.wingAmplitude=0.1

# Warping kernel
config.psfMatch.kernel['AL'].warpingConfig.warpingKernelName='lanczos3'

# Warping kernel for mask (use ``warpingKernelName`` if '')
config.psfMatch.kernel['AL'].warpingConfig.maskWarpingKernelName='bilinear'

# ``interpLength`` argument to `lsst.afw.math.warpExposure`
config.psfMatch.kernel['AL'].warpingConfig.interpLength=10

# ``cacheSize`` argument to `lsst.afw.math.SeparableKernel.computeCache`
config.psfMatch.kernel['AL'].warpingConfig.cacheSize=1000000

# mask bits to grow to full width of image/variance kernel,
config.psfMatch.kernel['AL'].warpingConfig.growFullMask=16

# type of statistic to use for grid points
config.psfMatch.kernel['AL'].afwBackgroundConfig.statisticsProperty='MEANCLIP'

# behaviour if there are too few points in grid for requested interpolation style
config.psfMatch.kernel['AL'].afwBackgroundConfig.undersampleStyle='REDUCE_INTERP_ORDER'

# how large a region of the sky should be used for each background point
config.psfMatch.kernel['AL'].afwBackgroundConfig.binSize=128

# Sky region size to be used for each background point in X direction. If 0, the binSize config is used.
config.psfMatch.kernel['AL'].afwBackgroundConfig.binSizeX=0

# Sky region size to be used for each background point in Y direction. If 0, the binSize config is used.
config.psfMatch.kernel['AL'].afwBackgroundConfig.binSizeY=0

# how to interpolate the background values. This maps to an enum; see afw::math::Background
config.psfMatch.kernel['AL'].afwBackgroundConfig.algorithm='AKIMA_SPLINE'

# Names of mask planes to ignore while estimating the background
config.psfMatch.kernel['AL'].afwBackgroundConfig.ignoredPixelMask=['BAD', 'EDGE', 'DETECTED', 'DETECTED_NEGATIVE', 'NO_DATA']

# Ignore NaNs when estimating the background
config.psfMatch.kernel['AL'].afwBackgroundConfig.isNanSafe=False

# Use Approximate (Chebyshev) to model background.
config.psfMatch.kernel['AL'].afwBackgroundConfig.useApprox=True

# Approximation order in X for background Chebyshev (valid only with useApprox=True)
config.psfMatch.kernel['AL'].afwBackgroundConfig.approxOrderX=6

# Approximation order in Y for background Chebyshev (valid only with useApprox=True)
config.psfMatch.kernel['AL'].afwBackgroundConfig.approxOrderY=-1

# Use inverse variance weighting in calculation (valid only with useApprox=True)
config.psfMatch.kernel['AL'].afwBackgroundConfig.weighting=True

# Convert from a photometrically flat image to one suitable to background subtraction? If True, then a backgroundToPhotometricRatio must be supplied to the task run method.
config.psfMatch.kernel['AL'].afwBackgroundConfig.doApplyFlatBackgroundRatio=False

# Use afw background subtraction instead of ip_diffim
config.psfMatch.kernel['AL'].useAfwBackground=False

# Include terms (including kernel cross terms) for background in ip_diffim
config.psfMatch.kernel['AL'].fitForBackground=False

# Type of basis set for PSF matching kernel.
config.psfMatch.kernel['AL'].kernelBasisSet='alard-lupton'

# Number of rows/columns in the convolution kernel; should be odd-valued.
#                  Modified by kernelSizeFwhmScaling if scaleByFwhm = true
config.psfMatch.kernel['AL'].kernelSize=29

# Scale kernelSize, alardGaussians by input Fwhm
config.psfMatch.kernel['AL'].scaleByFwhm=False

# Multiplier of the largest AL Gaussian basis sigma to get the kernel bbox (pixel) size.
config.psfMatch.kernel['AL'].kernelSizeFwhmScaling=6.0

# Minimum kernel bbox (pixel) size.
config.psfMatch.kernel['AL'].kernelSizeMin=21

# Maximum kernel bbox (pixel) size.
config.psfMatch.kernel['AL'].kernelSizeMax=35

# Type of spatial functions for kernel and background
config.psfMatch.kernel['AL'].spatialModelType='chebyshev1'

# Spatial order of convolution kernel variation
config.psfMatch.kernel['AL'].spatialKernelOrder=2

# Spatial order of differential background variation
config.psfMatch.kernel['AL'].spatialBgOrder=1

# Size (rows) in pixels of each SpatialCell for spatial modeling
config.psfMatch.kernel['AL'].sizeCellX=128

# Size (columns) in pixels of each SpatialCell for spatial modeling
config.psfMatch.kernel['AL'].sizeCellY=128

# Maximum number of KernelCandidates in each SpatialCell to use in the spatial fitting. Set to -1 to use all candidates in each cell.
config.psfMatch.kernel['AL'].nStarPerCell=5

# Maximum number of iterations for rejecting bad KernelCandidates in spatial fitting
config.psfMatch.kernel['AL'].maxSpatialIterations=3

# Use Pca to reduce the dimensionality of the kernel basis sets.
#                  This is particularly useful for delta-function kernels.
#                  Functionally, after all Cells have their raw kernels determined, we run
#                  a Pca on these Kernels, re-fit the Cells using the eigenKernels and then
#                  fit those for spatial variation using the same technique as for Alard-Lupton kernels.
#                  If this option is used, the first term will have no spatial variation and the
#                  kernel sum will be conserved.
config.psfMatch.kernel['AL'].usePcaForSpatialKernel=False

# Subtract off the mean feature before doing the Pca
config.psfMatch.kernel['AL'].subtractMeanForPca=True

# Number of principal components to use for Pca basis, including the
#                  mean kernel if requested.
config.psfMatch.kernel['AL'].numPrincipalComponents=5

# Do sigma clipping on each raw kernel candidate
config.psfMatch.kernel['AL'].singleKernelClipping=False

# Do sigma clipping on the ensemble of kernel sums
config.psfMatch.kernel['AL'].kernelSumClipping=False

# Do sigma clipping after building the spatial model
config.psfMatch.kernel['AL'].spatialKernelClipping=False

# Test for maximum condition number when inverting a kernel matrix.
#                  Anything above maxConditionNumber is not used and the candidate is set as BAD.
#                  Also used to truncate inverse matrix in estimateBiasedRisk.  However,
#                  if you are doing any deconvolution you will want to turn this off, or use
#                  a large maxConditionNumber
config.psfMatch.kernel['AL'].checkConditionNumber=False

# Mask planes to ignore when calculating diffim statistics
#                  Options: NO_DATA EDGE SAT BAD CR INTRP
config.psfMatch.kernel['AL'].badMaskPlanes=['NO_DATA', 'EDGE', 'SAT']

# Rejects KernelCandidates yielding bad difference image quality.
#                  Used by BuildSingleKernelVisitor, AssessSpatialKernelVisitor.
#                  Represents average over pixels of (image/sqrt(variance)).
config.psfMatch.kernel['AL'].candidateResidualMeanMax=0.25

# Rejects KernelCandidates yielding bad difference image quality.
#                  Used by BuildSingleKernelVisitor, AssessSpatialKernelVisitor.
#                  Represents stddev over pixels of (image/sqrt(variance)).
config.psfMatch.kernel['AL'].candidateResidualStdMax=1.5

# Use the core of the footprint for the quality statistics, instead of the entire footprint.
#                  WARNING: if there is deconvolution we probably will need to turn this off
config.psfMatch.kernel['AL'].useCoreStats=False

# Radius for calculation of stats in 'core' of KernelCandidate diffim.
#                  Total number of pixels used will be (2*radius)**2.
#                  This is used both for 'core' diffim quality as well as ranking of
#                  KernelCandidates by their total flux in this core
config.psfMatch.kernel['AL'].candidateCoreRadius=3

# Maximum allowed sigma for outliers from kernel sum distribution.
#                  Used to reject variable objects from the kernel model
config.psfMatch.kernel['AL'].maxKsumSigma=3.0

# Maximum condition number for a well conditioned matrix
config.psfMatch.kernel['AL'].maxConditionNumber=50000000.0

# Use singular values (SVD) or eigen values (EIGENVALUE) to determine condition number
config.psfMatch.kernel['AL'].conditionNumberType='EIGENVALUE'

# Maximum condition number for a well conditioned spatial matrix
config.psfMatch.kernel['AL'].maxSpatialConditionNumber=10000000000.0

# Remake KernelCandidate using better variance estimate after first pass?
#                  Primarily useful when convolving a single-depth image, otherwise not necessary.
config.psfMatch.kernel['AL'].iterateSingleKernel=False

# Use constant variance weighting in single kernel fitting?
#                  In some cases this is better for bright star residuals.
config.psfMatch.kernel['AL'].constantVarianceWeighting=True

# Calculate kernel and background uncertainties for each kernel candidate?
#                  This comes from the inverse of the covariance matrix.
#                  Warning: regularization can cause problems for this step.
config.psfMatch.kernel['AL'].calculateKernelUncertainty=False

# Use Bayesian Information Criterion to select the number of bases going into the kernel
config.psfMatch.kernel['AL'].useBicForKernelBasis=False

# Number of base Gaussians in alard-lupton kernel basis function generation.
config.psfMatch.kernel['AL'].alardNGauss=3

# Polynomial order of spatial modification of base Gaussians. List length must be `alardNGauss`.
config.psfMatch.kernel['AL'].alardDegGauss=[4, 2, 2]

# Default sigma values in pixels of base Gaussians. List length must be `alardNGauss`.Only used if the template and science image PSFs have equal size.
config.psfMatch.kernel['AL'].alardSigGauss=[1.0, 2.0, 4.5]

# Used if `scaleByFwhm==True`, scaling multiplier of base Gaussian sigmas for automated sigma determination
config.psfMatch.kernel['AL'].alardGaussBeta=2.0

# Used if `scaleByFwhm==True`, minimum sigma (pixels) for base Gaussians
config.psfMatch.kernel['AL'].alardMinSig=0.7

# Used if `scaleByFwhm==True`, minimum sigma (pixels) for base Gaussians during deconvolution; make smaller than `alardMinSig` as this is only indirectly used
config.psfMatch.kernel['AL'].alardMinSigDeconv=0.4

config.psfMatch.kernel.name='AL'
# If too small, automatically pad the science Psf? Pad to smallest dimensions appropriate for the matching kernel dimensions, as specified by autoPadPsfTo. If false, pad by the padPsfBy config.
config.psfMatch.doAutoPadPsf=True

# Minimum Science Psf dimensions as a fraction of matching kernel dimensions. If the dimensions of the Psf to be matched are less than the matching kernel dimensions * autoPadPsfTo, pad Science Psf to this size. Ignored if doAutoPadPsf=False.
config.psfMatch.autoPadPsfTo=1.4

# Pixels (even) to pad Science Psf by before matching. Ignored if doAutoPadPsf=True
config.psfMatch.padPsfBy=0

# name for connection sky_map
config.connections.sky_map='skyMap'

# name for connection direct_warp
config.connections.direct_warp='direct_warp'

# name for connection psf_matched_warp
config.connections.psf_matched_warp='psf_matched_warp'

# Template parameter used to format corresponding field template parameter
config.connections.coaddName='deep'

