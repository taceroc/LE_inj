import lsst.pipe.tasks.selectImages
assert type(config) is lsst.pipe.tasks.selectImages.BestSeeingSelectVisitsConfig, f"config is of type {type(config).__module__}.{type(config).__name__} instead of lsst.pipe.tasks.selectImages.BestSeeingSelectVisitsConfig"

import lsst.pipe.base.config
# Flag to enable/disable saving of log output for a task, enabled by default.
config.saveLogOutput=True

# Maximum number of visits to select; use -1 to select all.
config.nVisitsMax=-1

# Maximum PSF FWHM (in arcseconds) to select
config.maxPsfFwhm=1.7

# Minimum PSF FWHM (in arcseconds) to select
config.minPsfFwhm=0.0

# Do remove visits that do not actually overlap the patch?
config.doConfirmOverlap=True

# Minimum visit MJD to select
config.minMJD=None

# Maximum visit MJD to select
config.maxMJD=None

# name for connection skyMap
config.connections.skyMap='skyMap'

# name for connection visitSummaries
config.connections.visitSummaries='visit_summary'

# name for connection goodVisits
config.connections.goodVisits='deep_coadd_visit_selection'

# Template parameter used to format corresponding field template parameter
config.connections.coaddName='goodSeeing'

# Template parameter used to format corresponding field template parameter
config.connections.calexpType=''

