import pandas as pd
import os

import numpy as np

import os
from astropy.table import Table, vstack
import astropy
from astropy.coordinates import SkyCoord
from astropy.time import Time

import getpass

import lsst.afw.display as afwDisplay
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
from lsst.daf.butler import Butler
import lsst.geom as geom

from lsst.ctrl.mpexec import pipeline2dot
from lsst.ctrl.mpexec import SimplePipelineExecutor
from lsst.pipe.base import Pipeline, Instrument


#load collection with calexp 
my_collection_identifier_step1step2 = 'test_512025_1_t4026p15y2022'
print(my_collection_identifier_step1step2)

user = os.getenv("USER")
print(user)

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

# create the coadf for each month
for i in range(6, len(dates_calexps_ym)-1):
    print(dates_calexps_ym['mjd'][i], dates_calexps_ym['mjd'][i+1])
    my_range = np.array((visitTable['obsStartMJD'] >= dates_calexps_ym['mjd'][i]) & 
                    (visitTable['obsStartMJD'] < dates_calexps_ym['mjd'][i+1]))
    my_visits = visitTable_uni.visit[my_range]
    my_visits_tupleString = "("+",".join(my_visits.astype(str))+")"
    print(my_visits_tupleString)

    my_collection_identifier = f"{my_collection_identifier_step1step2}_coadd_5282025_1_{dates_calexps_ym['year'][i]+dates_calexps_ym['month'][i]}-{dates_calexps_ym['month'][i+1]}"
    # print(my_collection_identifier)

    my_outputCollection = f"u/{user}_pm/{my_collection_identifier}" 
    print('Name of new butler collection for my output: ', my_outputCollection)


    simpleButler = SimplePipelineExecutor.prep_butler(repo, 
                                                  inputs=[collections_step1step2], 
                                                  output=my_outputCollection)


    yaml_file = '$DRP_PIPE_DIR/pipelines/LSSTCam-imSim/DRP-test-med-1.yaml'
    steps = 'makeWarp,selectDeepCoaddVisits,assembleCoadd,detection,mergeDetections,deblend,measure'
    my_uri = yaml_file + '#' + steps
    print(my_uri)

    assembleCoaddPipeline = Pipeline.from_uri(my_uri)

    queryString = f"visit in {my_visits_tupleString} AND skymap = 'DC2'"
    print(queryString)


    spe = SimplePipelineExecutor.from_pipeline(assembleCoaddPipeline, 
                                           where=queryString, 
                                           butler=simpleButler)


    quanta = spe.run()

    del simpleButler
    # del assembleCoaddPipeline
    # del spe
    # del quanta

















