# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>
# Used and modified from https://github.com/argoverse/av2-api/blob/main/src/av2/geometry/utm.py#L53

"""Utilities for converting AV2 city coordinates to UTM or WGS84 coordinate systems.

Reference:
UTM: https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system
WGS84: https://en.wikipedia.org/wiki/World_Geodetic_System
"""

from enum import Enum, unique
from typing import Dict, Final, Tuple, Union

import numpy as np
from pyproj import Proj
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info


EARTH_RADIUS_METERS = 6.378137e6
# as (lat, long) tuples
ORIGIN_LATLON = {
    'boston-seaport':[42.336849169438615, -71.05785369873047],
    'singapore-onenorth': [1.2882100868743724, 103.78475189208984],
    'singapore-hollandvillage': [1.2993652317780957, 103.78217697143555],
    'singapore-queenstown': [1.2782562240223188, 103.76741409301758]
}


def convert_wgs84_to_city_coords(points_wgs84: np.ndarray, map_name: str) -> np.ndarray:

    latitude, longitude = ORIGIN_LATLON[map_name]
    
    phi_1 = latitude * np.pi/180
    phi_2 = points_wgs84[:, 0] * np.pi/180
    lambda_1 = longitude * np.pi/180
    lambda_2 = points_wgs84[:, 1] * np.pi/180
    delta_phi = (points_wgs84[:, 0]-latitude) * np.pi/180
    delta_lambda = (points_wgs84[:, 1]-longitude) * np.pi/180

    a = np.sin(delta_phi/2) * np.sin(delta_phi/2) + np.cos(phi_1) * np.cos(phi_2) *np.sin(delta_lambda/2) * np.sin(delta_lambda/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    d = EARTH_RADIUS_METERS * c

    y = np.sin(lambda_2-lambda_1) * np.cos(phi_2)
    x = np.cos(phi_1)*np.sin(phi_2) - np.sin(phi_1)*np.cos(phi_2)*np.cos(lambda_2-lambda_1)
    theta = np.arctan2(y, x)

    # import pdb; pdb.set_trace()

    # return np.stack([d*np.cos(theta), d*np.sin(theta)], axis=1)
    return np.stack([d*np.sin(theta), d*np.cos(theta)], axis=1)