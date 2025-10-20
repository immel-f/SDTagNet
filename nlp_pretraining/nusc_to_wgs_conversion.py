
# Used and modified from https://github.com/argoverse/av2-api/blob/main/src/av2/geometry/utm.py#L53

"""Utilities for converting nuScenes city coordinates to UTM or WGS84 coordinate systems.

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

# as (lat, long) tuples
ORIGIN_LATLON = {
    'boston-seaport': [42.336849169438615, -71.05785369873047],
    'singapore-onenorth': [1.2882100868743724, 103.78475189208984],
    'singapore-hollandvillage': [1.2993652317780957, 103.78217697143555],
    'singapore-queenstown': [1.2782562240223188, 103.76741409301758]
}


def convert_gps_to_utm(
    latitude: float, longitude: float, map_name: str
) -> Tuple[float, float]:
    """Convert GPS coordinates to UTM coordinates.

    Args:
        latitude: latitude of query point.
        longitude: longitude of query point.
        city_name: name of city, where query point is located.

    Returns:
        easting: corresponding UTM Easting.
        northing: corresponding UTM Northing.
    """

    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=longitude,
            south_lat_degree=latitude,
            east_lon_degree=longitude,
            north_lat_degree=latitude,
        ),
    )
    utm_crs = CRS.from_epsg(utm_crs_list[0].code)

    # print(int(utm_crs.utm_zone[:-1]))
    # print(utm_crs.utm_zone)

    projector = Proj(
        proj="utm",
        zone=int(utm_crs.utm_zone[:-1]),
        ellps="WGS84",
        datum="WGS84",
        units="m",
    )

    # convert to UTM.
    easting, northing = projector(longitude, latitude)

    return easting, northing


def convert_city_coords_to_utm(
    points_city: np.ndarray, map_name: str
) -> np.ndarray:
    """Convert city coordinates to UTM coordinates.

    Args:
        points_city: (N,2) array, representing 2d query points in the city coordinate frame.
        city_name: name of city, where query points are located.

    Returns:
        Array of shape (N,2), representing points in the UTM coordinate system, as (easting, northing).
    """
    latitude, longitude = ORIGIN_LATLON[map_name]
    # get (easting, northing) of origin
    origin_utm = convert_gps_to_utm(
        latitude=latitude, longitude=longitude, map_name=map_name
    )
    points_utm = points_city.astype(float) + np.array(
        origin_utm, dtype=float
    )
    return points_utm


def convert_city_coords_to_wgs84(
    points_city: np.ndarray, map_name: str
) -> np.ndarray:
    """Convert city coordinates to WGS84 coordinates.

    Args:
        points_city: (N,2) array, representing 2d query points in the city coordinate frame.
        city_name: name of city, where query points are located.

    Returns:
        Array of shape (N,2), representing points in the WGS84 coordinate system, as (latitude, longitude).
    """
    points_utm = convert_city_coords_to_utm(points_city, map_name)

    latitude, longitude = ORIGIN_LATLON[map_name]
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=longitude,
            south_lat_degree=latitude,
            east_lon_degree=longitude,
            north_lat_degree=latitude,
        ),
    )

    utm_crs = CRS.from_epsg(utm_crs_list[0].code)

    # print(int(utm_crs.utm_zone[:-1]))

    projector = Proj(
        proj="utm",
        zone=int(utm_crs.utm_zone[:-1]),
        ellps="WGS84",
        datum="WGS84",
        units="m",
    )

    points_wgs84 = []
    for easting, northing in points_utm:
        longitude, latitude = projector(easting, northing, inverse=True)
        points_wgs84.append((latitude, longitude))

    return np.array(points_wgs84)
