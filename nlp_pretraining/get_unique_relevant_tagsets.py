import pickle
import os
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import gc


def tag_dict_to_str(tag_dict):
    string = ""
    for key, val in tag_dict.items():
        string += key
        string += ': '
        string += str(val)
        string += ', '
    return string


def has_not_relevant_key(string, not_relevant_keys):
    split = [string]
    if ':' in string:
        split = string.split(':')
    elif '_' in string:
        split = string.split('_')
    for str in split:
        if str.lower() in not_relevant_keys:
            return True
    return False


not_relevant_keys = {'addr', 'comment', 'contact', 'source', 'name', 'tiger',
                     'ref', 'created_by', 'nysgissam', 'wikidata', 'operator',
                     'lacounty', 'osak', 'source_ref', 'nhd', 'admin_level',
                     'wikipedia', 'yh', 'gnis', 'at_bev', 'mml', 'postal_code',
                     'raba', 'nycdoitt', 'maaamet', 'pmfsefin', 'old_name', 'official_name',
                     'chicago', 'linz', 'it', 'destination', 'date', 'lojic'
                     'geobase', 'mapillary', 'clc', 'ssr', 'unsigned_ref', 'naptan'
                     'mvdgis', 'linz2osm', 'gns', 'note', 'metcouncil', 'url',
                     'route_ref', 'gtfs', 'uic', 'attribution', 'date', 'ts',
                     'id', 'survey', 'stif', 'network', 'naptan', 'location',
                     'tmc', 'fixme', 'wabe', 'object', 'description', 'check_date',
                     'tec', 'qroti', 'dcgis', 'website', 'short_name', 'image',
                     'NaPTANAreaCode', 'vrs', 'cxx', 'in', 'code', 'massgis', 'original_osm_id', 
                     'bbr', 'shape', 'lnam', 'redwood_city_ca', 'email', 'KSJ2',
                     'canvec', 'uuid', 'sorting_name', 'phone', 'inegi', 'ine', 
                     'brand', 'cesena', 'geobase', 'mobile', 'strazakosm', 'ipp',
                     'fhrs', 'alt_name', 'old_street', 'ksj2', 'unocha',
                     'wikimedia_commons', 'lojic', 'brn', 'fid', 'notas',
                     'fax', 'sangis', 'okato', 'nhd-shp', 'surrey', 'statscan',
                     'panoramax'}


if __name__ == '__main__':

    tag_dict_path = "osm_planet_tags_unique.pkl"
    tag_data = []

    with open(tag_dict_path, 'rb') as file:
        tag_data = pickle.load(file)
        print("Loaded file " + file.name + " with " + str(len(tag_data)) + " tags")

    unique_relevant_tagsets = defaultdict(list)

    for tagset in tqdm(tag_data):
        if 'original_osm_id' in tagset:
            del tagset['original_osm_id']
        relevant_tags = [frozenset([key, val]) for key, val in tagset.items(
        ) if not has_not_relevant_key(key, not_relevant_keys)]
        relevant_tags = frozenset(relevant_tags)
        unique_relevant_tagsets[relevant_tags].append(tagset)
    print("Unique relevant tagsets: " + str(len(unique_relevant_tagsets)))

    with open('osm_planet_unique_relevant_tagsets.pkl', 'wb') as handle:
        pickle.dump(unique_relevant_tagsets, handle, protocol=pickle.HIGHEST_PROTOCOL) 