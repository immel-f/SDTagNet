from xml.etree import cElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
import numpy as np
import av2_to_wgs_conversion
import nusc_to_wgs_conversion
from shapely.geometry import Polygon, LineString, box, MultiPolygon, MultiLineString, Point
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from tqdm import tqdm


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


NOT_RELEVANT_KEYS = {'addr', 'comment', 'contact', 'source', 'name', 'tiger',
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


def iterparse(fileobj):
    """
    Return root object and iterparser for given ``fileobj``. 
    """
    context = ET.iterparse(fileobj, events=("start", "end"))
    context = iter(context)
    _event, root = context.__next__()
    return root, context


@contextmanager
def log_file_on_exception(xml):
    try:
        yield
    except SyntaxError as ex:
        import tempfile
        fd_, filename = tempfile.mkstemp('.osm')
        xml.seek(0)
        with open(filename, 'w') as f:
            f.write(xml.read())
        print('SyntaxError in xml: %s, (stored dump %s)' % (ex, filename))


@dataclass
class Node:
    id: int
    lat_lon: np.ndarray
    tags: dict


@dataclass
class Way:
    id: int
    node_lat_lon: np.ndarray
    node_ids: np.ndarray
    tags: dict


@dataclass
class RelationMember:
    id: int
    el_type: str
    role: str


@dataclass
class Relation:
    id: int
    members: list
    member_ids: np.ndarray
    tags: dict


def tag_dict_to_str(tag_dict, remove_not_relevant_keys=False):
    string = ""
    for key, val in tag_dict.items():

        if remove_not_relevant_keys and has_not_relevant_key(key, NOT_RELEVANT_KEYS):
            continue

        string += key
        string += ': '
        string += str(val)
        string += ', '
    return string

def all_members_in_patch(relation, ways_patch_ids, nodes_in_patch_ids, relation_ids, filter_with_relations=False):
    for id in relation.member_ids:
        if id in relation_ids and not filter_with_relations:
            continue
        elif id not in ways_patch_ids and id not in nodes_in_patch_ids and id not in relation_ids and filter_with_relations:
            return False
        elif id not in ways_patch_ids and id not in nodes_in_patch_ids and not filter_with_relations:
            return False
    
    return True


class OSMMapElements:
    def __init__(self, nodes=dict(), ways=dict(), relations=dict()):
        self.nodes = nodes
        self.ways = ways
        self.relations = relations

    def build_node_way_lists(self, city_name, nusc_mode=False):
        self.node_list = [node for node in self.nodes.values()]
        self.node_point_array = np.array([node.lat_lon for node in self.node_list])
        
        if nusc_mode:
            self.node_point_array_city = nusc_to_wgs_conversion.convert_wgs84_to_city_coords(self.node_point_array, city_name)
        else:
            self.node_point_array_city = av2_to_wgs_conversion.convert_wgs84_to_city_coords(self.node_point_array, city_name)

        self.node_id_array = np.array([node.id for node in self.node_list])

        # print("nodes converted")

        self.way_list = [way for way in self.ways.values()]
        self.way_point_list = [way.node_lat_lon for way in self.way_list]
        self.way_point_ids = [way.node_ids for way in self.way_list]

        if nusc_mode:

            self.way_point_list_city = [nusc_to_wgs_conversion.convert_wgs84_to_city_coords(way.node_lat_lon, city_name) for way in tqdm(self.way_list)]
        else:
            self.way_point_list_city = [av2_to_wgs_conversion.convert_wgs84_to_city_coords(way.node_lat_lon, city_name) for way in self.way_list]
        
        # print("Ways converted")
        
        self.way_id_array = np.array([way.id for way in self.way_list])

    def get_elements_in_patch(self, patch, remove_not_relevant_keys):

        ways_patch_intersection = [LineString(way).intersection(patch) for way in self.way_point_list_city]
        ways_in_patch_indices = [i for i in range(0, len(self.way_list)) if not ways_patch_intersection[i].is_empty]

        ways_patch_intersection = [ways_patch_intersection[i] for i in ways_in_patch_indices]
        ways_patch_tags = [tag_dict_to_str(self.way_list[i].tags, remove_not_relevant_keys) for i in ways_in_patch_indices]
        ways_patch_ids = [self.way_id_array[i] for i in ways_in_patch_indices]

        ways_patch_no_multilines = []
        ways_patch_tags_no_multilines = []
        ways_patch_ids_no_multilines = []
        for id, tags, lstring in zip(ways_patch_ids, ways_patch_tags, ways_patch_intersection):
            if lstring.geom_type == 'LineString':
                ways_patch_ids_no_multilines.append(id)
                ways_patch_tags_no_multilines.append(tags)
                ways_patch_no_multilines.append(lstring)
            if lstring.geom_type == 'MultiLineString':
                for single_line in lstring.geoms:
                    ways_patch_ids_no_multilines.append(id)
                    ways_patch_tags_no_multilines.append(tags)
                    ways_patch_no_multilines.append(single_line)

        nodes_in_patch_indices = [i for i in range(0, len(self.node_list)) if patch.contains(Point(self.node_point_array_city[i]))]
        nodes_in_patch = [self.node_point_array_city[i] for i in nodes_in_patch_indices]
        nodes_in_patch_ids = [self.node_id_array[i] for i in nodes_in_patch_indices]

        relation_ids = [rel.id for rel in self.relations.values()]
        rels_in_patch_1st_pass = [rel for rel in self.relations.values() if all_members_in_patch(rel, ways_patch_ids, 
                                                                                                 nodes_in_patch_ids, relation_ids)]
        rels_in_patch_1st_pass_ids = [rel.id for rel in rels_in_patch_1st_pass]

        # indices of nodes in patch, not all nodes
        nodes_in_patch_used_indices = []
        nodes_in_patch_used_tags = []
        
        if rels_in_patch_1st_pass:
            rels_in_patch = [rel for rel in self.relations.values() if all_members_in_patch(rel, ways_patch_ids, nodes_in_patch_ids, 
                                                                                            rels_in_patch_1st_pass_ids, filter_with_relations=True)]
            rels_in_patch_ids = [rel.id for rel in rels_in_patch]
            rels_in_patch_tags = [tag_dict_to_str(rel.tags, remove_not_relevant_keys) for rel in rels_in_patch]

        else:
            rels_in_patch = []                                                          
            rels_in_patch_ids = []
            rels_in_patch_tags = []        
     
        rels_in_patch_node_member_indices = [list() for el in rels_in_patch]
        rels_in_patch_way_member_indices = [list() for el in rels_in_patch]
        rels_in_patch_relation_member_indices = [list() for el in rels_in_patch]

        rels_in_patch_node_member_tags = [list() for el in rels_in_patch]
        rels_in_patch_way_member_tags = [list() for el in rels_in_patch]
        rels_in_patch_relation_member_tags = [list() for el in rels_in_patch]

        for i, rel in enumerate(rels_in_patch):
            for member in rel.members:
                if member.el_type == 'node':
                    nodes_in_patch_used_indices.append(nodes_in_patch_ids.index(member.id))
                    node_tags = self.nodes[member.id].tags
                    if node_tags:
                        nodes_in_patch_used_tags.append(tag_dict_to_str(node_tags, remove_not_relevant_keys))
                    else:
                        nodes_in_patch_used_tags.append("")

                    rels_in_patch_node_member_indices[i].append(len(nodes_in_patch_used_indices)-1)
                    rels_in_patch_node_member_tags[i].append('type: ' + member.el_type + ', role: ' + member.role + ', ')
                if member.el_type == 'way':
                    related_way_indices = [i for i, id in enumerate(ways_patch_ids_no_multilines) if id == member.id]
                    rels_in_patch_way_member_indices[i].extend(related_way_indices)
                    rels_in_patch_way_member_tags[i].extend(['type: ' + member.el_type + ', role: ' + member.role + ', ' for el in related_way_indices])
                if member.el_type == 'relation':
                    rels_in_patch_relation_member_indices[i].append(rels_in_patch_ids.index(member.id))
                    rels_in_patch_relation_member_tags[i].append('type: ' + member.el_type + ', role: ' + member.role + ', ')   

        for i in range(0, len(nodes_in_patch_ids)):
            if i in nodes_in_patch_used_indices:
                continue
            elif self.node_list[nodes_in_patch_indices[i]].tags:
                nodes_in_patch_used_indices.append(i)
                nodes_in_patch_used_tags.append(tag_dict_to_str(self.node_list[nodes_in_patch_indices[i]].tags, remove_not_relevant_keys))
        
        #import pdb;pdb.set_trace()
        
        nodes_in_patch_used = np.array([nodes_in_patch[i] for i in nodes_in_patch_used_indices])

        result_dict = dict(
            osm_map_nodes_pts=nodes_in_patch_used,
            osm_map_nodes_tags=nodes_in_patch_used_tags,
            osm_map_ways_pts=ways_patch_no_multilines,
            osm_map_ways_tags=ways_patch_tags_no_multilines,
            osm_map_relations_tags=rels_in_patch_tags,
            osm_map_relations_node_member_indices=rels_in_patch_node_member_indices,
            osm_map_relations_way_member_indices=rels_in_patch_way_member_indices,
            osm_map_relations_relation_member_indices=rels_in_patch_relation_member_indices,
            osm_map_relations_node_member_tags=rels_in_patch_node_member_tags,
            osm_map_relations_way_member_tags=rels_in_patch_way_member_tags,
            osm_map_relations_relation_member_tags=rels_in_patch_relation_member_tags,
        )

        return result_dict
            


def parse(xml):
    nodes = {}
    ways = {}
    relations = {}
    tags = {}
    refs = []
    members = []
    root, context = iterparse(xml)

    with log_file_on_exception(xml):
        for event, elem in context:
            if event == 'start':
                continue
            if elem.tag == 'tag':
                tags[elem.attrib['k']] = elem.attrib['v']
            elif elem.tag == 'node':
                osmid = int(elem.attrib['id'])
                lat, lon = float(elem.attrib['lat']), float(elem.attrib['lon'])
                nodes[osmid] = ((lat, lon), tags)
                tags = {}
            elif elem.tag == 'nd':
                refs.append(int(elem.attrib['ref']))
            elif elem.tag == 'member':
                members.append(
                    (int(elem.attrib['ref']), elem.attrib['type'], elem.attrib['role']))
            elif elem.tag == 'way':
                osm_id = int(elem.attrib['id'])
                ways[osm_id] = (osm_id, tags, refs)
                refs = []
                tags = {}
            elif elem.tag == 'relation':
                osm_id = int(elem.attrib['id'])
                relations[osm_id] = (osm_id, tags, members)
                members = []
                tags = {}

            root.clear()

    # print("OSM xml parsed!")

    map_els = OSMMapElements()

    for id, node in nodes.items():
        map_els.nodes[id] = Node(id, np.array(
            [node[0][0], node[0][1]]), node[1])
    for id, way in ways.items():
        map_els.ways[id] = Way(id, np.array(
            [map_els.nodes[node_id].lat_lon for node_id in way[2]]), np.array(way[2]), way[1])
    for id, relation in relations.items():
        members = [RelationMember(member[0], member[1], member[2])
                   for member in relation[2]]
        map_els.relations[id] = Relation(id, members, np.array(
            [member.id for member in members]), relation[1])

    return map_els


