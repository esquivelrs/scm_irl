import os
import numpy as np 
import pandas as pd
import json
import pickle
from shapely.geometry import Polygon
import sllib.parsers.ais_scenario as ais_scenario
from sllib.datatypes.vessel import VesselState
from sllib.conversions.geo_conversions import lat_lon_to_north_east
import overpy
import math 
import yaml

from . import CONFIG_DIR

class Scenario:
    def __init__(self, cfg, scenario_path):
        self.cfg = cfg
        self.api = overpy.Overpass()
        self.scenario_path = scenario_path
        self._load_metadata(self.scenario_path)
        self._load_states(self.scenario_path)
        self._compute_vessel_relative_neighbor_states()
        self.depth_lands_polygons, self.depth_lands_polygons_lat_lon = self.scenario_depth_lands_polygons()
        #print(cfg['env'])
        self.seamark_conf = cfg['env']['seamarks']

    def _load_metadata(self, scenario_path):
        metadata_path = os.path.join(scenario_path, "metadata.json")
        metadata = json.load(open(metadata_path))
        self.scenario_id = metadata["scenario_id"]
        self.start_time = metadata["start_time"]
        self.end_time = metadata["end_time"]
        self.start_epoch = metadata["start_epoch"]
        self.end_epoch = metadata["end_epoch"]
        self.sampling_time = metadata["sampling_time"]
        self.lat_limits = metadata["lat_limits"]
        self.lon_limits = metadata["lon_limits"]
        self.lon0, self.lat0 = ais_scenario.scenario_to_origin_coordinate(scenario_path)
        self.north_min, self.east_min = lat_lon_to_north_east(self.lat_limits[0], self.lon_limits[0], self.lat0, self.lon0)
        self.north_max, self.east_max = lat_lon_to_north_east(self.lat_limits[1], self.lon_limits[1], self.lat0, self.lon0)

        self.n_vessels = metadata["n_vessels"]
        self.collision_mmsis = metadata["collision_mmsis"]

    def _load_vessel_states_lat_lon(self, scenario_path):
        states = ais_scenario.scenario_to_vessel_states(scenario_path)
        vessels= {}
        for mmsi, vessel_states in states.items():
            vessels[mmsi] = vessel_states["data"]
        return vessels


    def _load_states(self, scenario_path):
        self.vessels_states_lat_lon = self._load_vessel_states_lat_lon(scenario_path)
        states = ais_scenario.scenario_to_vessel_states_and_lands_north_east(scenario_path)
        self.vessels = states["vessels"]
        self.lands = states["lands"]
        self.mmsis = list(self.vessels.keys()) 


    def _compute_vessel_relative_neighbor_states(self):
        for mmsi in self.mmsis:
            self.vessels[mmsi]["relative_neighbor_states"] = {}
            for neighbor_mmsi in self.mmsis:
                if neighbor_mmsi != mmsi:
                    self.vessels[mmsi]["relative_neighbor_states"][neighbor_mmsi] = self.get_vessel_relative_neighbor_states(mmsi, neighbor_mmsi)

    
    def is_vessel_active(self, mmsi, time):
        states = self.get_vessel_states(mmsi)
        return time in states

    def get_vessel(self, mmsi):
        return self.vessels[mmsi]
    
    def get_vessel_metadata(self, mmsi):
        return self.vessels[mmsi]["metadata"]

    # get the states of a vessel
    def get_vessel_states(self, mmsi):
        return self.vessels[mmsi]["states"]

    def get_vessel_state_time(self, mmsi, time):
        states = self.get_vessel_states(mmsi)
        if time in states:
            return states[time]
        else:
            # TODO: Do de intepolation insted of return the previuos state
            return states[time - self.sampling_time]
    
    def get_vessel_neighbor_states(self, mmsi, neighbor_mmsi):
        return self.vessels[mmsi]["neighbor_states"][neighbor_mmsi]
    
    def get_vessel_neighbor_state_time(self, mmsi, neighbor_mmsi, time):
        states = self.get_vessel_neighbor_states(mmsi, neighbor_mmsi)
        return states[time]
    
    # transform the states of the neighbors relative to the vessel
    def get_vessel_relative_neighbor_states(self, mmsi, neighbor_mmsi):
        neighbor_states = self.get_vessel_neighbor_states(mmsi, neighbor_mmsi)
        vessel_states = self.get_vessel_states(mmsi)
        relative_states = {}
        for time, state in vessel_states.items():
            if time in neighbor_states:
                neighbor_state = neighbor_states[time]
                relative_states[time] = self.relative_state(neighbor_state, state)
        return relative_states
    
    def relative_state(self, state, vessel_state):
        vessel_state = VesselState(timestamp=state.timestamp,
                                       lat=state.lat - vessel_state.lat,
                                       lon=state.lon - vessel_state.lon,
                                       sog=state.sog - vessel_state.sog * np.cos(vessel_state.cog - state.cog),
                                       cog=state.cog - vessel_state.cog)
        return vessel_state
    

    def scenario_to_lands(self):
        with open(os.path.join(self.scenario_path,"land.pickle"), 'rb') as file:
            lands = pickle.load(file)
        land_idx = 0
        land_polygons = []
        land_polygons_lat_lon = []
        for polygon in lands:
            if polygon.geom_type == 'Polygon':
                x, y = polygon.exterior.xy
                polygon_list = []
                polygon_list_lat_long = []
                for i in range(len(x)):
                    north, east = lat_lon_to_north_east(y[i], x[i], self.lat0, self.lon0)
                    polygon_list.append((north, east))
                    polygon_list_lat_long.append((y[i], x[i]))
                #ax.annotate(str(land_idx), xy=(np.mean(x),np.mean(y)))
                land_idx += 1
                land_polygons.append(polygon_list)
                land_polygons_lat_lon.append(polygon_list_lat_long)
            elif polygon.geom_type == 'MultiPolygon':
                for poly in polygon.geoms:
                    x, y = poly.exterior.xy
                    #ax.annotate(str(land_idx), xy=(np.mean(x),np.mean(y)))
                    land_idx += 1
                    polygon_list = []
                    polygon_list_lat_long = []
                    for i in range(len(x)):
                        north, east = lat_lon_to_north_east(y[i], x[i], self.lat0, self.lon0)
                        polygon_list.append((north, east))
                        polygon_list_lat_long.append((y[i], x[i]))
                    land_polygons.append(polygon_list)
                    land_polygons_lat_lon.append(polygon_list_lat_long)
        return land_polygons, land_polygons_lat_lon
    

    def scenario_to_depths(self):
        with open(os.path.join(self.scenario_path,"depth.pickle"), 'rb') as file:
            depths = pickle.load(file)
        depth_idx = 0
        depth_polygons = []
        depth_polygons_lat_lon = []
        for depth in depths:
            polygon = depth["exterior"]
            depth2 = depth["depth2"]
            if polygon.geom_type == 'Polygon':
                x, y = polygon.exterior.xy
                polygon_list = []
                polygon_list_lat_lon = []
                for i in range(len(x)):
                    north, east = lat_lon_to_north_east(y[i], x[i], self.lat0, self.lon0)
                    polygon_list.append((north, east))
                    polygon_list_lat_lon.append((y[i], x[i]))
                #ax.annotate(str(depth_idx), xy=(np.mean(x),np.mean(y)))
                depth_idx += 1
                depth_polygons.append((polygon_list, depth2))
                depth_polygons_lat_lon.append((polygon_list_lat_lon, depth2))
            elif polygon.geom_type == 'MultiPolygon':
                for poly in polygon.geoms:
                    x, y = poly.exterior.xy
                    #ax.annotate(str(depth_idx), xy=(np.mean(x),np.mean(y)))
                    depth_idx += 1
                    polygon_list = []
                    polygon_list_lat_lon = []
                    for i in range(len(x)):
                        north, east = lat_lon_to_north_east(y[i], x[i], self.lat0, self.lon0)
                        polygon_list.append((north, east))
                        polygon_list_lat_lon.append((y[i], x[i]))
                    depth_polygons.append((polygon_list, depth2))
                    depth_polygons_lat_lon.append((polygon_list_lat_lon, depth2))
        
        return depth_polygons, depth_polygons_lat_lon
    

    def scenario_depth_lands_polygons(self):
        lands, lands_lat_lon = self.scenario_to_lands()
        depths, depth_lat_lon = self.scenario_to_depths()
        depths_lands = [(Polygon([(x[1], x[0]) for x in depth[0]]), depth[1]) for depth in depths] + [(Polygon([(x[1], x[0]) for x in land]), 0) for land in lands]
        
        depths_lands_lat_lon= [(Polygon([(x[1], x[0]) for x in depth[0]]), depth[1]) for depth in depth_lat_lon] + [(Polygon([(x[1], x[0]) for x in land]), 0) for land in lands_lat_lon]
        
        return depths_lands, depths_lands_lat_lon


    def get_nodes_and_ways_scenario(self):
        bbox = (self.lat_limits[0], self.lon_limits[0], self.lat_limits[1], self.lon_limits[1])
        result = self.api.query(f"""
            (
            way["seamark:type"="shipping_lane"]({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]});
            way["route"="ferry"]({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]});
            way["route"="motorboat"]({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]});
            way["route"="canoe"]({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]});
            way["route"="waterway"]({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]});
            way["route"="shipping_lane"]({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]});
            way["seamark:type"="harbour"]({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]});
            way["seamark:type"]({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]});
            way["seamark:type"="separation_boundary"]({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]});
            way["seamark:type"="separation_zone"]({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]});
            way["seamark:type"="separation_lane"]({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]});
            node["seamark:type"]({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]});
            );
            (._;>;);
            out body;
        """)
        return result.nodes, result.ways
    
    def get_nodes_and_ways_scenario_north_east(self):
        nodes, ways = self.get_nodes_and_ways_scenario()
        nodes_north_east = []
        ways_north_east = []
        for node in nodes:
            north, east = lat_lon_to_north_east(float(node.lat), float(node.lon), self.lat0, self.lon0)
            node_val = self.seamark_conf[node.tags.get('seamark:type') if node.tags.get('seamark:type') else 'Unknown']
            nodes_north_east.append({'id':node.id, 'north':north, 'east':east, 
                                     'type':node.tags.get('seamark:type'), 
                                     'color': node_val['color'],
                                     'value': node_val["value"],
                                     'size': node_val["size"]})
            
        for way in ways:
            way_nodes = []
            for node in way.nodes:
                north, east = lat_lon_to_north_east(float(node.lat), float(node.lon), self.lat0, self.lon0)
                way_nodes.append((north, east))
            way_val = self.seamark_conf['way']
            ways_north_east.append({'id':way.id, 'nodes':way_nodes, 
                                    'type':way.tags.get('seamark:type'), 
                                    'color': way_val['color'],
                                    'value': way_val["value"],})
        
        return nodes_north_east, ways_north_east
    
    def rotate_point(self, x, y, centerX, centerY, angle):
        # Create a rotation matrix
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        # Subtract the center point, rotate, then add the center point
        point = np.dot(rotation_matrix, [x - centerX, y - centerY]) + [centerX, centerY]

        return int(point[0]), int(point[1])
    
    def nodes_list(self):
        # sort the nodes based on the value
        nodes = self.seamark_conf
        nodes_val_list = []
        for node in nodes:
            nodes_val_list.append((node, nodes[node]["value"], nodes[node]["color"]))

        nodes_val_list = sorted(nodes_val_list, key=lambda x: x[1], reverse=True)
        # get list of colors
        colors = [node[2] for node in nodes_val_list]

        return nodes_val_list, colors

        
        



