import os
import numpy as np 
import pandas as pd
import json
import pickle
from shapely.geometry import Polygon
import sllib.parsers.ais_scenario as ais_scenario
from sllib.datatypes.vessel import VesselState
from sllib.conversions.geo_conversions import lat_lon_to_north_east


class Scenario:
    def __init__(self, scenario_path):
        self.scenario_path = scenario_path
        self._load_metadata(self.scenario_path)
        self._load_states(self.scenario_path)
        self._compute_vessel_relative_neighbor_states()
        self.depth_lands_polygons = self.scenario_depth_lands_polygons()

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


    def _load_states(self, scenario_path):
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
        return states[time]
    
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
    

    def scenario_to_lands_north_east(self):
        with open(os.path.join(self.scenario_path,"land.pickle"), 'rb') as file:
            lands = pickle.load(file)
        land_idx = 0
        land_polygons = []
        for polygon in lands:
            if polygon.geom_type == 'Polygon':
                x, y = polygon.exterior.xy
                polygon_list = []
                for i in range(len(x)):
                    north, east = lat_lon_to_north_east(y[i], x[i], self.lat0, self.lon0)
                    polygon_list.append((north, east))
                #ax.annotate(str(land_idx), xy=(np.mean(x),np.mean(y)))
                land_idx += 1
                land_polygons.append(polygon_list)
            elif polygon.geom_type == 'MultiPolygon':
                for poly in polygon.geoms:
                    x, y = poly.exterior.xy
                    #ax.annotate(str(land_idx), xy=(np.mean(x),np.mean(y)))
                    land_idx += 1
                    polygon_list = []
                    for i in range(len(x)):
                        north, east = lat_lon_to_north_east(y[i], x[i], self.lat0, self.lon0)
                        polygon_list.append((north, east))
                    land_polygons.append(polygon_list)
        return land_polygons
    

    def scenario_to_depths_north_east(self):
        with open(os.path.join(self.scenario_path,"depth.pickle"), 'rb') as file:
            depths = pickle.load(file)
        depth_idx = 0
        depth_polygons = []
        for depth in depths:
            polygon = depth["exterior"]
            depth2 = depth["depth2"]
            if polygon.geom_type == 'Polygon':
                x, y = polygon.exterior.xy
                polygon_list = []
                for i in range(len(x)):
                    north, east = lat_lon_to_north_east(y[i], x[i], self.lat0, self.lon0)
                    polygon_list.append((north, east))
                #ax.annotate(str(depth_idx), xy=(np.mean(x),np.mean(y)))
                depth_idx += 1
                depth_polygons.append((polygon_list, depth2))
            elif polygon.geom_type == 'MultiPolygon':
                for poly in polygon.geoms:
                    x, y = poly.exterior.xy
                    #ax.annotate(str(depth_idx), xy=(np.mean(x),np.mean(y)))
                    depth_idx += 1
                    polygon_list = []
                    for i in range(len(x)):
                        north, east = lat_lon_to_north_east(y[i], x[i], self.lat0, self.lon0)
                        polygon_list.append((north, east))
                    depth_polygons.append((polygon_list, depth2))
        
        return depth_polygons
    

    def scenario_depth_lands_polygons(self):
        lands = self.scenario_to_lands_north_east()
        depths = self.scenario_to_depths_north_east()
        depths_lands = [(Polygon([(x[1], x[0]) for x in depth[0]]), depth[1]) for depth in depths] + [(Polygon([(x[1], x[0]) for x in land]), 0) for land in lands]
        return depths_lands


        


 

        
        



