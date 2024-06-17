import gymnasium as gym
import numpy as np
import pygame

from gymnasium import spaces
from scm_irl.utils.process_scenario import Scenario
from scm_irl.utils.plot_scenario import cmap_seachart, apply_cmap, create_color_map
from sllib.datatypes.vessel import VesselState


from shapely.geometry import box
from shapely.ops import cascaded_union
from shapely.ops import unary_union

import shapely
import cv2
from rasterio.features import rasterize
from rasterio.transform import from_origin

from shapely.affinity import rotate
from shapely.geometry import Polygon
from shapely.ops import unary_union, polygonize
from shapely.geometry import MultiPolygon
from scipy import ndimage
import math
import random


def modulate_color(color, modulation):
    return [int(c * modulation) for c in color]



class ScmIrlEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, cfg=None, scenario_path=None, render_mode=None, start_time_reference=None, end_time_override=None, mmsi=None, mmsi_in_collision=False, awareness_zone = [1000, 1000, 1000, 1000], resolution=1, dict_scenarios={}):
        self.cfg = cfg
        self.episode_number = 0
        self.render_mode = render_mode

        self.scenario_path = scenario_path

        self.scenario = Scenario(cfg, scenario_path)

        self.sampling_time = self.scenario.sampling_time

        mmsis = []
        if mmsi_in_collision:
            mmsis = self.scenario.collision_mmsis
        else:
            mmsis = self.scenario.mmsis
            

        if mmsi is None:
            mmsi = self.select_valid_mmsi(mmsis, scenario_path, dict_scenarios, self.cfg)

        self.mmsi = mmsi # Return None if no valid mmsi is found

        if self.mmsi in mmsis:
            self.vessel_metadata = self.scenario.get_vessel_metadata(self.mmsi)

            if start_time_reference is None:
                self.start_time = next(iter(self.scenario.vessels[mmsi]['states']))
            else:
                self.start_time = start_time_reference

            self.timestep = self.start_time

            self.end_time = next(reversed(self.scenario.vessels[mmsi]['states'])) 
            if end_time_override is not None:
                self.end_time = end_time_override
            
            print(f"mmsi: {self.mmsi}, start_time: {self.start_time}, end_time: {self.end_time}")

            self.agent_state = self.scenario.get_vessel_state_time(self.mmsi, self.start_time)

            self.agent_final_location = self.scenario.get_vessel_state_time(self.mmsi, self.end_time)
            #self.agent_final_location = np.array([self.agent_final_location.lat, self.agent_final_location.lon])

            assert render_mode is None or render_mode in self.metadata["render_modes"]
            self.render_mode = render_mode

            # front, back, left, right
            self.awareness_zone = awareness_zone

            self.resolution = resolution
            self.nodes, self.ways = self.scenario.get_nodes_and_ways_scenario_north_east()
            self.observation_matrix = self._get_observation_matrix()

            self.cmap = cmap_seachart()

            self.color_map = create_color_map()
            _, self.nodes_color = self.scenario.nodes_list()
            # append to color_map the nodes colors
            self.color_map = np.concatenate((self.color_map, self.nodes_color), axis=0)

            # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
            # self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
            # self.state = None
            # self.timestep = 0
            # self.truncated = False
            # self.reward = 0
            # self.info = {}

            self.sog_scale = cfg['env']['sog_scale']
            self.cog_scale = cfg['env']['cog_scale'] * np.pi

            self.bicycle_model = cfg['env']['bicycle_model']
            self.copy_expert = cfg['env']['copy_expert']
            self.dist_metric = cfg['env']['dist_metric']
            self.start_random = cfg['env']['start_random']


            self.truncated = False
            self.terminate = False
            self.done = False
            self.info = {}

            # self.observation_space = spaces.Dict({
            #     'observation_matrix': spaces.Box(low=0, high=255, shape=self.observation_matrix.shape, dtype=np.float32),
            #     'agent_state': spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            #     'expert_state': spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            #     'target': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
            # })
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
            #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)


            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

            
            self.window_size = (700,500)  # The size of the PyGame window
            self.padding = 20  # The padding around the grid
            self.window_size_total = (self.window_size[0] + 3*self.padding, self.window_size[1] + 2* self.padding)
            self.panel_size = 200  # The size of the panel for additional graphics
            self.window_scenario_size = (self.window_size[0] - self.panel_size, self.window_size[1])
            self.window_observation_size = (self.panel_size, int(self.window_size[1]/2))

            self.window = None
            self.clock = None


    def select_valid_mmsi(self, mmsis, scenario_path, dict_scenarios, cfg):

        # Filter MMSIs based on ship type being in valid vessels
        valid_mmsis = [mmsi for mmsi in mmsis if self.scenario.get_vessel_metadata(mmsi).ship_type in cfg['env']['valid_vessels']]
        
        if not valid_mmsis:
            print(f"no valid mmsi in the scenario {scenario_path}")
            return None

        # Exclude already used MMSIs if dict_scenarios is provided
        if self.scenario.scenario_id not in dict_scenarios:
            return np.random.choice(valid_mmsis)
        
        mmsi_used = dict_scenarios.get(self.scenario.scenario_id, [])
        remaining_mmsis = [mmsi for mmsi in valid_mmsis if mmsi not in mmsi_used]
        
        if not remaining_mmsis:
            print(f"no more mmsi to use in the scenario {scenario_path}")
            return None

        # Randomly select from the remaining valid MMSIs
        return np.random.choice(remaining_mmsis)     
    
    def scale_coords(self, coords, north_min, north_max, east_min, east_max, height, width):
        north_range = north_max - north_min
        east_range = east_max - east_min
        return [(width * (x - east_min) / east_range, height * (1 - (y - north_min) / north_range)) for x, y in coords]


    def get_vessel_metadata(self):
        return self.scenario.get_vessel_metadata(self.mmsi)

    def get_action_from_vessel(self, timestep):
        state = self.scenario.get_vessel_state_time(self.mmsi, timestep)
        return (state.sog/self.sog_scale, state.cog/self.cog_scale)


    def _pre_step(self):
        north, east = self.agent_state.lat , self.agent_state.lon
        #print(self.action)
        sog_target, cog_target = self.action[0] * self.sog_scale, self.action[1] * self.cog_scale

        sog_initial = self.agent_state.sog
        cog_initial = self.agent_state.cog

        if self.bicycle_model:
            L =  self.vessel_metadata.length
            
            delta_cog = cog_target - cog_initial
            delta_sog = sog_target - sog_initial

            beta = np.arctan(np.tan(delta_cog)/2)
            self.timestep += self.sampling_time
            north += np.cos(cog_initial + beta) * sog_target * self.scenario.sampling_time
            east += np.sin(cog_initial + beta) * sog_target * self.scenario.sampling_time
            sog = sog_target
            cog = cog_initial + sog*np.tan(delta_cog)*np.cos(beta)/L * self.scenario.sampling_time
        else:
            self.timestep += self.sampling_time
            north += np.cos(cog_target) * sog_target * self.scenario.sampling_time
            east += np.sin(cog_target) * sog_target * self.scenario.sampling_time
            sog = sog_target
            cog = cog_target
        
        if self.copy_expert:
            expert_state = self.scenario.get_vessel_state_time(self.mmsi, self.timestep)
            north, east = expert_state.lat, expert_state.lon
            sog = expert_state.sog
            cog = expert_state.cog
        
        self.agent_state = VesselState(timestamp=self.timestep - self.start_time, lat=north, lon=east, sog=sog, cog=cog)



    def _get_obs(self):
        observation_matrix = self._get_observation_matrix()

        if not np.isnan(observation_matrix).any():
            self.observation_matrix = observation_matrix
            # scale channel 0 and 1 to 255
            self.observation_matrix[:,:,0] = self.observation_matrix[:,:,0] * 255 / self.cfg['env']['vessel_types_max']
            self.observation_matrix[:,:,1] = self.observation_matrix[:,:,1] * 255 / self.cfg['env']['seamark_max']



        #agent_obs = np.array([self.agent_state.lat, self.agent_state.lon, self.agent_state.sog, self.agent_state.cog])
        
        expert_state = self.scenario.get_vessel_state_time(self.mmsi, self.timestep)
        expert_obs = np.array([expert_state.lat, expert_state.lon, expert_state.sog/self.sog_scale, expert_state.cog/self.cog_scale])
        # expert_action = np.array([expert_state.sog/self.sog_scale, expert_state.cog/self.cog_scale])
        
        agent_obs = self.scenario.relative_state(expert_state, self.agent_state)
        agent_obs = np.array([agent_obs.lat, agent_obs.lon, agent_obs.sog, agent_obs.cog])
        
        target_state = self.scenario.relative_state(self.agent_state, self.agent_final_location)
        target_state = np.array([target_state.lat, target_state.lon])
        
        # obs = {'observation_matrix': self.observation_matrix,
        #         'agent_state': agent_obs, 
        #         'expert_state': expert_obs,
        #         'target': target_state}
      
        obs = {'agent_state': agent_obs, 
                'target': target_state}
        #obs = {'expert_action': expert_action}

        transformed_obs_values = [np.array(v) for v in obs.values()]

        # Concatenate all values into a single vector
        obs = np.concatenate(transformed_obs_values)
        # add one level of nesting
        #obs = np.expand_dims(obs, axis=0)

        terminate = False
        if self.observation_matrix is None or np.isnan(agent_obs).any() or np.isnan(observation_matrix).any():
            print(f"######## NAN agent_obs: {agent_obs}, observation_matrix: {observation_matrix}")
            terminate = True

        return obs, terminate
 
    def _compute_reward(self):
        # compute the 
        expert_state = self.scenario.get_vessel_state_time(self.mmsi, self.timestep)
        # compute the distance between the expert and the agent
        
        if self.dist_metric == "l2":
            pos_error = np.sqrt((expert_state.lat - self.agent_state.lat)**2 + (expert_state.lon - self.agent_state.lon)**2)
        elif self.dist_metric == "l1":
            pos_error = np.abs(expert_state.lat - self.agent_state.lat) + np.abs(expert_state.lon - self.agent_state.lon)
        else:
            raise ValueError(f"dist_metric {self.dist_metric} is not valid")
               
        #distance_to_target = np.sqrt((self.agent_final_location.lat - self.agent_state.lat)**2 + (self.agent_final_location.lon - self.agent_state.lon)**2)
        cog_diff = np.abs(expert_state.cog - self.agent_state.cog)
        #cog_diff = np.minimum(cog_diff, 2*np.pi - cog_diff)
        sog_diff = np.abs(expert_state.sog - self.agent_state.sog)

        reward = np.exp(-pos_error*0.1) # + 0.5 * np.exp(- cog_diff) + 0.5 * np.exp(- sog_diff)
        #print(f"timestep {self.timestep}, reward: {reward}, pos_error: {pos_error}, expert_state: {expert_state}, agent_state: {self.agent_state}")
        #reward = - pos_error

        #print(f"pos_error {pos_error} : {np.exp(- pos_error/10000)}, cog_diff: {cog_diff}, sog_diff: {sog_diff}")
        if np.isnan(reward):
            print("reward is nan")
            reward = -1
            

        return reward
    

    def step(self, action):
        # print(f"timestep: {self.timestep}, action: {action}")
        self.action = action
        self._pre_step()
        
        obs, term = self._get_obs()
        self.reward = self._compute_reward()



        # terminate if the agent out of the scenario
        if (self.agent_state.lat < self.scenario.north_min 
            or self.agent_state.lat > self.scenario.north_max 
            or self.agent_state.lon < self.scenario.east_min 
            or self.agent_state.lon > self.scenario.east_max 
            or term):
            self.truncated = False
            self.terminate = True
            print("agent out of the scenario")
            self.reward = -10
        elif self.timestep >= self.end_time:
            self.truncated = True
            self.terminate = True
        else:
            self.truncated = False
            self.terminate = False
            if self.render_mode == "human":
                self.render()
        #self.info = {}
        self.done = self.terminate or self.truncated

        self.info = {"rollout": {"obs": obs, "rews": self.reward, "done": self.done}}


        return obs, self.reward, self.terminate, self.truncated, self.info
                     

    def _get_observation_matrix(self):
        crop_box = self._create_crop_box(angle=90 - self.agent_state.cog*180/np.pi)
        
        depths_lands_inside = self.scenario.depth_lands_polygons.copy()
        nodes_ways_inside = []
        vessels_inside = []

        # add the nodes to the depths_lands_inside
        for node in self.nodes:
            node_polygon = box(node['east'] - 5, node['north'] - 5, node['east'] + 5, node['north'] + 5)
            nodes_ways_inside.append((node_polygon, node['value']))

        # add ways to the depths_lands_inside
        for way in self.ways:
            way_polygon = shapely.geometry.LineString([(node[1], node[0]) for node in way['nodes']])
            nodes_ways_inside.append((way_polygon, self.cfg['env']['seamarks']['way']['value']))

        # add the vessel polygon to the depths_lands_inside                
        metadata = self.scenario.get_vessel_metadata(self.mmsi)
        vessel_polygon = self._render_vessel(metadata, self.agent_state) 
        vessels_inside.append((vessel_polygon, 1))
                
        for vessel_mmsi in self.scenario.mmsis:
            if vessel_mmsi != self.mmsi and self.scenario.is_vessel_active(vessel_mmsi, self.timestep):
                metadata = self.scenario.get_vessel_metadata(vessel_mmsi)
                vessel_states = self.scenario.get_vessel_state_time(vessel_mmsi, self.timestep)
                vessel_polygon = self._render_vessel(metadata, vessel_states) 
                vessels_inside.append((vessel_polygon, 1))

        depths_lands_inside = [(polygon.intersection(crop_box), depth) for polygon, depth in depths_lands_inside if crop_box.intersects(polygon)]
        nodes_ways_inside = [(polygon.intersection(crop_box), depth) for polygon, depth in nodes_ways_inside if crop_box.intersects(polygon)]
        vessels_inside = [(polygon.intersection(crop_box), depth) for polygon, depth in vessels_inside if crop_box.intersects(polygon)]
        
        # rotate the depths_lands_inside polygons to the cog of the vessel
        depths_lands_inside = [(rotate(polygon, self.agent_state.cog*180/np.pi, origin=(self.agent_state.lon, self.agent_state.lat)), depth) for polygon, depth in depths_lands_inside]
        nodes_ways_inside = [(rotate(polygon, self.agent_state.cog*180/np.pi, origin=(self.agent_state.lon, self.agent_state.lat)), depth) for polygon, depth in nodes_ways_inside]
        vessels_inside = [(rotate(polygon, self.agent_state.cog*180/np.pi, origin=(self.agent_state.lon, self.agent_state.lat)), depth) for polygon, depth in vessels_inside]

        if len(depths_lands_inside) == 0:
            return None

        crop_box = self._create_crop_box(angle=90)
        min_x, min_y, max_x, max_y = crop_box.bounds

        epsilon = 1e-10  # a small value

        # Determine the number of rows and columns of the raster
        num_rows = math.ceil((max_y - min_y - epsilon) / self.resolution)
        num_cols = math.ceil((max_x - min_x - epsilon) / self.resolution)

        # Create a transformation matrix
        transform = from_origin(min_x, max_y, self.resolution, self.resolution)
        # Create the matrix
        #print(depths_lands_inside)
        matrix_depth_land = rasterize(depths_lands_inside, out_shape=(num_rows, num_cols), transform=transform, fill=np.nan)
        
        #if nodes_ways_inside is not empty
        if len(nodes_ways_inside) > 0:
            matrix_nodes_ways = rasterize(nodes_ways_inside, out_shape=(num_rows, num_cols), transform=transform, fill=0)
        else:
            matrix_nodes_ways = np.zeros((num_rows, num_cols))
        
        matrix_vessels = rasterize(vessels_inside, out_shape=(num_rows, num_cols), transform=transform, fill=0)

        matrix_depth_land[-1,:] = (matrix_depth_land[-2,:] + matrix_depth_land[-3,:])/2
        matrix_depth_land[:,-1] = (matrix_depth_land[:,-2] + matrix_depth_land[:,-3])/2
        matrix_depth_land[0,:] = (matrix_depth_land[1,:] + matrix_depth_land[2,:])/2
        matrix_depth_land[:,0] = (matrix_depth_land[:,1] + matrix_depth_land[:,2])/2

        # Convert the matrix to a floating-point image
        matrix_depth_land_float = matrix_depth_land.astype(np.float32)

        # get only positive and non-nan values
        valid_mask = np.logical_and(matrix_depth_land_float > 0, ~np.isnan(matrix_depth_land_float))

        # Get the coordinates of the nearest valid cell for each missing cell
        nearest_valid_coords = np.array(ndimage.distance_transform_edt(~valid_mask, return_distances=False, return_indices=True))

        # Create a copy of the original matrix
        matrix_depth_land_filled = np.copy(matrix_depth_land_float)

        # Replace only the nan values in the original matrix with the values of the nearest valid cells
        matrix_depth_land_filled[np.isnan(matrix_depth_land_float)] = matrix_depth_land_float[tuple(nearest_valid_coords[:, np.isnan(matrix_depth_land_float)])]

        # Normalize the matrix
        matrix_depth_land_filled = matrix_depth_land_filled * 255 / self.cfg['env']['depth_max']


        # concat matrix in a way that they form a set 3 chanels
        matrix_concat = np.stack((matrix_vessels, matrix_nodes_ways, matrix_depth_land_filled), axis=-1)

        # flip the matrix horizontally
        matrix_concat = cv2.flip(matrix_concat, 1)
        # rotate the matrix to the original orientation
        matrix_concat = cv2.rotate(matrix_concat, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return matrix_concat




    def _create_crop_box(self, angle = 0): 

        min_e = self.agent_state.lon - self.awareness_zone[0]
        max_e = self.agent_state.lon + self.awareness_zone[1]
        min_n = self.agent_state.lat - self.awareness_zone[2]
        max_n = self.agent_state.lat + self.awareness_zone[3]

        crop_box = box(min_e, min_n, max_e, max_n)

        # center_e = (min_e + max_e) / 2
        # center_n = (min_n + max_n) / 2
        rotated_crop_box = rotate(crop_box, angle, origin=(self.agent_state.lon, self.agent_state.lat))
        return rotated_crop_box


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # self.start_time
        # self.end_time
        # self.sampling_time
        if self.start_random:
            time_steps = list(range(int(self.start_time), int(self.end_time - 2 * self.sampling_time ), int(self.sampling_time)))
            random_time = random.choice(time_steps)
            self.timestep = random_time

        else:
            self.timestep = self.start_time
        

        self.agent_state = self.scenario.get_vessel_state_time(self.mmsi, self.timestep)
        self.truncated = False
        self.terminate = False
        self.done = False
        self.reward = 0
        obs,_ = self._get_obs()
        #print(f'Episode number {self.episode_number} started')
        self.episode_number += 1

        return obs, self.info
    

    def _render_scenario(self):
        north_range = self.scenario.north_max - self.scenario.north_min
        east_range = self.scenario.east_max - self.scenario.east_min
        aspect_ratio = east_range / north_range

        # Calculate the dimensions of the new canvas
        if aspect_ratio > 1:
            # Scenario is wider than it is tall
            width = self.window_scenario_size[0]
            height = int(self.window_scenario_size[1] / aspect_ratio)
        else:
            # Scenario is taller than it is wide
            width = int(self.window_scenario_size[0] * aspect_ratio)
            height = self.window_scenario_size[1]


        canvas_aratio = pygame.Surface((width, height))



        for depth in self.scenario.depth_lands_polygons:
            # Convert and scale the coordinates
            coords = self.scale_coords([(x[0], x[1]) for x in depth[0].exterior.coords], 
                                self.scenario.north_min, self.scenario.north_max, 
                                self.scenario.east_min, self.scenario.east_max, 
                                height, width)
            
            #color = (0, 0, 255)
            # Modulate the color based on the depth
            #color = modulate_color(color, 1 - depth[1] / 100)
            color = self.color_map[int(depth[1]*255/self.cfg['env']['depth_max'])]
            
            # Draw the polygon
            pygame.draw.polygon(canvas_aratio, color, coords)

        # Draw the crop box
        crop_box = self._create_crop_box(angle=90 - self.agent_state.cog*180/np.pi)
        coords = self.scale_coords([(x[0], x[1]) for x in crop_box.exterior.coords],
                            self.scenario.north_min, self.scenario.north_max,
                            self.scenario.east_min, self.scenario.east_max,
                            height, width)
        pygame.draw.polygon(canvas_aratio, (255, 0, 0), coords, 2)

        # draw vessel: 
        metadata = self.scenario.get_vessel_metadata(self.mmsi)
        vessel_states = self.scenario.get_vessel_state_time(self.mmsi, self.timestep)
        vessel_polygon = self._render_vessel(metadata, vessel_states) 
        coords = self.scale_coords([(x[0], x[1]) for x in vessel_polygon.exterior.coords],
                            self.scenario.north_min, self.scenario.north_max,
                            self.scenario.east_min, self.scenario.east_max,
                            height, width)
        pygame.draw.lines(canvas_aratio, (0, 160, 0), True, coords, 2)
        
        for vessel_mmsi in self.scenario.mmsis:
            if vessel_mmsi != self.mmsi and self.scenario.is_vessel_active(vessel_mmsi, self.timestep):
                metadata = self.scenario.get_vessel_metadata(vessel_mmsi)
                vessel_states = self.scenario.get_vessel_state_time(vessel_mmsi, self.timestep)
                vessel_polygon = self._render_vessel(metadata, vessel_states) 
                coords = self.scale_coords([(x[0], x[1]) for x in vessel_polygon.exterior.coords],
                            self.scenario.north_min, self.scenario.north_max,
                            self.scenario.east_min, self.scenario.east_max,
                            height, width)
                pygame.draw.lines(canvas_aratio, (0, 0, 0), True, coords, 2)


        # draw agent: 
        metadata = self.scenario.get_vessel_metadata(self.mmsi)
        vessel_polygon = self._render_vessel(metadata, self.agent_state) 
        coords = self.scale_coords([(x[0], x[1]) for x in vessel_polygon.exterior.coords],
                            self.scenario.north_min, self.scenario.north_max,
                            self.scenario.east_min, self.scenario.east_max,
                            height, width)
        pygame.draw.lines(canvas_aratio, (0, 0, 0), True, coords, 2)

        for way in self.ways:
            coords = self.scale_coords([(node[1], node[0]) for node in way['nodes']],
                            self.scenario.north_min, self.scenario.north_max,
                            self.scenario.east_min, self.scenario.east_max,
                            height, width)
            pygame.draw.lines(canvas_aratio, tuple(way['color']), False, coords, 1)

        
        for node in self.nodes:
            coords = self.scale_coords([(node['east'], node['north'])],
                            self.scenario.north_min, self.scenario.north_max,
                            self.scenario.east_min, self.scenario.east_max,
                            height, width)
            pygame.draw.circle(canvas_aratio, tuple(node['color']), coords[0], 1)   

        # Create a new square canvas and fill it with black
        canvas_scenario = pygame.Surface(self.window_scenario_size)
        canvas_scenario.fill((0, 0, 0))  # RGB for black

        # Calculate the size of the original surface
        original_width, original_height = canvas_aratio.get_size()

        # Calculate the position to blit the original surface onto the new one
        position = ((self.window_scenario_size[0] - original_width) // 2, (self.window_scenario_size[1] - original_height) // 2)

        # Blit the original surface onto the new one
        canvas_scenario.blit(canvas_aratio, position)


        #if self.render_mode == "human":
        self.canvas.blit(canvas_scenario, (0, 0))  # Blit the scenario onto the left part of the window
        #pygame.event.pump()


    def _render_observartion_matrix(self):

        observation_matrix_in = self.observation_matrix

        #observation_matrix = np.copy(observation_matrix_in[:,:,0])

        # combine the 3 channels into a single channel taking the nonzero values from channel 0 and 1 and the values from channel 2
        observation_matrix = np.where(observation_matrix_in[:,:,1] > 0, - observation_matrix_in[:,:,1], observation_matrix_in[:,:,2])
        observation_matrix = np.where(observation_matrix_in[:,:,0] > 0, - observation_matrix_in[:,:,0], observation_matrix)

        # positive values only multiply by 255 
        observation_matrix[observation_matrix > 0] = observation_matrix[observation_matrix > 0]

        indices = (observation_matrix).astype(int)
        # print max and min values
        #print(np.max(indices), np.min(indices))



        color_map_array = np.array(self.color_map)
        observation_matrix_rgb = color_map_array[indices.ravel()].reshape(observation_matrix.shape + (3,))

        # negative values in the observation matrix are nodes
        # observation_matrix_rgb[observation_matrix < 0] = (255, 255, 255)

        # scale to the window_observation_size while keeping the aspect ratio
        # calculate the aspect ratio of the observation matrix
        aspect_ratio = observation_matrix.shape[1] / observation_matrix.shape[0]
        if aspect_ratio > 1:
            # Observation matrix is wider than it is tall
            width = self.window_observation_size[0]
            height = int(self.window_observation_size[1] / aspect_ratio)
        else:
            # Observation matrix is taller than it is wide
            width = int(self.window_observation_size[0] * aspect_ratio)
            height = self.window_observation_size[1]
        
        observation_matrix_rgb = cv2.resize(observation_matrix_rgb, (width, height))
        
        # Convert the RGB image to a Pygame surface
        observation_surface = pygame.surfarray.make_surface(observation_matrix_rgb)

        #return observation_surface
        
        canvas_obs_matrix = pygame.Surface(self.window_observation_size)
        canvas_obs_matrix.fill((0, 0, 0))

        original_width, original_height = canvas_obs_matrix.get_size()

        # Calculate the position to blit the original surface onto the new one
        position = ((self.window_observation_size[0] - original_width) // 2, (self.window_observation_size[1] - original_height) // 2)

        canvas_obs_matrix.blit(observation_surface, position)

        #if self.render_mode == "human":
        self.canvas.blit(canvas_obs_matrix, (self.window_scenario_size[0] + self.padding, 0))  # Blit the observation matrix onto the right part of the window
        #pygame.event.pump()
        
    def render(self):
        
        self.canvas = pygame.Surface((self.window_scenario_size[0] + self.window_observation_size[0] + self.padding, max(self.window_scenario_size[1], self.window_observation_size[1])))
        
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_scenario_size[0] + self.window_observation_size[0] + self.padding, max(self.window_scenario_size[1], self.window_observation_size[1])))
            self.clock = pygame.time.Clock()
        
        if self.render_mode == "human":
            self._render_scenario()
            self._render_observartion_matrix()

            self.window.blit(self.canvas, (0, 0))   
            pygame.event.pump()

            # Update the display only once after both methods have been called
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            self._render_scenario()
            self._render_observartion_matrix()

            # Convert the Pygame surface to an RGB array
            rgb_array = pygame.surfarray.array3d(self.canvas)

            # Transpose the array so it has the shape (height, width, 3)
            rgb_array = np.transpose(rgb_array, (1, 0, 2))

            return rgb_array

    def _render_vessel(self, metadata, vessel_states):
        length = metadata.length
        width = metadata.width

        x = vessel_states.lon
        y = vessel_states.lat
       

        vertices = [
            (x - length / 2, y - width / 2),
            (x  + length / 4, y - width / 2),
            (x + length / 2 , y),
            (x + length / 4, y + width / 2),
            (x - length / 2, y + width / 2),
            #(x, y - half_size),
        ]
        
        vessel_polygon = rotate(Polygon(vertices), 90 - vessel_states.cog*180/np.pi, origin=(x, y))
        return vessel_polygon

    def _render_full_seachart(self):
        pass



    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        
    def load_trajectories(self, scenario_path):
        pass
    
    def calculate_reward(self, action):
        pass
    