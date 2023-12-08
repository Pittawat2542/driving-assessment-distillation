"""
This module provides an NPC agent to control the ego vehicle
"""

from __future__ import print_function

import carla
import numpy as np
import time
import os
from agents.navigation.basic_agent import BasicAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

def get_entry_point():
    return 'NpcAgent'

class NpcAgent(AutonomousAgent):

    """
    NPC autonomous agent to control the ego vehicle
    """

    _agent = None
    _route_assigned = False

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self.track = Track.SENSORS

        self._agent = None

    def sensors(self):
        """
        Define the sensor suite required by the agent
        :return: a list containing the required sensors in the following format:
        """

        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},
        ]

        return sensors

    _last_update_time = 0
    _update_interval = 3  # 3 second interval for updates

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation. 
        """
        if not self._agent:

            # Search for the ego actor
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    break

            if not hero_actor:
                return carla.VehicleControl()

            # Add an agent that follows the route to the ego
            self._agent = BasicAgent(hero_actor, 30)

            plan = []
            prev_wp = None
            for transform, _ in self._global_plan_world_coord:
                wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
                if prev_wp:
                    plan.extend(self._agent.trace_route(prev_wp, wp))
                prev_wp = wp

            self._agent.set_global_plan(plan)

            return carla.VehicleControl()

        else:
            vehicle_control = self._agent.run_step()
            vehicle = self._agent._vehicle
            threshold = 50
            t = self._agent._vehicle.get_transform()
            obj = CarlaDataProvider.get_world().get_actors()

            if len(obj) > 1:
                distance = lambda l: np.sqrt((l.x - t.location.x) ** 2 + (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2)
                objects = [(int(np.floor(distance(x.get_location()))), x.type_id)
                          for x in obj
                          if x.id != self._agent._vehicle.id and (distance(x.get_location()) <= threshold) and 'static' not in x.type_id and 'sensor' not in x.type_id]

            # print(f"count: {len(objects)}")
            # print(f"objects: {objects}")
            # print(f"Throttle: {vehicle_control.throttle:.2f}, Steer: {vehicle_control.steer:.2f}, H_Brake: {vehicle_control.hand_brake}, Brake: {vehicle_control.brake:.2f}, Speed: {vehicle.get_velocity().length():.2f}, Spd_Limit: {vehicle.get_speed_limit():.2f}")
            ctrl_str = "['throttle':{}, 'steer':{}, 'brake':{}, 'speed':{}, 'speedLimit':{}]\n\n".format(
                        max(vehicle_control.throttle, 0),
                        max(-1, min(1, vehicle_control.steer)),
                        vehicle_control.hand_brake or vehicle_control.brake > 0,
                        vehicle.get_velocity().length(),
                        vehicle.get_speed_limit()
            )

            file_prefix = os.environ.get('OBJ_ID')  # Prefix for the file name
            directory = './objects/'  # Directory for the output files

            # Create the directory if it doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Generating the file name dynamically using the counter
            file_name = f'{file_prefix}.txt'
            file_path = os.path.join(directory, file_name)

            # Check if one second has passed since the last update
            current_time = time.time()
            if current_time - self._last_update_time >= self._update_interval:
                # Update the last update time
                self._last_update_time = current_time
                # Write to the file
                with open(file_path, 'a') as f:
                    f.write(
                        "count: {}\n"
                        "objects: {}\n"
                        "ctrl: {}".format(
                            len(objects),
                            objects,
                            ctrl_str
                        )
                    )
            return vehicle_control