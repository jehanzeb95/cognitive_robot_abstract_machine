from typing import Dict, Tuple, DefaultDict, List, Set, Optional, Iterable

import betterpybullet as bpb
from line_profiler import profile

from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.model.bpb_wrapper import create_shape_from_link, create_collision
from giskardpy.model.collision_detector import CollisionDetector, Collisions
from giskardpy.model.collision_matrix_manager import CollisionCheck
from semantic_world.prefixed_name import PrefixedName
from semantic_world.world_entity import Body


class BulletCollisionDetector(CollisionDetector):
    collision_list_sizes: int = 1000

    def __init__(self, ):
        self.kw = bpb.KineverseWorld()
        self.body_to_bpb_obj: Dict[Body, bpb.CollisionObject] = {}
        self.query: Optional[DefaultDict[PrefixedName, Set[Tuple[bpb.CollisionObject, float]]]] = None
        super().__init__()

    @profile
    def add_object(self, body: Body):
        if not body.has_collision() or body.collision_config.disabled:
            return
        o = create_shape_from_link(body)
        self.kw.add_collision_object(o)
        self.body_to_bpb_obj[body] = o

    def reset_cache(self):
        self.query = None

    @profile
    def cut_off_distances_to_query(self, collision_matrix: Set[CollisionCheck],
                                   buffer: float = 0.05) -> DefaultDict[
        PrefixedName, Set[Tuple[bpb.CollisionObject, float]]]:
        if self.query is None:
            self.query = {(self.body_to_bpb_obj[check.body_a],
                           self.body_to_bpb_obj[check.body_b]): check.distance + buffer for check in
                          collision_matrix}
        return self.query

    def check_collisions(self,
                         collision_matrix: Set[CollisionCheck],
                         buffer: float = 0.05) -> Collisions:

        query = self.cut_off_distances_to_query(collision_matrix, buffer=buffer)
        result: List[bpb.Collision] = self.kw.get_closest_filtered_map_batch(query)
        self.closest_points = self.bpb_result_to_collisions(result, self.collision_list_sizes)
        return self.closest_points

    @profile
    def find_colliding_combinations(self, body_combinations: Iterable[Tuple[Body, Body]],
                                    distance: float,
                                    update_query: bool) -> Set[CollisionCheck]:
        if update_query:
            self.query = None
            self.collision_matrix = {CollisionCheck(body_a=body_a,
                                                    body_b=body_b,
                                                    distance=distance) for body_a, body_b in body_combinations}
        else:
            self.collision_matrix = set()
        god_map.collision_scene.sync()
        collisions = self.check_collisions(collision_matrix=self.collision_matrix, buffer=0.0)
        colliding_combinations = {CollisionCheck(body_a=c.original_link_a,
                                                 body_b=c.original_link_b,
                                                 distance=c.contact_distance) for c in collisions.all_collisions
                                  if c.contact_distance <= distance}
        return colliding_combinations

    @profile
    def bpb_result_to_collisions(self, result: List[bpb.Collision],
                                 collision_list_size: int) -> Collisions:
        collisions = Collisions(collision_list_size)

        for collision in result:
            giskard_collision = create_collision(collision, god_map.world)
            collisions.add(giskard_collision)
        return collisions

    # def check_collision(self, link_a, link_b, distance):
    #     self.sync()
    #     query = defaultdict(set)
    #     query[self.object_name_to_id[link_a]].add((self.object_name_to_id[link_b], distance))
    #     return self.kw.get_closest_filtered_POD_batch(query)

    def sync_world_model(self) -> None:
        self.reset_cache()
        get_middleware().logdebug('hard sync')
        for o in self.kw.collision_objects:
            self.kw.remove_collision_object(o)
        self.body_to_bpb_obj = {}
        self.objects_in_order = []

        for body in sorted(god_map.world.bodies_with_enabled_collision, key=lambda b: b.name):
            self.add_object(body)
            self.objects_in_order.append(self.body_to_bpb_obj[body])

    def sync_world_state(self) -> None:
        bpb.batch_set_transforms(self.objects_in_order,
                                 god_map.world.compute_forward_kinematics_of_all_collision_bodies())

    @profile
    def get_map_T_geometry(self, body: Body, collision_id: int = 0):
        collision_object = self.body_to_bpb_obj[body]
        return collision_object.compound_transform(collision_id)
