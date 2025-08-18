from __future__ import division

from typing import Optional

import semantic_world.spatial_types.spatial_types as cas
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.tasks.task import WEIGHT_BELOW_CA, Task
from semantic_world.geometry import Color
from semantic_world.prefixed_name import PrefixedName
from semantic_world.spatial_types.symbol_manager import symbol_manager


class Pointing(Task):
    def __init__(self,
                 tip_link: PrefixedName,
                 goal_point: cas.Point3,
                 root_link: PrefixedName,
                 pointing_axis: cas.Vector3,
                 max_velocity: float = 0.3,
                 threshold: float = 0.01,
                 weight: float = WEIGHT_BELOW_CA,
                 name: Optional[str] = None):
        """
        Will orient pointing_axis at goal_point.
        :param tip_link: tip link of the kinematic chain.
        :param goal_point: where to point pointing_axis at.
        :param root_link: root link of the kinematic chain.
        :param pointing_axis: the axis of tip_link that will be used for pointing
        :param max_velocity: rad/s
        :param weight:
        """
        self.weight = weight
        self.max_velocity = max_velocity
        self.root = root_link
        self.tip = tip_link
        self.root_P_goal_point = god_map.world.transform(target_frame=self.root, spatial_object=goal_point).to_np()
        if name is None:
            name = f'{self.__class__.__name__}/{self.root}/{self.tip}'
        super().__init__(name=name)

        self.tip_V_pointing_axis = god_map.world.transform(target_frame=self.tip, spatial_object=pointing_axis)
        self.tip_V_pointing_axis.scale(1)

        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip)
        root_P_goal_point = symbol_manager.register_point3(name=f'{self.name}.root_P_goal_point',
                                                           provider=lambda: self.root_P_goal_point)
        root_P_goal_point.reference_frame = self.root
        tip_V_pointing_axis = cas.Vector3(self.tip_V_pointing_axis)

        root_V_goal_axis = root_P_goal_point - root_T_tip.to_position()
        root_V_goal_axis.scale(1)
        root_V_pointing_axis = root_T_tip.dot(tip_V_pointing_axis)
        root_V_pointing_axis.vis_frame = self.tip
        root_V_goal_axis.vis_frame = self.tip
        god_map.debug_expression_manager.add_debug_expression('root_V_pointing_axis',
                                                              root_V_pointing_axis,
                                                              color=Color(1, 0, 0, 1))
        god_map.debug_expression_manager.add_debug_expression('goal_point',
                                                              root_P_goal_point,
                                                              color=Color(0, 0, 1, 1))
        god_map.debug_expression_manager.add_debug_expression('root_V_goal_axis',
                                                              root_V_goal_axis,
                                                              color=Color(0, 1, 0, 1))
        self.add_vector_goal_constraints(frame_V_current=root_V_pointing_axis,
                                         frame_V_goal=root_V_goal_axis,
                                         reference_velocity=self.max_velocity,
                                         weight=self.weight)
        self.observation_expression = cas.less_equal(cas.angle_between_vector(root_V_pointing_axis, root_V_goal_axis), threshold)


class PointingCone(Task):
    def __init__(self,
                 tip_link: PrefixedName,
                 goal_point: cas.Point3,
                 root_link: PrefixedName,
                 pointing_axis: cas.Vector3,
                 cone_theta: float = 0.0,
                 max_velocity: float = 0.3,
                 threshold: float = 0.01,
                 weight: float = WEIGHT_BELOW_CA,
                 name: Optional[str] = None):
        """
        Will orient pointing_axis at goal_point.
        :param tip_link: tip link of the kinematic chain.
        :param goal_point: where to point pointing_axis at.
        :param root_link: root link of the kinematic chain.
        :param pointing_axis: the axis of tip_link that will be used for pointing
        :param max_velocity: rad/s
        :param weight:
        """
        self.weight = weight
        self.max_velocity = max_velocity
        self.root = root_link
        self.tip = tip_link
        self.root_P_goal_point = god_map.world.transform(target_frame=self.root, spatial_object=goal_point).to_np()
        if name is None:
            name = f'{self.__class__.__name__}/{self.root}/{self.tip}'
        super().__init__(name=name)

        self.tip_V_pointing_axis = god_map.world.transform(target_frame=self.tip, spatial_object=pointing_axis)
        self.tip_V_pointing_axis.scale(1)

        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip)
        root_P_goal_point = symbol_manager.register_point3(name=f'{self.name}.root_P_goal_point',
                                                           provider=lambda: self.root_P_goal_point)
        root_P_goal_point.reference_frame = self.root
        tip_V_pointing_axis = cas.Vector3(self.tip_V_pointing_axis)

        root_V_goal_axis = root_P_goal_point - root_T_tip.to_position()
        root_V_goal_axis.scale(1)
        root_V_pointing_axis = root_T_tip.dot(tip_V_pointing_axis)
        root_V_pointing_axis.vis_frame = self.tip
        root_V_goal_axis.vis_frame = self.tip
        god_map.debug_expression_manager.add_debug_expression('root_V_pointing_axis',
                                                              root_V_pointing_axis,
                                                              color=Color(1, 0, 0, 1))
        god_map.debug_expression_manager.add_debug_expression('goal_point',
                                                              root_P_goal_point,
                                                              color=Color(0, 0, 1, 1))

        root_V_goal_axis_proj = cas.project_to_cone(root_V_pointing_axis, root_V_goal_axis, cone_theta)
        root_V_goal_axis_proj.vis_frame = self.tip
        god_map.debug_expression_manager.add_debug_expression('cone_axis',
                                                              root_V_goal_axis,
                                                              color=Color(1, 1, 0, 1))
        god_map.debug_expression_manager.add_debug_expression('projected_axis',
                                                              root_V_goal_axis_proj,
                                                              color=Color(1, 1, 0, 1))

        self.add_vector_goal_constraints(frame_V_current=root_V_pointing_axis,
                                         frame_V_goal=root_V_goal_axis_proj,
                                         reference_velocity=self.max_velocity,
                                         weight=self.weight)
        self.observation_expression = cas.less_equal(cas.angle_between_vector(root_V_pointing_axis, root_V_goal_axis_proj), threshold)
