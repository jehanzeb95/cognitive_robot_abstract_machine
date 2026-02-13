from dataclasses import dataclass

import numpy as np

from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.exceptions import NodeInitializationError
from giskardpy.motion_statechart.graph_node import (
    Goal,
    Task,
    NodeArtifacts,
    DebugExpression,
)
from giskardpy.motion_statechart.context import BuildContext
from semantic_digital_twin.spatial_types import (
    Point3,
    RotationMatrix,
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class AlignToPushDoor(Goal):
    root_link: Body
    tip_link: Body
    door_object: Body
    door_handle: Body
    tip_gripper_axis: Vector3
    distance_threshold: float = 0.01
    angle_threshold: float = 0.01
    reference_linear_velocity: float = 0.1
    reference_angular_velocity: float = 0.5
    intermediate_point_scale: float = 0.75
    weight: float = DefaultWeights.WEIGHT_BELOW_CA

    def expand(self, context: BuildContext) -> None:
        """
        Instantiate and add the alignment task as a child node
        """
        # Get door joint information
        door_joint = context.world.compute_parent_connection(self.door_object)
        door_dofs = list(door_joint.dofs)  # Convert set to list
        object_joint_angle = context.world.state[door_dofs[0].id].position
        joint_limits = door_dofs[0].limits

        # Check if door is open enough before creating task
        minimum_angle_to_push_door = joint_limits.upper.position / 4

        if object_joint_angle >= minimum_angle_to_push_door:
            # Create alignment task
            align_to_push_task = Task(name="align_to_push_door")
            self.add_node(align_to_push_task)
        else:
            raise NodeInitializationError(node=self, reason="target door is not open")

    def build(self, context: BuildContext) -> NodeArtifacts:
        """
        The objective is to reach an intermediate point before pushing the door
        """
        # Resolve links
        root = self.root_link
        tip = self.tip_link
        handle = self.door_handle
        door_object = self.door_object

        # Get door joint information
        door_joint = context.world.compute_parent_connection(door_object)
        door_dofs = list(door_joint.dofs)  # Convert set to list
        object_joint_angle = context.world.state[door_dofs[0].id].position

        # Normalize tip gripper axis - directly use the Vector3
        tip_gripper_axis = self.tip_gripper_axis
        tip_gripper_axis.scale(1)

        # Get door rotation axis from the connection, not the DOF
        # The axis is a property of the joint/connection, typically stored in the connection
        # For a revolute joint, we need to get the axis from the connection
        if hasattr(door_joint, "axis"):
            door_joint_axis = door_joint.axis
        else:
            # Fallback: assume Z axis if not specified
            door_joint_axis = [0, 0, 1]

        object_V_object_rotation_axis = Vector3.from_iterable(door_joint_axis)

        # Compute forward kinematics expressions
        root_T_tip = context.world.compose_forward_kinematics_expression(root, tip)
        root_T_door_expr = context.world.compose_forward_kinematics_expression(
            root, door_object
        )

        # Compute vectors in root frame - directly use tip_gripper_axis
        tip_V_tip_grasp_axis = tip_gripper_axis
        root_V_object_rotation_axis = root_T_door_expr @ object_V_object_rotation_axis
        root_V_tip_grasp_axis = root_T_tip @ tip_V_tip_grasp_axis

        # Compute intermediate point
        root_T_handle_numeric = context.world.compute_forward_kinematics(root, handle)
        root_T_door_numeric = context.world.compute_forward_kinematics(
            root, door_object
        )
        door_T_handle_numeric = root_T_door_numeric.inverse() @ root_T_handle_numeric
        door_P_handle_numeric = door_T_handle_numeric.to_position()

        temp_point = np.asarray(
            [
                float(door_P_handle_numeric.x),
                float(door_P_handle_numeric.y),
                float(door_P_handle_numeric.z),
            ]
        )
        door_P_intermediate_point = np.zeros(3)

        # Axis pointing in the direction of handle frame from door joint frame
        direction_axis = np.argmax(np.abs(temp_point))
        door_P_intermediate_point[direction_axis] = (
            temp_point[direction_axis] * self.intermediate_point_scale
        )
        door_P_intermediate_point = Point3(
            x=door_P_intermediate_point[0],
            y=door_P_intermediate_point[1],
            z=door_P_intermediate_point[2],
        )

        # Point w.r.t door
        desired_angle = object_joint_angle * 0.5

        # Find point w.r.t rotated door in local frame
        door_R_door_rotated = RotationMatrix.from_axis_angle(
            axis=object_V_object_rotation_axis, angle=desired_angle
        )
        door_T_door_rotated = (
            HomogeneousTransformationMatrix.from_point_rotation_matrix(
                point=None, rotation_matrix=door_R_door_rotated
            )
        )

        # As the root_T_door is already pointing to a completely rotated door,
        # we invert desired angle to get to the intermediate point
        door_rotated_P_top = door_T_door_rotated.inverse() @ door_P_intermediate_point
        root_P_top = (
            HomogeneousTransformationMatrix.from_point_rotation_matrix(
                point=None, rotation_matrix=root_T_door_expr.to_rotation_matrix()
            )
            @ door_rotated_P_top
        )

        # Set observation expression for goal completion
        dist = (root_T_tip.to_position() - root_P_top).norm()
        angle = root_V_tip_grasp_axis.angle_between(root_V_object_rotation_axis)

        # Create Scalar comparisons properly - use sm module
        import krrood.symbolic_math.symbolic_math as sm

        observation_expr = sm.Scalar(dist) <= sm.Scalar(self.distance_threshold)
        observation_expr = observation_expr & (
            sm.Scalar(angle) <= sm.Scalar(self.angle_threshold)
        )

        # Create debug expressions
        debug_expressions = [
            DebugExpression(
                name="goal_point", expression=root_P_top, color=Color(0, 0.5, 0.5, 1)
            ),
            DebugExpression(
                name="root_V_grasp_axis",
                expression=root_V_tip_grasp_axis,
                color=Color(0, 0, 1, 1),
            ),
            DebugExpression(
                name="root_V_object_axis",
                expression=root_V_object_rotation_axis,
                color=Color(1, 0, 0, 1),
            ),
        ]

        return NodeArtifacts(
            observation=observation_expr, debug_expressions=debug_expressions
        )
