from __future__ import annotations

import json

import numpy as np
import pytest
from geometry_msgs.msg import Vector3Stamped as ROSVector3Stamped

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.ros2_nodes.topic_monitor import (
    PublishOnStart,
    WaitForMessage,
)
from giskardpy.ros_executor import Ros2Executor
from semantic_digital_twin.adapters.ros.ros_msg_serializer import (
    Ros2MessageJSONSerializer,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3, HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
    Body,
)

from giskardpy.motion_statechart.ros2_nodes.handle_offset_correction import (
    HandleOffsetCorrection,
)

_original_ros_from_json = Ros2MessageJSONSerializer.from_json.__func__


@classmethod
def _patched_ros_from_json(cls, data, clazz, **kwargs):
    return _original_ros_from_json(cls, data, clazz)


Ros2MessageJSONSerializer.from_json = _patched_ros_from_json


def _make_ros_vector3_stamped(
    x: float, y: float, z: float, frame_id: str = "map"
) -> ROSVector3Stamped:
    """Helper to build a ROSVector3Stamped message."""
    msg = ROSVector3Stamped()
    msg.header.frame_id = frame_id
    msg.vector.x = x
    msg.vector.y = y
    msg.vector.z = z
    return msg


def _make_test_world() -> (
    tuple[World, KinematicStructureEntity, KinematicStructureEntity, Connection6DoF]
):
    """
    Build a minimal World with:
      - a 'map' root body
      - a 'camera_link' child body connected via FixedConnection
      - a 'door_link' child body connected via Connection6DoF

    Returns (world, root_link, tip_link, door_connection).
    """
    world = World()

    map_body = Body(name=PrefixedName("map"))
    camera_body = Body(name=PrefixedName("camera_link"))
    door_body = Body(name=PrefixedName("door_link"))

    with world.modify_world():
        world.add_kinematic_structure_entity(map_body)

        camera_connection = FixedConnection(
            parent=map_body,
            child=camera_body,
            parent_T_connection_expression=HomogeneousTransformationMatrix(
                reference_frame=map_body, child_frame=camera_body
            ),
        )
        world.add_connection(camera_connection)

        door_connection = Connection6DoF.create_with_dofs(
            parent=map_body,
            child=door_body,
            world=world,
        )
        world.add_connection(door_connection)

    return world, map_body, camera_body, door_connection


def _make_context_with_world(world: World) -> MotionStatechartContext:
    return MotionStatechartContext(world=world)


def _msc_from_json(json_data: dict, world: World) -> MotionStatechart:
    """Deserialise a MotionStatechart, supplying the world so that entity UUIDs can be resolved."""
    tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    return MotionStatechart.from_json(json_data, **tracker.create_kwargs())


class TestHandleOffsetCorrection:
    """Tests for the HandleOffsetCorrection motion statechart node."""

    def _build_msc_with_node(
        self,
        node: HandleOffsetCorrection,
        trigger_msgs: list[ROSVector3Stamped],
        topic_name: str = "/robokudo/handle_offset",
    ) -> MotionStatechart:
        publish_nodes = [
            PublishOnStart(topic_name=topic_name, msg=msg) for msg in trigger_msgs
        ]
        wait_node = WaitForMessage(topic_name=topic_name, msg_type=ROSVector3Stamped)

        msc = MotionStatechart()
        msc.add_node(
            parallel := Parallel(
                [
                    node,
                    Sequence(nodes=[*publish_nodes, wait_node]),
                ]
            )
        )
        msc.add_node(EndMotion.when_true(parallel))
        return msc

    def test_unknown_before_first_message(self, rclpy_node):
        """Node should report UNKNOWN when no message has been received yet."""
        topic_name = "/robokudo/handle_offset"
        world, root_link, tip_link, door_connection = _make_test_world()

        node = HandleOffsetCorrection(
            topic_name=topic_name,
            root_link=root_link,
            tip_link=tip_link,
            door_move_connection=door_connection,
            goal_vector=Vector3(x=0.0, y=0.0, z=0.0, reference_frame=root_link),
            threshold=50.0,
        )

        large_offset_msg = _make_ros_vector3_stamped(100.0, 100.0, 100.0)
        msc = self._build_msc_with_node(
            node, trigger_msgs=[large_offset_msg], topic_name=topic_name
        )

        msc_copy = _msc_from_json(json.loads(json.dumps(msc.to_json())), world)

        executor = Ros2Executor(
            context=_make_context_with_world(world), ros_node=rclpy_node
        )
        executor.compile(motion_statechart=msc_copy)

        handle_node = msc_copy.nodes[0].nodes[0]
        executor.tick_until_end(timeout=5_000)

        history = msc_copy.history.get_observation_history_of_node(handle_node)
        assert history[0] == ObservationStateValues.UNKNOWN

    def test_false_when_offset_above_threshold(self, rclpy_node):
        """Node should return FALSE when the received offset magnitude > threshold."""
        topic_name = "/robokudo/handle_offset"
        world, root_link, tip_link, door_connection = _make_test_world()

        threshold = 10.0
        large_offset_msg = _make_ros_vector3_stamped(100.0, 100.0, 100.0)

        node = HandleOffsetCorrection(
            topic_name=topic_name,
            root_link=root_link,
            tip_link=tip_link,
            door_move_connection=door_connection,
            goal_vector=Vector3(x=1.0, y=0.0, z=0.0, reference_frame=root_link),
            threshold=threshold,
        )

        msc = self._build_msc_with_node(
            node, trigger_msgs=[large_offset_msg], topic_name=topic_name
        )
        msc_copy = _msc_from_json(json.loads(json.dumps(msc.to_json())), world)

        executor = Ros2Executor(
            context=_make_context_with_world(world), ros_node=rclpy_node
        )
        executor.compile(motion_statechart=msc_copy)

        handle_node = msc_copy.nodes[0].nodes[0]
        executor.tick_until_end(timeout=5_000)

        history = msc_copy.history.get_observation_history_of_node(handle_node)
        assert ObservationStateValues.FALSE in history

    def test_true_when_offset_below_threshold(self, rclpy_node):
        """Node should return TRUE once the received offset magnitude <= threshold."""
        topic_name = "/robokudo/handle_offset"
        world, root_link, tip_link, door_connection = _make_test_world()

        threshold = 50.0
        small_offset_msg = _make_ros_vector3_stamped(1.0, 1.0, 1.0)

        node = HandleOffsetCorrection(
            topic_name=topic_name,
            root_link=root_link,
            tip_link=tip_link,
            door_move_connection=door_connection,
            goal_vector=Vector3(x=1.0, y=0.0, z=0.0, reference_frame=root_link),
            threshold=threshold,
        )

        msc = self._build_msc_with_node(
            node, trigger_msgs=[small_offset_msg], topic_name=topic_name
        )
        msc_copy = _msc_from_json(json.loads(json.dumps(msc.to_json())), world)

        executor = Ros2Executor(
            context=_make_context_with_world(world), ros_node=rclpy_node
        )
        executor.compile(motion_statechart=msc_copy)

        handle_node = msc_copy.nodes[0].nodes[0]
        executor.tick_until_end(timeout=5_000)

        history = msc_copy.history.get_observation_history_of_node(handle_node)
        assert history[-1] == ObservationStateValues.TRUE

    def test_transitions_false_then_true(self, rclpy_node):
        """Node should first report FALSE (large offset) then TRUE (small offset)."""
        topic_name = "/robokudo/handle_offset"
        world, root_link, tip_link, door_connection = _make_test_world()

        threshold = 50.0
        large_offset_msg = _make_ros_vector3_stamped(100.0, 100.0, 100.0)
        small_offset_msg = _make_ros_vector3_stamped(1.0, 1.0, 1.0)

        node = HandleOffsetCorrection(
            topic_name=topic_name,
            root_link=root_link,
            tip_link=tip_link,
            door_move_connection=door_connection,
            goal_vector=Vector3(x=1.0, y=0.0, z=0.0, reference_frame=root_link),
            threshold=threshold,
        )

        msc = self._build_msc_with_node(
            node,
            trigger_msgs=[large_offset_msg, small_offset_msg],
            topic_name=topic_name,
        )
        msc_copy = _msc_from_json(json.loads(json.dumps(msc.to_json())), world)

        executor = Ros2Executor(
            context=_make_context_with_world(world), ros_node=rclpy_node
        )
        executor.compile(motion_statechart=msc_copy)

        handle_node = msc_copy.nodes[0].nodes[0]
        executor.tick_until_end(timeout=5_000)

        history = msc_copy.history.get_observation_history_of_node(handle_node)
        assert history[0] == ObservationStateValues.UNKNOWN
        assert ObservationStateValues.FALSE in history
        assert history[-1] == ObservationStateValues.TRUE

    def test_json_serialization_round_trip(self, rclpy_node):
        """Serialising and deserialising the MSC must not break execution."""
        topic_name = "/robokudo/handle_offset"
        world, root_link, tip_link, door_connection = _make_test_world()

        small_offset_msg = _make_ros_vector3_stamped(1.0, 0.0, 0.0)

        node = HandleOffsetCorrection(
            topic_name=topic_name,
            root_link=root_link,
            tip_link=tip_link,
            door_move_connection=door_connection,
            goal_vector=Vector3(x=1.0, y=0.0, z=0.0, reference_frame=root_link),
            threshold=50.0,
        )

        msc = self._build_msc_with_node(
            node, trigger_msgs=[small_offset_msg], topic_name=topic_name
        )

        msc_copy = _msc_from_json(json.loads(json.dumps(msc.to_json())), world)

        executor = Ros2Executor(
            context=_make_context_with_world(world), ros_node=rclpy_node
        )
        executor.compile(motion_statechart=msc_copy)

        handle_node = msc_copy.nodes[0].nodes[0]
        executor.tick_until_end(timeout=5_000)

        history = msc_copy.history.get_observation_history_of_node(handle_node)
        assert len(history) > 0
        assert history[-1] == ObservationStateValues.TRUE

    def test_door_connection_origin_is_updated(self, rclpy_node):
        """After receiving a non-zero offset, the door connection's origin should change."""
        topic_name = "/robokudo/handle_offset"
        world, root_link, tip_link, door_connection = _make_test_world()

        initial_origin = np.copy(door_connection.origin.to_np())

        offset_msg = _make_ros_vector3_stamped(100.0, 0.0, 0.0)
        small_close_msg = _make_ros_vector3_stamped(1.0, 0.0, 0.0)

        node = HandleOffsetCorrection(
            topic_name=topic_name,
            root_link=root_link,
            tip_link=tip_link,
            door_move_connection=door_connection,
            goal_vector=Vector3(x=1.0, y=0.0, z=0.0, reference_frame=root_link),
            threshold=50.0,
        )

        msc = self._build_msc_with_node(
            node,
            trigger_msgs=[offset_msg, small_close_msg],
            topic_name=topic_name,
        )
        msc_copy = _msc_from_json(json.loads(json.dumps(msc.to_json())), world)

        executor = Ros2Executor(
            context=_make_context_with_world(world), ros_node=rclpy_node
        )
        executor.compile(motion_statechart=msc_copy)
        executor.tick_until_end(timeout=5_000)

        updated_origin = door_connection.origin.to_np()
        assert not np.allclose(
            initial_origin, updated_origin
        ), "Expected door_move_connection.origin to be modified after offset correction."

    def test_default_topic_name(self, rclpy_node):
        """HandleOffsetCorrection should subscribe to /robokudo/handle_offset by default."""
        world, root_link, tip_link, door_connection = _make_test_world()

        node = HandleOffsetCorrection(
            root_link=root_link,
            tip_link=tip_link,
            door_move_connection=door_connection,
            goal_vector=Vector3(x=0.0, y=0.0, z=0.0, reference_frame=root_link),
        )

        assert node.topic_name == "/robokudo/handle_offset"

    def test_default_parameters(self, rclpy_node):
        """Verify default threshold and error_adjustment match docstring spec."""
        world, root_link, tip_link, door_connection = _make_test_world()

        node = HandleOffsetCorrection(
            root_link=root_link,
            tip_link=tip_link,
            door_move_connection=door_connection,
            goal_vector=Vector3(x=0.0, y=0.0, z=0.0, reference_frame=root_link),
        )

        assert node.threshold == 50.0
        assert node.error_adjustment == 20.0
