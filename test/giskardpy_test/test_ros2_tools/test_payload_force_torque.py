from __future__ import annotations

import json

import pytest
from geometry_msgs.msg import WrenchStamped

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
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
    Body,
)

from giskardpy.motion_statechart.ros2_nodes.payload_force_torque import (
    PayloadForceTorque,
    ForceTorqueThresholds,
    ObjectTypes,
    GraspThresholdStrategy,
    PlaceThresholdStrategy,
    DoorThresholdStrategy,
    ThresholdStrategyFactory,
)

_original_ros_from_json = Ros2MessageJSONSerializer.from_json.__func__


@classmethod
def _patched_ros_from_json(cls, data, clazz, **kwargs):
    return _original_ros_from_json(cls, data, clazz)


Ros2MessageJSONSerializer.from_json = _patched_ros_from_json


def _make_wrench_stamped(
    fx: float = 0.0,
    fy: float = 0.0,
    fz: float = 0.0,
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.0,
    frame_id: str = "base_footprint",
) -> WrenchStamped:
    """Helper to build a WrenchStamped message."""
    msg = WrenchStamped()
    msg.header.frame_id = frame_id
    msg.wrench.force.x = fx
    msg.wrench.force.y = fy
    msg.wrench.force.z = fz
    msg.wrench.torque.x = tx
    msg.wrench.torque.y = ty
    msg.wrench.torque.z = tz
    return msg


def _make_test_world() -> (
    tuple[World, KinematicStructureEntity, KinematicStructureEntity]
):
    """
    Build a minimal World with:
      - a 'map' root body
      - a 'base_footprint' body connected via FixedConnection
      - a 'hand_gripper_tool_frame' body connected via FixedConnection

    Returns (world, base_footprint, hand_gripper_tool_frame).
    """
    world = World()

    map_body = Body(name=PrefixedName("map"))
    base_footprint = Body(name=PrefixedName("base_footprint"))
    hand_gripper = Body(name=PrefixedName("hand_gripper_tool_frame"))

    with world.modify_world():
        world.add_kinematic_structure_entity(map_body)

        base_connection = FixedConnection(
            parent=map_body,
            child=base_footprint,
            parent_T_connection_expression=HomogeneousTransformationMatrix(
                reference_frame=map_body, child_frame=base_footprint
            ),
        )
        world.add_connection(base_connection)

        gripper_connection = FixedConnection(
            parent=map_body,
            child=hand_gripper,
            parent_T_connection_expression=HomogeneousTransformationMatrix(
                reference_frame=map_body, child_frame=hand_gripper
            ),
        )
        world.add_connection(gripper_connection)

    return world, base_footprint, hand_gripper


def _make_context_with_world(world: World) -> MotionStatechartContext:
    return MotionStatechartContext(world=world)


def _msc_from_json(json_data: dict, world: World) -> MotionStatechart:
    tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    return MotionStatechart.from_json(json_data, **tracker.create_kwargs())


class TestPayloadForceTorque:
    """Tests for the PayloadForceTorque motion statechart node."""

    def _build_msc_with_node(
        self,
        node: PayloadForceTorque,
        trigger_msgs: list[WrenchStamped],
        topic_name: str = "/ft_sensor/wrench",
    ) -> MotionStatechart:
        publish_nodes = [
            PublishOnStart(topic_name=topic_name, msg=msg) for msg in trigger_msgs
        ]
        wait_node = WaitForMessage(topic_name=topic_name, msg_type=WrenchStamped)

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
        topic_name = "/ft_sensor/wrench"
        world, base_footprint, _ = _make_test_world()

        node = PayloadForceTorque(
            topic_name=topic_name,
            threshold_enum=ForceTorqueThresholds.DOOR.value,
        )

        # Send a message well above threshold so the MSC terminates
        high_force_msg = _make_wrench_stamped(fz=200.0)
        msc = self._build_msc_with_node(
            node, trigger_msgs=[high_force_msg], topic_name=topic_name
        )
        msc_copy = _msc_from_json(json.loads(json.dumps(msc.to_json())), world)

        executor = Ros2Executor(
            context=_make_context_with_world(world), ros_node=rclpy_node
        )
        executor.compile(motion_statechart=msc_copy)

        ft_node = msc_copy.nodes[0].nodes[0]
        executor.tick_until_end(timeout=5_000)

        history = msc_copy.history.get_observation_history_of_node(ft_node)
        assert history[0] == ObservationStateValues.UNKNOWN

    def test_false_when_force_below_door_threshold(self, rclpy_node):
        """Node should return FALSE when force_z is below the DOOR threshold (80)."""
        topic_name = "/ft_sensor/wrench"
        world, _, _ = _make_test_world()

        node = PayloadForceTorque(
            topic_name=topic_name,
            threshold_enum=ForceTorqueThresholds.DOOR.value,
        )

        low_force_msg = _make_wrench_stamped(
            fz=10.0, frame_id="hand_gripper_tool_frame"
        )
        high_force_msg = _make_wrench_stamped(
            fz=200.0, frame_id="hand_gripper_tool_frame"
        )
        msc = self._build_msc_with_node(
            node, trigger_msgs=[low_force_msg, high_force_msg], topic_name=topic_name
        )
        msc_copy = _msc_from_json(json.loads(json.dumps(msc.to_json())), world)

        executor = Ros2Executor(
            context=_make_context_with_world(world), ros_node=rclpy_node
        )
        executor.compile(motion_statechart=msc_copy)

        ft_node = msc_copy.nodes[0].nodes[0]
        executor.tick_until_end(timeout=5_000)

        history = msc_copy.history.get_observation_history_of_node(ft_node)
        assert ObservationStateValues.FALSE in history

    def test_true_when_force_exceeds_door_threshold(self, rclpy_node):
        """Node should return TRUE when force_z exceeds the DOOR threshold (80)."""
        topic_name = "/ft_sensor/wrench"
        world, _, _ = _make_test_world()

        node = PayloadForceTorque(
            topic_name=topic_name,
            threshold_enum=ForceTorqueThresholds.DOOR.value,
        )

        high_force_msg = _make_wrench_stamped(
            fz=200.0, frame_id="hand_gripper_tool_frame"
        )
        msc = self._build_msc_with_node(
            node, trigger_msgs=[high_force_msg], topic_name=topic_name
        )
        msc_copy = _msc_from_json(json.loads(json.dumps(msc.to_json())), world)

        executor = Ros2Executor(
            context=_make_context_with_world(world), ros_node=rclpy_node
        )
        executor.compile(motion_statechart=msc_copy)

        ft_node = msc_copy.nodes[0].nodes[0]
        executor.tick_until_end(timeout=5_000)

        history = msc_copy.history.get_observation_history_of_node(ft_node)
        assert history[-1] == ObservationStateValues.TRUE

    def test_transitions_false_then_true_door(self, rclpy_node):
        """Node should first report FALSE (low force) then TRUE (high force)."""
        topic_name = "/ft_sensor/wrench"
        world, _, _ = _make_test_world()

        node = PayloadForceTorque(
            topic_name=topic_name,
            threshold_enum=ForceTorqueThresholds.DOOR.value,
        )

        low_force_msg = _make_wrench_stamped(
            fz=10.0, frame_id="hand_gripper_tool_frame"
        )
        high_force_msg = _make_wrench_stamped(
            fz=200.0, frame_id="hand_gripper_tool_frame"
        )
        msc = self._build_msc_with_node(
            node, trigger_msgs=[low_force_msg, high_force_msg], topic_name=topic_name
        )
        msc_copy = _msc_from_json(json.loads(json.dumps(msc.to_json())), world)

        executor = Ros2Executor(
            context=_make_context_with_world(world), ros_node=rclpy_node
        )
        executor.compile(motion_statechart=msc_copy)

        ft_node = msc_copy.nodes[0].nodes[0]
        executor.tick_until_end(timeout=5_000)

        history = msc_copy.history.get_observation_history_of_node(ft_node)
        assert history[0] == ObservationStateValues.UNKNOWN
        assert ObservationStateValues.FALSE in history
        assert history[-1] == ObservationStateValues.TRUE

    def test_stay_true_latches_true_state(self, rclpy_node):
        """With stay_true=True, once TRUE the node must remain TRUE on subsequent ticks."""
        topic_name = "/ft_sensor/wrench"
        world, _, _ = _make_test_world()

        node = PayloadForceTorque(
            topic_name=topic_name,
            threshold_enum=ForceTorqueThresholds.DOOR.value,
            stay_true=True,
        )

        high_force_msg = _make_wrench_stamped(
            fz=200.0, frame_id="hand_gripper_tool_frame"
        )
        low_force_msg = _make_wrench_stamped(fz=1.0, frame_id="hand_gripper_tool_frame")
        msc = self._build_msc_with_node(
            node, trigger_msgs=[high_force_msg, low_force_msg], topic_name=topic_name
        )
        msc_copy = _msc_from_json(json.loads(json.dumps(msc.to_json())), world)

        executor = Ros2Executor(
            context=_make_context_with_world(world), ros_node=rclpy_node
        )
        executor.compile(motion_statechart=msc_copy)

        ft_node = msc_copy.nodes[0].nodes[0]
        executor.tick_until_end(timeout=5_000)

        history = msc_copy.history.get_observation_history_of_node(ft_node)
        # Once TRUE appears, all subsequent states must also be TRUE
        true_index = next(
            i for i, v in enumerate(history) if v == ObservationStateValues.TRUE
        )
        assert all(v == ObservationStateValues.TRUE for v in history[true_index:])

    def test_true_when_torque_y_exceeds_grasp_default_threshold(self, rclpy_node):
        """GRASP strategy for OT_Default triggers on |torque_y| > 2."""
        topic_name = "/ft_sensor/wrench"
        world, _, _ = _make_test_world()

        node = PayloadForceTorque(
            topic_name=topic_name,
            threshold_enum=ForceTorqueThresholds.GRASP.value,
            object_type=ObjectTypes.OT_Default.value,
        )

        high_torque_msg = _make_wrench_stamped(ty=5.0)
        msc = self._build_msc_with_node(
            node, trigger_msgs=[high_torque_msg], topic_name=topic_name
        )
        msc_copy = _msc_from_json(json.loads(json.dumps(msc.to_json())), world)

        executor = Ros2Executor(
            context=_make_context_with_world(world), ros_node=rclpy_node
        )
        executor.compile(motion_statechart=msc_copy)

        ft_node = msc_copy.nodes[0].nodes[0]
        executor.tick_until_end(timeout=5_000)

        history = msc_copy.history.get_observation_history_of_node(ft_node)
        assert history[-1] == ObservationStateValues.TRUE

    def test_true_when_force_z_exceeds_place_bowl_threshold(self, rclpy_node):
        """PLACE strategy for OT_Bowl triggers on |force_z| >= 35."""
        topic_name = "/ft_sensor/wrench"
        world, _, _ = _make_test_world()

        node = PayloadForceTorque(
            topic_name=topic_name,
            threshold_enum=ForceTorqueThresholds.PLACE.value,
            object_type=ObjectTypes.OT_Bowl.value,
        )

        high_force_msg = _make_wrench_stamped(fz=50.0)
        msc = self._build_msc_with_node(
            node, trigger_msgs=[high_force_msg], topic_name=topic_name
        )
        msc_copy = _msc_from_json(json.loads(json.dumps(msc.to_json())), world)

        executor = Ros2Executor(
            context=_make_context_with_world(world), ros_node=rclpy_node
        )
        executor.compile(motion_statechart=msc_copy)

        ft_node = msc_copy.nodes[0].nodes[0]
        executor.tick_until_end(timeout=5_000)

        history = msc_copy.history.get_observation_history_of_node(ft_node)
        assert history[-1] == ObservationStateValues.TRUE

    def test_json_serialization_round_trip(self, rclpy_node):
        """Serialising and deserialising the MSC must not break execution."""
        topic_name = "/ft_sensor/wrench"
        world, _, _ = _make_test_world()

        node = PayloadForceTorque(
            topic_name=topic_name,
            threshold_enum=ForceTorqueThresholds.DOOR.value,
        )

        high_force_msg = _make_wrench_stamped(
            fz=200.0, frame_id="hand_gripper_tool_frame"
        )
        msc = self._build_msc_with_node(
            node, trigger_msgs=[high_force_msg], topic_name=topic_name
        )

        msc_copy = _msc_from_json(json.loads(json.dumps(msc.to_json())), world)

        executor = Ros2Executor(
            context=_make_context_with_world(world), ros_node=rclpy_node
        )
        executor.compile(motion_statechart=msc_copy)

        ft_node = msc_copy.nodes[0].nodes[0]
        executor.tick_until_end(timeout=5_000)

        history = msc_copy.history.get_observation_history_of_node(ft_node)
        assert len(history) > 0
        assert history[-1] == ObservationStateValues.TRUE

    def test_factory_returns_correct_strategy_types(self, rclpy_node):
        """ThresholdStrategyFactory must return the correct strategy class per enum."""
        grasp_strategy = ThresholdStrategyFactory.get_strategy(
            ObjectTypes.OT_Default.value, ForceTorqueThresholds.GRASP.value
        )
        place_strategy = ThresholdStrategyFactory.get_strategy(
            ObjectTypes.OT_Default.value, ForceTorqueThresholds.PLACE.value
        )
        door_strategy = ThresholdStrategyFactory.get_strategy(
            None, ForceTorqueThresholds.DOOR.value
        )

        assert isinstance(grasp_strategy, GraspThresholdStrategy)
        assert isinstance(place_strategy, PlaceThresholdStrategy)
        assert isinstance(door_strategy, DoorThresholdStrategy)

    def test_factory_raises_on_invalid_enum(self, rclpy_node):
        """ThresholdStrategyFactory must raise ValueError for an unknown enum value."""
        with pytest.raises(ValueError, match="Invalid threshold_enum"):
            ThresholdStrategyFactory.get_strategy(None, threshold_enum=999)

    def test_grasp_strategy_raises_on_unknown_object_type(self, rclpy_node):
        """GraspThresholdStrategy must raise ValueError for an unrecognised object type."""
        strategy = GraspThresholdStrategy("UnknownObject")

        class _FakeVec:
            def __getitem__(self, i):
                return 0.0

        with pytest.raises(ValueError, match="No valid object_type found"):
            strategy.check_thresholds(_FakeVec(), _FakeVec())

    def test_true_when_force_z_exceeds_shelf_grasp_threshold(self, rclpy_node):
        """SHELF_GRASP strategy triggers on |force_z| >= 30."""
        topic_name = "/ft_sensor/wrench"
        world, _, _ = _make_test_world()

        node = PayloadForceTorque(
            topic_name=topic_name,
            threshold_enum=ForceTorqueThresholds.SHELF_GRASP.value,
        )

        high_force_msg = _make_wrench_stamped(
            fz=50.0, frame_id="hand_gripper_tool_frame"
        )
        msc = self._build_msc_with_node(
            node, trigger_msgs=[high_force_msg], topic_name=topic_name
        )
        msc_copy = _msc_from_json(json.loads(json.dumps(msc.to_json())), world)

        executor = Ros2Executor(
            context=_make_context_with_world(world), ros_node=rclpy_node
        )
        executor.compile(motion_statechart=msc_copy)

        ft_node = msc_copy.nodes[0].nodes[0]
        executor.tick_until_end(timeout=5_000)

        history = msc_copy.history.get_observation_history_of_node(ft_node)
        assert history[-1] == ObservationStateValues.TRUE

    def test_default_stay_true_is_true(self, rclpy_node):
        """PayloadForceTorque should default stay_true to True."""
        world, _, _ = _make_test_world()

        node = PayloadForceTorque(
            topic_name="/ft_sensor/wrench",
            threshold_enum=ForceTorqueThresholds.DOOR.value,
        )

        assert node.stay_true is True
