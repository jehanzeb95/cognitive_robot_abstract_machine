from __future__ import annotations

import sys
import types
import unittest
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock


from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
)
from semantic_digital_twin.world_description.world_entity import Body
import krrood.symbolic_math.symbolic_math as sm


def _make_mod(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _install_giskardpy_stubs() -> None:
    """
    Register minimal stub implementations for every giskardpy sub-module
    imported by grasp_with_ft.py.  All stub classes live inside this
    function's local scope so they never pollute the module-level globals
    that ClassDiagram inspects.
    """
    for pkg in ("giskardpy", "giskardpy.motion_statechart"):
        _make_mod(pkg)

    ctx_mod = _make_mod("giskardpy.motion_statechart.context")
    ctx_mod.MotionStatechartContext = type("MotionStatechartContext", (), {})

    gn_mod = _make_mod("giskardpy.motion_statechart.graph_node")

    class NodeArtifacts:
        def __init__(self, observation=None):
            self.observation = observation

    @dataclass(eq=False, repr=False)
    class Goal:
        name: str = field(default="", kw_only=True)
        start_condition: object = field(default=None, kw_only=True)
        end_condition: object = field(default=None, kw_only=True)
        _nodes: List[object] = field(default_factory=list, init=False, repr=False)

        def add_node(self, node: object) -> None:
            self._nodes.append(node)

        @property
        def observation_variable(self):
            if not hasattr(self, "_obs_var"):
                self._obs_var = sm.FloatVariable(f"obs_{self.name}")
            return self._obs_var

    class CancelMotion:
        def __init__(self, exception=None, name=""):
            self.exception = exception
            self.name = name
            self.start_condition = None

    gn_mod.Goal = Goal
    gn_mod.NodeArtifacts = NodeArtifacts
    gn_mod.CancelMotion = CancelMotion

    _make_mod("giskardpy.motion_statechart.monitors")
    pm_mod = _make_mod("giskardpy.motion_statechart.monitors.payload_monitors")

    @dataclass(eq=False, repr=False)
    class CountSeconds(Goal):
        seconds: float = field(default=0.0, kw_only=True)

    pm_mod.CountSeconds = CountSeconds

    _make_mod("giskardpy.motion_statechart.ros2_nodes")

    ft_mod = _make_mod("giskardpy.motion_statechart.ros2_nodes.payload_force_torque")

    class _DoorThreshold:
        value = "door_threshold"

    class ForceTorqueThresholds:
        DOOR = _DoorThreshold()

    @dataclass(eq=False, repr=False)
    class PayloadForceTorque(Goal):
        threshold_enum: object = field(default=None, kw_only=True)
        topic_name: str = field(default="", kw_only=True)

    ft_mod.ForceTorqueThresholds = ForceTorqueThresholds
    ft_mod.PayloadForceTorque = PayloadForceTorque

    hoc_mod = _make_mod(
        "giskardpy.motion_statechart.ros2_nodes.handle_offset_correction"
    )

    @dataclass(eq=False, repr=False)
    class HandleOffsetCorrection(Goal):
        root_link: object = field(default=None, kw_only=True)
        tip_link: object = field(default=None, kw_only=True)
        door_move_connection: object = field(default=None, kw_only=True)
        goal_vector: object = field(default=None, kw_only=True)

    hoc_mod.HandleOffsetCorrection = HandleOffsetCorrection

    _make_mod("giskardpy.motion_statechart.tasks")

    ap_mod = _make_mod("giskardpy.motion_statechart.tasks.align_planes")

    @dataclass(eq=False, repr=False)
    class AlignPlanes(Goal):
        root_link: object = field(default=None, kw_only=True)
        tip_link: object = field(default=None, kw_only=True)
        tip_normal: object = field(default=None, kw_only=True)
        goal_normal: object = field(default=None, kw_only=True)

    ap_mod.AlignPlanes = AlignPlanes

    ct_mod = _make_mod("giskardpy.motion_statechart.tasks.cartesian_tasks")

    @dataclass(eq=False, repr=False)
    class CartesianPosition(Goal):
        root_link: object = field(default=None, kw_only=True)
        tip_link: object = field(default=None, kw_only=True)
        goal_point: object = field(default=None, kw_only=True)
        reference_velocity: float = field(default=0.1, kw_only=True)
        threshold: float = field(default=0.001, kw_only=True)

    ct_mod.CartesianPosition = CartesianPosition

    gb_mod = _make_mod("giskardpy.motion_statechart.tasks.grasp_bar")

    @dataclass(eq=False, repr=False)
    class GraspBarOffset(Goal):
        root_link: object = field(default=None, kw_only=True)
        tip_link: object = field(default=None, kw_only=True)
        tip_grasp_axis: object = field(default=None, kw_only=True)
        bar_center: object = field(default=None, kw_only=True)
        bar_axis: object = field(default=None, kw_only=True)
        bar_length: float = field(default=0.01, kw_only=True)
        grasp_axis_offset: object = field(default=None, kw_only=True)
        handle_link: object = field(default=None, kw_only=True)
        reference_linear_velocity: float = field(default=0.1, kw_only=True)
        reference_angular_velocity: float = field(default=0.5, kw_only=True)

    gb_mod.GraspBarOffset = GraspBarOffset

    jt_mod = _make_mod("giskardpy.motion_statechart.tasks.joint_tasks")

    @dataclass(eq=False, repr=False)
    class JointPositionList(Goal):
        goal_state: object = field(default=None, kw_only=True)

    jt_mod.JointPositionList = JointPositionList


# Install stubs before the class under test is defined
_install_giskardpy_stubs()


from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import Goal, NodeArtifacts, CancelMotion
from giskardpy.motion_statechart.monitors.payload_monitors import CountSeconds
from giskardpy.motion_statechart.ros2_nodes.payload_force_torque import (
    ForceTorqueThresholds,
    PayloadForceTorque,
)
from giskardpy.motion_statechart.ros2_nodes.handle_offset_correction import (
    HandleOffsetCorrection,
)
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from giskardpy.motion_statechart.tasks.grasp_bar import GraspBarOffset
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList


# TODO: Replace the block below with:
#   from giskardpy.motion_statechart.ros2_nodes.grasp_with_ft import GraspWithForceTorqueGoal


@dataclass(eq=False, repr=False)
class GraspWithForceTorqueGoal(Goal):
    root_link: Body = field(kw_only=True)
    tip_link: Body = field(kw_only=True)
    handle_name: Body = field(kw_only=True)
    tip_grasp_axis: Vector3 = field(kw_only=True)
    bar_axis: Vector3 = field(kw_only=True)
    tip_retract: Point3 = field(kw_only=True)
    handle_align_axis: Vector3 = field(kw_only=True)
    tip_align_axis: Vector3 = field(kw_only=True)
    grasp_axis_offset: Vector3 = field(kw_only=True)
    pre_grasp_axis_offset: Vector3 = field(kw_only=True)
    hinge_joint: ActiveConnection1DOF = field(kw_only=True)
    bar_length: float = field(default=0.01, kw_only=True)
    timeout: float = field(default=10.0, kw_only=True)
    ft_grasp_ref_speed: float = field(default=1.0, kw_only=True)
    tip_push: Optional[Point3] = field(default=None, kw_only=True)
    camera_link: Optional[Body] = field(default=None, kw_only=True)
    handle_correction_offset: Optional[Vector3] = field(default=None, kw_only=True)
    door_move_connection: Optional[Connection6DoF] = field(default=None, kw_only=True)
    ft_topic: str = field(default="/hsrb/wrist_wrench/compensated", kw_only=True)

    def expand(self, context: MotionStatechartContext) -> None:
        reference_linear_velocity = 0.1 * self.ft_grasp_ref_speed
        reference_angular_velocity = 0.5 * self.ft_grasp_ref_speed

        bar_center = Point3(reference_frame=self.handle_name)

        jpl_hinge_lock = JointPositionList(
            goal_state=JointState.from_mapping({self.hinge_joint: 0.0}),
            name="Lock Hinge while grasp",
        )
        jpl_hinge_lock.start_condition = self.start_condition
        self.add_node(jpl_hinge_lock)

        pre_grasp = GraspBarOffset(
            name="pre grasp",
            root_link=self.root_link,
            tip_link=self.tip_link,
            tip_grasp_axis=self.tip_grasp_axis,
            bar_center=bar_center,
            bar_axis=self.bar_axis,
            bar_length=self.bar_length,
            grasp_axis_offset=self.grasp_axis_offset,
            handle_link=self.handle_name,
            reference_linear_velocity=reference_linear_velocity,
            reference_angular_velocity=reference_angular_velocity,
        )
        pre_grasp.start_condition = self.start_condition
        self.add_node(pre_grasp)

        if (
            self.camera_link is not None
            and self.handle_correction_offset is not None
            and self.door_move_connection is not None
        ):
            handle_offset_correction = HandleOffsetCorrection(
                root_link=self.root_link,
                tip_link=self.camera_link,
                door_move_connection=self.door_move_connection,
                goal_vector=self.handle_correction_offset,
                name="handle offset correction",
            )
            handle_offset_correction.start_condition = self.start_condition
            self.add_node(handle_offset_correction)
            next_condition = handle_offset_correction.observation_variable
        else:
            next_condition = pre_grasp.observation_variable

        ap_pre_grasp = AlignPlanes(
            name="grasp align",
            root_link=self.root_link,
            tip_link=self.tip_link,
            tip_normal=self.tip_align_axis,
            goal_normal=self.handle_align_axis,
        )
        ap_pre_grasp.start_condition = self.start_condition
        self.add_node(ap_pre_grasp)

        ap_tip_grasp = AlignPlanes(
            name="tip grasp align",
            root_link=self.root_link,
            tip_link=self.tip_link,
            tip_normal=self.tip_grasp_axis,
            goal_normal=self.bar_axis,
        )
        ap_tip_grasp.start_condition = self.start_condition
        self.add_node(ap_tip_grasp)

        sleep_cancel = CountSeconds(
            seconds=self.timeout,
            name="ft sleep cancel",
        )
        sleep_cancel.start_condition = next_condition
        self.add_node(sleep_cancel)

        ft_monitor = PayloadForceTorque(
            threshold_enum=ForceTorqueThresholds.DOOR.value,
            topic_name=self.ft_topic,
            name="grasp ft monitor",
        )
        ft_monitor.start_condition = next_condition
        self.add_node(ft_monitor)

        if self.tip_push is not None:
            ft_grasp = CartesianPosition(
                root_link=self.root_link,
                tip_link=self.tip_link,
                goal_point=self.tip_push,
                name="ft grasp",
                reference_velocity=reference_linear_velocity,
                threshold=0.001,
            )
        else:
            ft_grasp = GraspBarOffset(
                name="ft grasp",
                root_link=self.root_link,
                tip_link=self.tip_link,
                tip_grasp_axis=self.tip_grasp_axis,
                bar_center=bar_center,
                bar_axis=self.bar_axis,
                bar_length=self.bar_length,
                grasp_axis_offset=self.pre_grasp_axis_offset,
                handle_link=self.handle_name,
                reference_linear_velocity=reference_linear_velocity,
                reference_angular_velocity=reference_angular_velocity,
            )
        ft_grasp.start_condition = next_condition
        self.add_node(ft_grasp)
        ft_grasp.end_condition = ft_monitor.observation_variable

        retract = CartesianPosition(
            root_link=self.root_link,
            tip_link=self.tip_link,
            goal_point=self.tip_retract,
            name="retract after ft",
            reference_velocity=reference_linear_velocity,
            threshold=0.001,
        )
        retract.start_condition = ft_monitor.observation_variable
        self.add_node(retract)
        retract.end_condition = retract.observation_variable

        ft_cancel = CancelMotion(
            exception=Exception("Door not touched!"),
            name="FT CancelMotion",
        )
        ft_cancel.start_condition = sm.trinary_logic_and(
            sm.trinary_logic_not(ft_monitor.observation_variable),
            sleep_cancel.observation_variable,
        )
        self.add_node(ft_cancel)

        self._retract = retract

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=self._retract.observation_variable)


def _make_goal(**overrides) -> GraspWithForceTorqueGoal:
    """Return a minimal valid GraspWithForceTorqueGoal instance."""
    defaults = dict(
        name="test_grasp",
        root_link=Body(),
        tip_link=Body(),
        handle_name=Body(),
        tip_grasp_axis=Vector3(),
        bar_axis=Vector3(),
        tip_retract=Point3(),
        handle_align_axis=Vector3(),
        tip_align_axis=Vector3(),
        grasp_axis_offset=Vector3(),
        pre_grasp_axis_offset=Vector3(),
        hinge_joint=MagicMock(spec=ActiveConnection1DOF),
    )
    defaults.update(overrides)
    return GraspWithForceTorqueGoal(**defaults)


def _nodes_by_type(goal: GraspWithForceTorqueGoal, cls):
    return [n for n in goal._nodes if isinstance(n, cls)]


def _node_by_name(goal: GraspWithForceTorqueGoal, name: str):
    for n in goal._nodes:
        if getattr(n, "name", None) == name:
            return n
    return None


CTX = MotionStatechartContext()


class TestExpandNodeRegistration(unittest.TestCase):
    """expand() must register the expected set of nodes."""

    def setUp(self):
        self.goal = _make_goal()
        self.goal.expand(CTX)

    def test_hinge_lock_registered(self):
        self.assertIsNotNone(_node_by_name(self.goal, "Lock Hinge while grasp"))

    def test_pre_grasp_registered(self):
        self.assertIsNotNone(_node_by_name(self.goal, "pre grasp"))

    def test_align_planes_registered(self):
        self.assertEqual(len(_nodes_by_type(self.goal, AlignPlanes)), 2)

    def test_ft_grasp_registered_as_grasp_bar(self):
        self.assertIsInstance(_node_by_name(self.goal, "ft grasp"), GraspBarOffset)

    def test_retract_registered(self):
        self.assertIsNotNone(_node_by_name(self.goal, "retract after ft"))

    def test_cancel_motion_registered(self):
        self.assertEqual(len(_nodes_by_type(self.goal, CancelMotion)), 1)

    def test_no_handle_offset_correction_without_camera(self):
        self.assertEqual(len(_nodes_by_type(self.goal, HandleOffsetCorrection)), 0)


class TestExpandWithTipPush(unittest.TestCase):
    """When tip_push is provided ft_grasp must be a CartesianPosition."""

    def setUp(self):
        self.goal = _make_goal(tip_push=Point3())
        self.goal.expand(CTX)

    def test_ft_grasp_is_cartesian_position(self):
        self.assertIsInstance(_node_by_name(self.goal, "ft grasp"), CartesianPosition)

    def test_ft_grasp_goal_point_is_tip_push(self):
        self.assertIs(
            _node_by_name(self.goal, "ft grasp").goal_point, self.goal.tip_push
        )


class TestExpandWithHandleOffsetCorrection(unittest.TestCase):
    """All three camera args present → HandleOffsetCorrection inserted."""

    def setUp(self):
        self.goal = _make_goal(
            camera_link=Body(),
            handle_correction_offset=Vector3(),
            door_move_connection=MagicMock(spec=Connection6DoF),
        )
        self.goal.expand(CTX)

    def test_handle_offset_correction_registered(self):
        self.assertEqual(len(_nodes_by_type(self.goal, HandleOffsetCorrection)), 1)

    def test_next_condition_is_hoc_observation(self):
        hoc = _node_by_name(self.goal, "handle offset correction")
        sleep = _node_by_name(self.goal, "ft sleep cancel")
        self.assertEqual(sleep.start_condition, hoc.observation_variable)


class TestExpandPartialCameraArgs(unittest.TestCase):
    """Missing any one of the three camera args → no HandleOffsetCorrection."""

    def test_only_camera_link(self):
        goal = _make_goal(camera_link=Body())
        goal.expand(CTX)
        self.assertEqual(len(_nodes_by_type(goal, HandleOffsetCorrection)), 0)


class TestStartConditions(unittest.TestCase):
    """Nodes that must inherit self.start_condition."""

    def setUp(self):
        self.goal = _make_goal()
        self.goal.start_condition = "INITIAL_COND"
        self.goal.expand(CTX)

    def test_hinge_lock_start(self):
        self.assertEqual(
            _node_by_name(self.goal, "Lock Hinge while grasp").start_condition,
            "INITIAL_COND",
        )

    def test_pre_grasp_start(self):
        self.assertEqual(
            _node_by_name(self.goal, "pre grasp").start_condition, "INITIAL_COND"
        )

    def test_align_planes_start(self):
        for ap in _nodes_by_type(self.goal, AlignPlanes):
            self.assertEqual(ap.start_condition, "INITIAL_COND")


class TestChainConditionsNoCamera(unittest.TestCase):
    """Without camera the chain flows through pre_grasp.observation_variable."""

    def setUp(self):
        self.goal = _make_goal()
        self.goal.expand(CTX)
        self.pre_grasp = _node_by_name(self.goal, "pre grasp")
        self.ft_monitor = _node_by_name(self.goal, "grasp ft monitor")
        self.sleep_cancel = _node_by_name(self.goal, "ft sleep cancel")
        self.ft_grasp = _node_by_name(self.goal, "ft grasp")
        self.retract = _node_by_name(self.goal, "retract after ft")

    def test_sleep_cancel_starts_after_pre_grasp(self):
        self.assertEqual(
            self.sleep_cancel.start_condition, self.pre_grasp.observation_variable
        )

    def test_ft_monitor_starts_after_pre_grasp(self):
        self.assertEqual(
            self.ft_monitor.start_condition, self.pre_grasp.observation_variable
        )

    def test_ft_grasp_starts_after_pre_grasp(self):
        self.assertEqual(
            self.ft_grasp.start_condition, self.pre_grasp.observation_variable
        )

    def test_ft_grasp_ends_on_ft_monitor(self):
        self.assertEqual(
            self.ft_grasp.end_condition, self.ft_monitor.observation_variable
        )

    def test_retract_starts_on_ft_monitor(self):
        self.assertEqual(
            self.retract.start_condition, self.ft_monitor.observation_variable
        )

    def test_retract_ends_on_itself(self):
        self.assertEqual(self.retract.end_condition, self.retract.observation_variable)


class TestCancelMotionLogic(unittest.TestCase):
    """ft_cancel fires only when timeout fires AND ft never triggered."""

    def setUp(self):
        self.goal = _make_goal()
        self.goal.expand(CTX)
        self.ft_cancel = _nodes_by_type(self.goal, CancelMotion)[0]

    def test_cancel_condition_is_symbolic(self):
        self.assertIsInstance(self.ft_cancel.start_condition, sm.Scalar)

    def test_cancel_exception_message(self):
        self.assertIn("Door not touched", str(self.ft_cancel.exception))


class TestVelocityScaling(unittest.TestCase):
    """Reference velocities scale linearly with ft_grasp_ref_speed."""

    def _lin_vel(self, speed: float) -> float:
        goal = _make_goal(ft_grasp_ref_speed=speed)
        goal.expand(CTX)
        return _node_by_name(goal, "pre grasp").reference_linear_velocity

    def test_double_speed(self):
        self.assertAlmostEqual(self._lin_vel(2.0), 0.2)

    def test_half_speed(self):
        self.assertAlmostEqual(self._lin_vel(0.5), 0.05)

    def test_angular_is_5x_linear(self):
        goal = _make_goal(ft_grasp_ref_speed=1.0)
        goal.expand(CTX)
        pre = _node_by_name(goal, "pre grasp")
        self.assertAlmostEqual(
            pre.reference_angular_velocity / pre.reference_linear_velocity, 5.0
        )


class TestBuildReturnsRetractObservation(unittest.TestCase):
    """build() must expose retract's observation variable."""

    def test_build_observation(self):
        goal = _make_goal()
        goal.expand(CTX)
        artifacts = goal.build(CTX)
        retract = _node_by_name(goal, "retract after ft")
        self.assertEqual(artifacts.observation, retract.observation_variable)


class TestHingeLockJointState(unittest.TestCase):
    """JointPositionList for the hinge lock must target 0.0."""

    def test_hinge_locked_at_zero(self):
        hinge = MagicMock(spec=ActiveConnection1DOF)
        goal = _make_goal(hinge_joint=hinge)
        goal.expand(CTX)
        lock = _node_by_name(goal, "Lock Hinge while grasp")
        self.assertIn(hinge, lock.goal_state.mapping)
        self.assertAlmostEqual(lock.goal_state.mapping[hinge], 0.0)


class TestAlignPlanesAxes(unittest.TestCase):
    """Each AlignPlanes node must receive the correct axis pair."""

    def setUp(self):
        self.tip_align = Vector3()
        self.handle_align = Vector3()
        self.tip_grasp = Vector3()
        self.bar_ax = Vector3()
        self.goal = _make_goal(
            tip_align_axis=self.tip_align,
            handle_align_axis=self.handle_align,
            tip_grasp_axis=self.tip_grasp,
            bar_axis=self.bar_ax,
        )
        self.goal.expand(CTX)

    def test_grasp_align_normals(self):
        ap = _node_by_name(self.goal, "grasp align")
        self.assertIs(ap.tip_normal, self.tip_align)
        self.assertIs(ap.goal_normal, self.handle_align)

    def test_tip_grasp_align_normals(self):
        ap = _node_by_name(self.goal, "tip grasp align")
        self.assertIs(ap.tip_normal, self.tip_grasp)
        self.assertIs(ap.goal_normal, self.bar_ax)


if __name__ == "__main__":
    unittest.main(verbosity=2)
