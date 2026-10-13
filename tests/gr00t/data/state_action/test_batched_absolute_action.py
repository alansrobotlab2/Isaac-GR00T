# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The batched relative-to-absolute action conversion must match the per-pose one.

``StateActionProcessor._convert_to_absolute_action`` works on whole (T, D) arrays;
``_convert_to_absolute_action_objects`` is the original formulation that builds
``Pose`` objects per timestep. They have to agree for every action format and dtype.
"""

from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.data.types import ActionFormat, ActionType
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


HORIZON = 40


def _processor() -> StateActionProcessor:
    # The conversion helpers do not touch any processor state.
    return StateActionProcessor.__new__(StateActionProcessor)


def _random_eef(rng, n: int, action_format: ActionFormat, dtype) -> np.ndarray:
    rotation = Rotation.random(n, random_state=rng)
    translation = rng.normal(size=(n, 3))
    if action_format == ActionFormat.XYZ_ROT6D:
        # Model outputs are not orthonormal: scale and perturb the two stored rows.
        rot6d = rotation.as_matrix()[:, :2].reshape(n, 6)
        rot6d = rot6d * rng.uniform(0.5, 1.5, size=(n, 1)) + rng.normal(scale=0.05, size=(n, 6))
        return np.concatenate([translation, rot6d], axis=1).astype(dtype)
    if action_format == ActionFormat.XYZ_ROTVEC:
        return np.concatenate([translation, rotation.as_rotvec()], axis=1).astype(dtype)
    homogeneous = np.tile(np.eye(4), (n, 1, 1))
    homogeneous[:, :3, :3] = rotation.as_matrix()
    homogeneous[:, :3, 3] = translation
    return homogeneous.reshape(n, 16).astype(dtype)


@pytest.mark.parametrize("action_format", list(ActionFormat))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_eef_batched_matches_per_pose(action_format, dtype):
    rng = np.random.default_rng(0)
    processor = _processor()
    tolerance = 1e-6 if dtype == np.float32 else 1e-12
    for _ in range(10):
        action = _random_eef(rng, HORIZON, action_format, dtype)
        reference = _random_eef(rng, 1, action_format, dtype)[0]
        batched = processor._convert_to_absolute_action(
            action, reference, ActionType.EEF, action_format
        )
        per_pose = processor._convert_to_absolute_action_objects(
            action, reference, ActionType.EEF, action_format
        )
        assert batched.shape == per_pose.shape
        assert batched.dtype == per_pose.dtype
        np.testing.assert_allclose(batched, per_pose, rtol=0, atol=tolerance)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_joint_batched_matches_per_pose(dtype):
    rng = np.random.default_rng(1)
    processor = _processor()
    action = rng.normal(size=(HORIZON, 7)).astype(dtype)
    reference = rng.normal(size=7).astype(dtype)
    batched = processor._convert_to_absolute_action(
        action, reference, ActionType.NON_EEF, ActionFormat.DEFAULT
    )
    per_pose = processor._convert_to_absolute_action_objects(
        action, reference, ActionType.NON_EEF, ActionFormat.DEFAULT
    )
    assert batched.dtype == per_pose.dtype
    np.testing.assert_array_equal(batched, per_pose)


def test_identity_relative_action_returns_reference():
    processor = _processor()
    rng = np.random.default_rng(2)
    reference = _random_eef(rng, 1, ActionFormat.XYZ_ROT6D, np.float64)[0]
    identity = np.tile(np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float64), (HORIZON, 1))
    absolute = processor._convert_to_absolute_action(
        identity, reference, ActionType.EEF, ActionFormat.XYZ_ROT6D
    )
    expected = processor._convert_to_absolute_action_objects(
        identity[:1], reference, ActionType.EEF, ActionFormat.XYZ_ROT6D
    )
    np.testing.assert_allclose(absolute, np.tile(expected, (HORIZON, 1)), atol=1e-12)
    np.testing.assert_allclose(absolute[:, :3], np.tile(reference[:3], (HORIZON, 1)), atol=1e-12)

