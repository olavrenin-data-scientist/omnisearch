"""Tests for the false-positive perception model (decoy landmarks).

Verifies that:
  * the model is OFF by default (n_decoys == 0) and changes nothing,
  * a drone within camera range of a decoy falsely scouts it at the
    configured rate,
  * a ground robot investigating a scouted decoy dismisses it (a wasted trip),
    pays the configured penalty, and the decoy is never confirmable.

The heavy VMAS dependency is mocked: we bind an extracted copy of
``_process_decoy_false_positives`` (kept in sync with envs/wildfire_search.py)
to a lightweight stub object and exercise the tensor bookkeeping directly.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from envs.wildfire_search import WildfireSearchScenario  # noqa: F401
    HAS_VMAS = True
except ImportError:
    HAS_VMAS = False

skip_no_vmas = pytest.mark.skipif(not HAS_VMAS, reason="vmas not installed")


def _process_decoy_false_positives(self, agent_pos, confirm_range, device):
    """Extracted decoy false-positive logic (mirrors envs/wildfire_search.py)."""
    n_ground = self.n_ground
    if self.n_decoys == 0:
        self.metric_false_positive_detections = torch.zeros(self.world.batch_dim, device=device)
        self.metric_false_positive_trips = torch.zeros(self.world.batch_dim, device=device)
        return torch.zeros(self.world.batch_dim, max(n_ground, 0), device=device)

    decoy_pos = torch.stack([d.state.pos for d in self._decoys], dim=1)
    agent_decoy_dists = torch.cdist(agent_pos, decoy_pos)

    if self.n_drones > 0 and self.drone_false_positive_rate > 0.0:
        drone_decoy_dists = agent_decoy_dists[:, :self.n_drones, :]
        footprint = self._drone_camera_ranges().unsqueeze(-1)
        in_view = drone_decoy_dists <= footprint
        draw = torch.rand_like(drone_decoy_dists)
        false_det = in_view & (draw < self.drone_false_positive_rate)
        false_det = false_det & ~self.dismissed_decoys.unsqueeze(1)
        self.step_decoy_false_detections = false_det
        newly_scouted_decoys = false_det.any(dim=1) & ~self.dismissed_decoys
        self.scouted_decoys = self.scouted_decoys | newly_scouted_decoys
        if self.n_drones:
            self.known_decoys_by_agent[:, :self.n_drones] |= false_det
    else:
        self.step_decoy_false_detections = torch.zeros(
            self.world.batch_dim, self.n_drones, self.n_decoys, dtype=torch.bool, device=device,
        )

    decoy_penalty = torch.zeros(self.world.batch_dim, max(n_ground, 0), device=device)
    newly_dismissed = torch.zeros_like(self.dismissed_decoys)
    if n_ground > 0:
        ground_decoy_dists = agent_decoy_dists[:, self.n_drones:, :]
        within = ground_decoy_dists < confirm_range
        investigatable = within & self.scouted_decoys.unsqueeze(1) & ~self.dismissed_decoys.unsqueeze(1)
        self.known_decoys_by_agent[:, self.n_drones:] |= investigatable
        newly_dismissed = investigatable.any(dim=1) & ~self.dismissed_decoys
        self.dismissed_decoys = self.dismissed_decoys | newly_dismissed
        trips_per_ground = (investigatable.float().sum(dim=2))
        decoy_penalty = trips_per_ground * self.r_decoy_pursuit_penalty

    self.metric_false_positive_detections = self.scouted_decoys.float().sum(dim=1)
    self.metric_false_positive_trips = newly_dismissed.float().sum(dim=1)
    return decoy_penalty


def _landmark_at(positions):
    """Build a decoy stub whose ``state.pos`` is [B, 2]."""
    lm = MagicMock()
    lm.state.pos = torch.as_tensor(positions, dtype=torch.float32)
    return lm


def _make_stub(B, D, G, K, *, fp_rate=1.0, penalty=-1.0, footprint=1.0):
    s = MagicMock()
    s.n_decoys = K
    s.n_drones = D
    s.n_ground = G
    s.n_agents = D + G
    s.drone_false_positive_rate = fp_rate
    s.r_decoy_pursuit_penalty = penalty
    s.world = MagicMock()
    s.world.batch_dim = B
    s.scouted_decoys = torch.zeros(B, K, dtype=torch.bool)
    s.dismissed_decoys = torch.zeros(B, K, dtype=torch.bool)
    s.known_decoys_by_agent = torch.zeros(B, D + G, K, dtype=torch.bool)
    s.step_decoy_false_detections = torch.zeros(B, D, K, dtype=torch.bool)
    s._drone_camera_ranges = lambda: torch.full((B, D), float(footprint))
    s._process_decoy_false_positives = types.MethodType(_process_decoy_false_positives, s)
    return s


class TestDecoyFalsePositives:
    def test_disabled_by_default(self):
        """n_decoys == 0 returns a zero penalty and zero metrics."""
        s = _make_stub(B=2, D=1, G=1, K=0)
        agent_pos = torch.zeros(2, 2, 2)
        penalty = s._process_decoy_false_positives(agent_pos, torch.tensor(0.5), torch.device("cpu"))
        assert penalty.shape == (2, 1)
        assert not penalty.any()
        assert float(s.metric_false_positive_detections.sum()) == 0.0
        assert float(s.metric_false_positive_trips.sum()) == 0.0

    def test_drone_in_view_falsely_scouts_decoy(self):
        """A decoy inside the footprint is scouted when fp_rate == 1.0."""
        B, D, G, K = 1, 1, 1, 1
        s = _make_stub(B, D, G, K, fp_rate=1.0, footprint=1.0)
        s._decoys = [_landmark_at([[0.0, 0.0]])]
        # Drone on top of decoy (in view); ground robot far away.
        agent_pos = torch.tensor([[[0.0, 0.0], [10.0, 10.0]]])
        s._process_decoy_false_positives(agent_pos, torch.tensor(0.5), torch.device("cpu"))
        assert bool(s.scouted_decoys[0, 0])
        assert bool(s.known_decoys_by_agent[0, 0, 0])   # drone knows the false report
        assert not bool(s.dismissed_decoys[0, 0])       # ground hasn't investigated yet
        assert float(s.metric_false_positive_detections[0]) == 1.0

    def test_zero_rate_never_scouts(self):
        """fp_rate == 0 produces no false detections even when in view."""
        s = _make_stub(B=1, D=1, G=1, K=1, fp_rate=0.0, footprint=1.0)
        s._decoys = [_landmark_at([[0.0, 0.0]])]
        agent_pos = torch.tensor([[[0.0, 0.0], [10.0, 10.0]]])
        s._process_decoy_false_positives(agent_pos, torch.tensor(0.5), torch.device("cpu"))
        assert not s.scouted_decoys.any()

    def test_out_of_view_not_scouted(self):
        """A decoy outside the camera footprint is never scouted."""
        s = _make_stub(B=1, D=1, G=1, K=1, fp_rate=1.0, footprint=1.0)
        s._decoys = [_landmark_at([[50.0, 50.0]])]
        agent_pos = torch.tensor([[[0.0, 0.0], [10.0, 10.0]]])
        s._process_decoy_false_positives(agent_pos, torch.tensor(0.5), torch.device("cpu"))
        assert not s.scouted_decoys.any()

    def test_ground_investigation_dismisses_and_penalizes(self):
        """A ground robot reaching a scouted decoy dismisses it and pays the penalty."""
        s = _make_stub(B=1, D=1, G=1, K=1, fp_rate=1.0, penalty=-2.0, footprint=1.0)
        s.scouted_decoys = torch.tensor([[True]])     # already falsely scouted
        s._decoys = [_landmark_at([[0.0, 0.0]])]
        # Ground robot sits on the decoy; drone elsewhere out of view.
        agent_pos = torch.tensor([[[20.0, 20.0], [0.0, 0.0]]])
        penalty = s._process_decoy_false_positives(agent_pos, torch.tensor(0.5), torch.device("cpu"))
        assert bool(s.dismissed_decoys[0, 0])
        assert float(s.metric_false_positive_trips[0]) == 1.0
        assert pytest.approx(float(penalty[0, 0])) == -2.0
        # Dismissed decoy stays known so observation can mark it as a false positive.
        assert bool(s.known_decoys_by_agent[0, 1, 0])

    def test_dismissed_decoy_not_retriggered(self):
        """Once dismissed, a decoy is never scouted again."""
        s = _make_stub(B=1, D=1, G=1, K=1, fp_rate=1.0, footprint=1.0)
        s.dismissed_decoys = torch.tensor([[True]])
        s._decoys = [_landmark_at([[0.0, 0.0]])]
        agent_pos = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
        s._process_decoy_false_positives(agent_pos, torch.tensor(0.5), torch.device("cpu"))
        assert not s.scouted_decoys.any()
        assert float(s.metric_false_positive_trips[0]) == 0.0


@skip_no_vmas
class TestDecoyConfigIntegration:
    def test_disabled_by_default_in_make_world(self):
        scenario = WildfireSearchScenario()
        scenario.make_world(batch_dim=1, device=torch.device("cpu"),
                            n_drones=1, n_ground=1, n_survivors=2)
        assert scenario.n_decoys == 0
        assert scenario.drone_false_positive_rate == 0.0
        assert len(scenario._decoys) == 0

    def test_decoys_configured(self):
        scenario = WildfireSearchScenario()
        scenario.make_world(batch_dim=1, device=torch.device("cpu"),
                            n_drones=1, n_ground=1, n_survivors=2,
                            n_decoys=3, drone_false_positive_rate=0.2,
                            r_decoy_pursuit_penalty=-1.5)
        assert scenario.n_decoys == 3
        assert scenario.drone_false_positive_rate == 0.2
        assert scenario.r_decoy_pursuit_penalty == -1.5
        assert len(scenario._decoys) == 3
        assert scenario.scouted_decoys.shape == (1, 3)

    def test_decoy_enabled_default_false_positive_rate(self):
        scenario = WildfireSearchScenario()
        scenario.make_world(batch_dim=1, device=torch.device("cpu"),
                            n_drones=1, n_ground=1, n_survivors=2,
                            n_decoys=1)
        assert scenario.n_decoys == 1
        assert scenario.drone_false_positive_rate == 0.05
