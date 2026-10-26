# DiverCam: underwater coral-reef surveys with Metashape pseudo-GT.
#
# Not a public dataset -- see loggerhead/docs and the manifest at
# third_party/Pi3/datasets/divercam_manifest.yaml for provenance.

"""
DiverCam underwater survey dataset using WAI format data.
"""

import json
import csv
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from mapanything.datasets.base.base_dataset import BaseDataset
from mapanything.utils.wai.core import load_data, load_frame


class _SplitMaskedCovisibility:
    """Covisibility rows with out-of-split and low-coverage views zeroed.

    BaseDataset._random_walk_sampling only ever takes a row, a column and len(),
    so masking can be a view rather than a 3 GB copy of the memmap. Zeroing is
    enough to make the sampler skip a view: it thresholds covisibility, so a zero
    row never becomes a candidate.
    """

    def __init__(self, covis, allowed):
        self._covis, self._allowed = covis, allowed
        # The sampler draws its start node by index, not through __getitem__, so
        # zeroing alone cannot keep an out-of-split view out of walk[0].
        self.allowed_nodes = np.flatnonzero(allowed)

    def __len__(self):
        return len(self._covis)

    def __getitem__(self, key):
        row, col = key if isinstance(key, tuple) else (key, slice(None))
        if isinstance(row, slice):  # column access: covis[:, i]
            return np.asarray(self._covis[:, col]) * self._allowed
        return np.asarray(self._covis[row, col]) * self._allowed


# Cam0 <- IMU/DVL (Nortek) extrinsic, Kalibr convention p_cam = T_CAM_IMU @ p_imu.
# Downward camera (left), 2026-05-18-20-52-54 session, cam0. Copied from
# loggerhead/scripts/tier0_fusion.py rather than imported: that lives outside this
# submodule, and the numbers are calibration constants, not logic.
T_CAM_IMU_DWN = np.array([
    [-0.9995122254784973, 0.030794861284595153, -0.00519496270292499, -0.003938369353851938],
    [-0.03076301546465094, -0.9995080768743684, -0.0061025521238302904, 0.18638744857761158],
    [-0.005380334426769991, -0.005939762736419775, 0.9999678850943623, 0.008426479247768631],
    [0.0, 0.0, 0.0, 1.0],
])
# Static TF base_link -> dvl_link from diver_zedxm_system.launch (yaw pi/2, roll pi).
_T_BASE_DVL = np.eye(4)
_T_BASE_DVL[:3, :3] = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
# base_link <- camera. The EKF publishes the BODY pose, so a body pose must be
# post-multiplied by this to become a camera pose. Skipping it leaves a constant
# conjugation by the ~19 cm camera/IMU offset in the relative poses, which at the
# rotations a diver makes is centimetres -- the size of the error being measured.
T_BASE_CAM_DWN = _T_BASE_DVL @ np.linalg.inv(T_CAM_IMU_DWN)

# Forward-left <- downward-left, the direct cross-camera Kalibr calibration (63.40 deg,
# 7.78 cm). Copied from scripts/divercam_rig.py, which documents why this calibration
# rather than the separate forward-IMU session one: the two disagree by 0.56 deg and
# mixing them would bias the prior against the target on every cross-camera pair.
# Named in the current T_A_B convention; the constants above keep their older names
# because tier0_fusion.py, which they were copied from, is not being touched.
T_FwdCam_DwnCam = np.array([
    [0.99969651911556, 0.008268429247972756, 0.023205661938492293, 0.0018840797739952551],
    [-0.024450998089428966, 0.4478687407825575, 0.8937649241955496, 0.06123247357646312],
    [-0.003003058551387631, -0.8940610852215956, 0.44793499252873664, -0.04801432725985006],
    [0.0, 0.0, 0.0, 1.0],
])


def _load_ekf_camera_poses(csv_path):
    """frame name -> cam2world 4x4 from an EKF sync CSV, in the odom frame.

    Returns absolute poses on purpose. MapAnything re-expresses pose priors
    relative to view 0 inside the model, and a camera-to-camera relative pose is
    independent of the world frame, so the EKF's drifting absolute XY cancels
    within a window and no composition of rel_* is needed here.
    """
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                t = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
                q = np.array([float(row["qx"]), float(row["qy"]),
                              float(row["qz"]), float(row["qw"])])
            except (ValueError, KeyError, TypeError):
                continue  # frames outside the odometry time range are NaN
            if not (np.isfinite(t).all() and np.isfinite(q).all()):
                continue
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(q).as_matrix()
            T[:3, 3] = t
            out[row["frame"]] = (T @ T_BASE_CAM_DWN).astype(np.float32)
    return out


class DiverCamWAI(BaseDataset):
    """
    Diver-operated ZED X Mini surveys of coral reef, 5 sites in the US Virgin
    Islands. Depth and poses come from Metashape, scaled against in-scene
    AprilTags; depth is stored in raw chunk-internal units and scaled here.
    """

    def __init__(
        self,
        *args,
        ROOT,
        split,
        split_dir=None,
        manifest=None,
        min_coverage=0.3,
        temporal_prob=0.0,
        temporal_strides=(1, 3, 5, 10, 15),
        overfit_num_sets=None,
        pose_prior="none",
        pose_prior_noise_trans=0.0,
        pose_prior_noise_rot=0.0,
        num_forward_views=0,
        forward_sampling="auto",
        forward_offset=-60,
        forward_min_coverage=0.0,
        forward_covis_thres=0.15,
        max_forward_fraction=0.25,
        covis_version="v0",
        **kwargs,
    ):
        """
        Args:
            ROOT: Root directory of the WAI-format divercam dataset.
            split: name of a yaml in split_dir, or "train"/"val"/"test" within it.
            split_dir: directory of split specs (default: loggerhead/configs/splits).
            manifest: divercam_manifest.yaml (default: alongside the Pi3 datasets).
            temporal_prob: probability of drawing a consecutive-frame window instead
                          of a covisibility set. Covisibility sets span minutes with
                          ~0.8 m baselines, so a model trained only on them has never
                          seen what a deployed system sees, and measures worse at
                          stride 1 than at stride 3. Mixing closes that gap.
            temporal_strides: strides to draw from when sampling a temporal window.
            min_coverage: drop frames whose valid-depth fraction is below this.
                          Metashape MVS leaves 40-50% of pixels empty on the worst
                          frames, which carry too little supervision to be worth a
                          view slot.
            overfit_num_sets: if set, truncate to this many sets.
            pose_prior: what to feed the model's camera pose conditioning.
                "none"      no prior; the vision-only baseline.
                "ekf"       DVL+IMU poses from the dive's left_ekf_sync.csv,
                            converted body->camera. This is the real signal.
                "gt"        Metashape poses. LEAKAGE -- the pose head is
                            supervised on these, so this is only an upper bound
                            and a plumbing check, never a reportable result.
                "gt+noise"  Metashape poses perturbed by the two noise args, to
                            measure how much prior accuracy actually buys before
                            depending on the EKF's real error characteristics.
                BaseDataset fills camera_pose_quats/trans from the ground-truth
                camera_pose, so MapAnything's own conditioning trains on perfect
                priors. Anything other than "none" replaces them after the fact.
            pose_prior_noise_trans: metres, per-axis Gaussian sigma on the prior.
            pose_prior_noise_rot: degrees, Gaussian sigma on the prior rotation.
            num_forward_views: how many of the set's num_views come from the forward
                          camera. Budgeted INSIDE num_views, never appended, because
                          BaseDataset._getitem_fn asserts len(views) == num_views.
                          Requires a scene converted with divercam_to_wai --forward.
            forward_sampling: how forward views are chosen once the downward views are.
                "auto"     follow the branch that chose the DOWNWARD views -- "offset"
                           for a temporal window, "covis" for a covisibility set. The
                           default, because the two are otherwise incoherent: with
                           temporal_prob 0.5 and a fixed "covis", half of all windows
                           are a temporally dense downward set plus forward frames
                           drawn from anywhere in the survey, which is precisely the
                           regime the temporal branch exists to avoid.
                "covis"    draw them from the covisibility matrix against the chosen
                           downward set. The principled option: measured same-instant
                           footprint overlap between the two cameras is exactly ZERO
                           (they abut -- downward reaches 0.75 m ahead, the forward
                           frame's near edge lands at 0.79 m), so pairing a forward
                           view with its own timestamp's downward view would supply
                           two views with no shared geometry.
                "offset"   take the forward frame `forward_offset` frames from a
                           chosen downward one. Peak overlap sits at -50 to -70
                           frames, i.e. the forward camera saw this patch ~4 s ago,
                           which makes it a look-ahead sensor. Cheaper than "covis"
                           and independent of the covisibility matrix, so it isolates
                           that look-ahead relation from the sampler.
            forward_offset: frame offset used by forward_sampling="offset".
            max_forward_fraction: ceiling on the forward share of a window, applied
                          to the SAMPLED view count rather than num_views, because
                          variable_num_views draws that count from U(2, num_views)
                          and a fixed forward count would make small sets
                          forward-dominated. 1.0 with num_forward_views == num_views
                          gives an all-forward window.
            forward_covis_thres: minimum score for a forward frame to be eligible.
                          NOT comparable to covisibility_thres: cross-camera cells
                          hold the fraction of the forward image covered, which tops
                          out near 0.31, so the 0.25 used for same-camera pairs would
                          admit only the extreme tail (measured median candidates per
                          downward frame falls from 416 to 15). 0.15 is about half the
                          achievable maximum and puts the candidate count on the same
                          scale as the same-camera neighbourhood (128 vs 99).
            covis_version: subdirectory of covisibility/ to read. "v1" holds the
                          cross-camera coverage statistic; "v0" predates it and has
                          cross-camera cells pinned near 1.0, useless for scoring.
                          Same-camera cells are bit-identical between the two.
            forward_min_coverage: coverage floor for forward frames. Separate from
                          min_coverage and defaulting to 0, because a forward frame's
                          depth is rendered from downward geometry and is structurally
                          sparser -- and is absent entirely in the input-only arm.
        """
        super().__init__(*args, **kwargs)
        self.ROOT = Path(ROOT)
        self.split_name, self.part = split.rsplit(":", 1) if ":" in split else (split, "train")
        self.split_dir = Path(split_dir) if split_dir else _default("configs/splits")
        self.manifest = Path(manifest) if manifest else _default(
            "third_party/Pi3/datasets/divercam_manifest.yaml")
        self.min_coverage = min_coverage
        self.temporal_prob = temporal_prob
        self.temporal_strides = list(temporal_strides)
        # when set, _get_views uses these indices instead of sampling. Evaluation
        # needs deployment-style consecutive windows as well as covisibility sets,
        # and going through the normal path keeps BaseDataset's post-processing
        # (transforms, pts3d, masks) identical between the two.
        self.forced_view_indices = None
        self.overfit_num_sets = overfit_num_sets
        assert pose_prior in ("none", "ekf", "gt", "gt+noise"), pose_prior
        self.pose_prior = pose_prior
        self.pose_prior_noise_trans = pose_prior_noise_trans
        self.pose_prior_noise_rot = pose_prior_noise_rot
        assert forward_sampling in ("auto", "covis", "offset"), forward_sampling
        # coverage.npy holds the DEPTH-validity fraction, which is 0 for every
        # forward frame by construction, so any positive floor silently excludes all
        # of them. Fail loudly instead of training a forward config with no forward
        # views, which looks exactly like the baseline in every log.
        assert not (num_forward_views and forward_min_coverage > 0), (
            "forward_min_coverage must stay 0: forward frames have no depth, so "
            "their coverage is 0 and any floor above it removes every one of them"
        )
        self.num_forward_views = num_forward_views
        self.forward_sampling = forward_sampling
        self.forward_offset = forward_offset
        self.forward_covis_thres = forward_covis_thres
        self.max_forward_fraction = max_forward_fraction
        self.covis_version = covis_version
        self.forward_min_coverage = forward_min_coverage
        self._ekf_cache = {}
        self._load_data()

        # overwritten per scene in _get_views; this is the default for a mixed split
        self.is_metric_scale = all(self.scene_metric.values())
        self.is_synthetic = False

    def _load_data(self):
        spec = yaml.safe_load((self.split_dir / f"{self.split_name}.yaml").read_text())
        dives = {d["label"]: d for d in yaml.safe_load(self.manifest.read_text())["dives"]}
        if "deprecated" in spec:
            raise ValueError(
                f"split {self.split_name} is deprecated and must not be used: "
                f"{spec['deprecated']}")
        if self.part not in spec:
            raise ValueError(f"split {self.split_name} has no '{self.part}' part")

        # Precomputed per-survey frame masks, when the split uses them instead of
        # segment indices. Every survey is one segment since the 2026-10 rebuild,
        # so `segments`/`exclude_segments` has nothing left to select; validation
        # is held back spatially instead. See scripts/divercam_make_split.py.
        masks = {}
        if "split_masks" in spec:
            mask_path = self.ROOT / "splits" / f"{spec['split_masks']}.npz"
            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"split {self.split_name} wants masks {mask_path}; build them "
                    f"with scripts/divercam_make_split.py --name {spec['split_masks']}")
            masks = dict(np.load(mask_path))

        self.scenes, self.scene_allowed, self.scene_metric = [], {}, {}
        self.scene_segments, self.scene_dive_root = {}, {}
        # forward-camera bookkeeping, all empty for a down-only scene
        self.scene_allowed_fwd, self.scene_fwd_of = {}, {}
        for member in spec[self.part]:
            label = member["survey"]
            scene_root = self.ROOT / label
            meta = json.loads((scene_root / "scene_meta.json").read_text())
            segments = np.array([f["segment"] for f in meta["frames"]])
            coverage = np.load(scene_root / "coverage.npy")

            # Forward frames are appended after all downward ones by
            # divercam_to_wai.py --forward, and carry a `partner_frame` naming the
            # downward frame they are rigidly attached to.
            is_fwd = np.array([f["frame_name"].endswith("_fwd") for f in meta["frames"]])
            keep = set(member.get("segments", range(segments.max() + 1)))
            keep -= set(member.get("exclude_segments", []))
            in_segment = np.isin(segments, list(keep))
            # Per-camera coverage floors: a forward frame's depth is rendered from
            # downward geometry and is structurally sparser, and in the input-only
            # arm it has none at all, so the downward 0.3 would exclude every one.
            allowed = in_segment & (coverage >= self.min_coverage) & ~is_fwd
            key = f"{label}__{self.part}"
            if masks and key in masks:
                # The masks cover DOWNWARD frames only, which the converter writes
                # first; pad with False so forward frames are admitted solely
                # through the partner rule below.
                m = masks[key]
                if len(m) > len(meta["frames"]):
                    raise ValueError(
                        f"mask {key} has {len(m)} frames but {label} has "
                        f"{len(meta['frames'])}; the masks predate this tree")
                padded = np.zeros(len(meta["frames"]), bool)
                padded[: len(m)] = m
                allowed &= padded
            elif masks and self.part in ("train", "val"):
                # A train/val member with no mask is a silent full-survey include,
                # which is how a validation set ends up containing its own training
                # frames. A held-out test site legitimately has no mask.
                raise KeyError(
                    f"split {self.split_name} lists {label} in '{self.part}' but "
                    f"its mask file has no {key}; rebuild with "
                    f"scripts/divercam_make_split.py")
            # the dive's down/ directory, which is where left_ekf_sync.csv lives
            self.scene_dive_root[label] = Path(dives[label]["root"])
            if self.pose_prior == "ekf":
                # Exclude frames whose EKF pose is NaN -- the handful at each end
                # of a dive, outside the odometry time range -- from sampling.
                # A view set missing the prior for even one view has to drop it
                # for all of them (the model reads view 0 as the reference), and
                # a BATCH that mixes sets with and without prior_camera_pose_*
                # dies in default_collate, which is not something the dataset can
                # see. That killed a 12-epoch run in epoch 11. Costs ~9 frames of
                # 64k, so excluding them is cheaper than handling them.
                ekf = self._ekf_poses(label)
                allowed &= np.array(
                    [f["frame_name"] in ekf for f in meta["frames"]])
            if not allowed.any():
                continue

            # down index -> its forward frame's index, -1 where there is none
            idx_of = {f["frame_name"]: i for i, f in enumerate(meta["frames"])}
            fwd_of = np.full(len(meta["frames"]), -1, np.int64)
            for i, f in enumerate(meta["frames"]):
                if "partner_frame" in f:
                    fwd_of[idx_of[f["partner_frame"]]] = i
            # A forward frame is allowed only if its PARTNER is. That inherits the
            # segment keep, the coverage floor and -- critically -- the EKF-pose
            # requirement in one step, which is what keeps the all-or-nothing prior
            # invariant true by construction: the forward prior is derived from the
            # partner's EKF row, so it exists exactly when the partner's does.
            allowed_fwd = is_fwd & (coverage >= self.forward_min_coverage)
            has_allowed_partner = np.zeros(len(meta["frames"]), bool)
            has_allowed_partner[fwd_of[allowed & (fwd_of >= 0)]] = True
            allowed_fwd &= has_allowed_partner

            self.scenes.append(label)
            self.scene_allowed[label] = allowed.astype(np.float32)
            self.scene_allowed_fwd[label] = allowed_fwd
            self.scene_fwd_of[label] = fwd_of
            self.scene_segments[label] = segments
            # metric supervision needs the dive scaled AND the split to trust it
            self.scene_metric[label] = bool(
                dives[label]["metric"] and member.get("metric_eval", True))

        self.num_of_scenes = len(self.scenes)
        if self.overfit_num_sets is not None:
            self.scenes = self.scenes[: self.overfit_num_sets]
            self.num_of_scenes = len(self.scenes)

    def _ekf_poses(self, scene_name):
        """Cached frame -> cam2world from the dive's EKF sync CSV ({} if absent)."""
        if scene_name not in self._ekf_cache:
            # The dive tree holds the authoritative copy, but it is a separate
            # tree from the WAI one and hydra has no /srv/warplab mount -- only
            # the WAI scenes get rsynced there. So fall back to a copy inside the
            # scene directory, which keeps the WAI tree self-contained.
            for csv_path in (self.scene_dive_root[scene_name] / "left_ekf_sync.csv",
                             Path(self.ROOT) / scene_name / "left_ekf_sync.csv"):
                if csv_path.is_file():
                    self._ekf_cache[scene_name] = _load_ekf_camera_poses(csv_path)
                    break
            else:
                self._ekf_cache[scene_name] = {}
        return self._ekf_cache[scene_name]

    def _getitem_fn(self, idx):
        """BaseDataset's item, plus the pose prior the model conditions on.

        Done here rather than in _get_views because BaseDataset fills
        camera_pose_quats/trans from the ground-truth pose AFTER _get_views
        returns, so anything set earlier would be overwritten.
        """
        views = super()._getitem_fn(idx)
        if self.pose_prior == "none":
            return views

        scene_name = views[0]["label"]
        priors = []
        for v in views:
            if self.pose_prior == "ekf":
                stem = Path(v["instance"]).stem
                # A forward frame is named after its downward partner, so one EKF row
                # serves both views of a rig pair. The two eyes are hardware-synced to
                # ~24 us, so this is not an approximation -- and it makes the pair's
                # RELATIVE prior exactly the rig calibration, with no interpolation
                # noise of its own.
                is_fwd = stem.endswith("_fwd")
                T = self._ekf_poses(scene_name).get(stem[:-4] if is_fwd else stem)
                if T is not None and is_fwd:
                    T = T @ np.linalg.inv(T_FwdCam_DwnCam)
            else:
                T = np.asarray(v["camera_pose"], dtype=np.float32)
                if self.pose_prior == "gt+noise":
                    T = self._perturb(T)
            priors.append(T)

        # All or nothing per view set: the model reads view 0's prior as the
        # reference for every other view, so a set missing any one of them cannot
        # be partially conditioned. Falling back to the GT pose would leak.
        #
        # With pose_prior='ekf', _load_data already excludes frames that have no
        # EKF pose, so this is unreachable from the sampling path. It remains a
        # backstop for forced_view_indices, which bypasses the allowed mask --
        # callers doing that must use batch size 1, because a batch mixing sets
        # with and without these keys fails in default_collate.
        if any(T is None for T in priors):
            if self.forced_view_indices is None:
                # Sampled sets must never reach here: _load_data excludes
                # frames without an EKF pose. If one does, the set goes into the
                # batch without prior_camera_pose_* and default_collate dies far
                # from the cause -- which cost a 12-epoch run and a 21-epoch one.
                missing = [Path(v["instance"]).stem
                           for v, T in zip(views, priors) if T is None]
                raise RuntimeError(
                    f"{scene_name}: sampled views have no {self.pose_prior} pose "
                    f"prior: {missing}. The split mask should have excluded them."
                )
            return views

        for v, T in zip(views, priors):
            v["prior_camera_pose_quats"] = (
                Rotation.from_matrix(T[:3, :3]).as_quat().astype(np.float32))
            v["prior_camera_pose_trans"] = np.ascontiguousarray(
                T[:3, 3], dtype=np.float32)
        return views

    def _perturb(self, T):
        """Gaussian noise on a pose: metres on translation, degrees on rotation."""
        out = np.array(T, dtype=np.float32)
        if self.pose_prior_noise_trans > 0:
            out[:3, 3] += self._rng.normal(
                0.0, self.pose_prior_noise_trans, 3).astype(np.float32)
        if self.pose_prior_noise_rot > 0:
            axis = self._rng.normal(size=3)
            axis /= np.linalg.norm(axis) + 1e-12
            angle = np.deg2rad(self._rng.normal(0.0, self.pose_prior_noise_rot))
            out[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix() @ out[:3, :3]
        return out

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        scene_name = self.scenes[sampled_idx]
        scene_root = self.ROOT / scene_name
        # BaseDataset._getitem_fn stamps view["is_metric_scale"] from this attribute
        # immediately after _get_views returns, so setting it here makes the flag
        # per-scene rather than per-dataset -- which is what metric_eval: false in a
        # split needs. Each dataloader worker holds its own copy of self.
        self.is_metric_scale = self.scene_metric[scene_name]
        scene_meta = load_data(scene_root / "scene_meta.json", "scene_meta")
        depth_scale = scene_meta["divercam"]["depth_scale_to_metres"]
        file_names = list(scene_meta["frame_names"].keys())

        covis_dir = scene_root / "covisibility" / self.covis_version
        # WAI's mmap loader takes the shape from the filename; one npy per dir
        covis_name = next(f for f in os.listdir(covis_dir) if f.endswith(".npy"))
        covis_raw = load_data(covis_dir / covis_name, "mmap")
        covis = _SplitMaskedCovisibility(covis_raw, self.scene_allowed[scene_name])

        # Budget forward views INSIDE num_views: BaseDataset._getitem_fn asserts
        # len(views) == num_views. At least one downward view must remain, and it
        # must be view 0 -- the model reads views[0]'s prior as the reference frame
        # for every other view and cannot mask it (model.py:745-778).
        #
        # num_forward_views is the count at the FULL view budget, and it scales down
        # with num_views_to_sample rather than staying fixed. variable_num_views
        # draws that count from U(2, num_views), so a fixed 2 would make a 3-view set
        # 67% forward while the 8-view set it was chosen for is 25% -- the small sets
        # would train a different recipe than the one being evaluated. Capping at a
        # max_forward_fraction of the window holds the ratio at or below its nominal
        # value for every draw; at the default 0.25 the cost is that sets of 2 or 3
        # views are purely downward. Raise it to 1.0 for an all-forward window.
        n_fwd = 0
        if self.num_forward_views and self.forced_view_indices is None:
            n_fwd = min(
                self.num_forward_views,
                int(num_views_to_sample * self.max_forward_fraction),
            )
        n_down = num_views_to_sample - n_fwd

        if n_down == 0:
            # All-forward window. There is no downward set to score against, and
            # forward-to-forward cells of the covisibility table are degenerate (a
            # median of 1 neighbour, i.e. only itself) because the statistic needs
            # the SOURCE frame's own 3D points and a forward frame has no depth. So
            # the only selection available is temporal: take a window of downward
            # partners and map every one of them to its forward frame. Covisibility
            # selection for all-forward windows becomes possible once
            # divercam_fuse_depth.py gives forward frames depth of their own.
            part = self._sample_temporal_window(scene_name, n_fwd)
            if part is None:
                part = self._sample_view_indices(n_fwd, len(file_names), covis)
            fwd_of = self.scene_fwd_of[scene_name]
            allowed_fwd = self.scene_allowed_fwd[scene_name]
            picked = [int(fwd_of[p]) for p in part
                      if fwd_of[p] >= 0 and allowed_fwd[fwd_of[p]]]
            if len(picked) < n_fwd:  # top up with downward rather than return short
                extra = self._sample_view_indices(
                    n_fwd - len(picked), len(file_names), covis)
                picked = list(extra) + picked
            return self._load_views(scene_name, picked[:n_fwd], resolution)

        used_temporal = False
        if self.forced_view_indices is not None:
            view_indices = self.forced_view_indices
        elif self.temporal_prob > 0 and self._rng.random() < self.temporal_prob:
            view_indices = self._sample_temporal_window(scene_name, n_down)
            used_temporal = view_indices is not None
            if view_indices is None:  # no window fits; fall back
                view_indices = self._sample_view_indices(
                    n_down, len(file_names), covis)
        else:
            view_indices = self._sample_view_indices(
                n_down, len(file_names), covis)

        # Match the forward sampler to the branch that chose the downward views.
        # Mixing them is incoherent: a temporally dense downward window plus two
        # forward frames drawn by covisibility can put those forward frames
        # hundreds of seconds away, which is exactly what the temporal branch
        # exists to avoid. "offset" keeps a temporal window temporal, and
        # covisibility sets have no temporal structure to preserve.
        fwd_mode = self.forward_sampling
        if fwd_mode == "auto":
            fwd_mode = "offset" if used_temporal else "covis"

        if n_fwd:
            view_indices = list(view_indices)
            fwd = self._sample_forward_views(
                scene_name, covis_raw, view_indices, n_fwd, fwd_mode)
            # Short of forward candidates, top up with downward views rather than
            # returning a short set, which would trip the num_views assert. Rare,
            # and it degrades towards the downward-only baseline rather than failing.
            if len(fwd) < n_fwd:
                extra = self._sample_view_indices(
                    n_down + (n_fwd - len(fwd)), len(file_names), covis)
                view_indices = list(extra)[:n_down + n_fwd - len(fwd)]
            view_indices = list(view_indices) + fwd
        return self._load_views(scene_name, view_indices, resolution)

    def _sample_forward_views(self, scene_name, covis_raw, down_indices, n_fwd, mode):
        """Forward-camera frame indices to accompany an already-chosen downward set.

        One scoring rule for both branches. A forward frame's score is the largest
        value it reaches against any of the chosen downward frames, read from the
        cross-camera cells of the covisibility table. Those cells hold the fraction
        of the FORWARD image that downward-surveyed seafloor fills (see TWO
        STATISTICS in scripts/divercam_covisibility.py), so the score answers "how
        much of this forward frame does the window already cover" -- which is the
        question that matters for putting them in a window together. Candidates are
        sampled with probability proportional to that score rather than uniformly,
        because the cells now discriminate: they run 0.01-0.31 with the ceiling set
        by how much of a forward frame one downward footprint can cover.

        The branches differ only in which candidates are eligible:

          "offset"  a temporal window, so candidates are restricted to forward
                    frames whose downward PARTNER sits in the window's own index
                    range extended backwards by forward_offset. The forward camera
                    is a look-ahead -- it imaged this patch roughly 4 s before the
                    downward camera reached it -- so the useful partners sit before
                    the window, not inside it. Scoring within that range beats
                    taking a fixed offset, which cannot adapt to the diver's speed,
                    the altitude or the window's stride.
          "covis"   a covisibility set, which has no temporal structure to preserve,
                    so every allowed forward frame is eligible.

        Never returns an index that could land at position 0 of the set, because the
        caller appends these after the downward views.
        """
        allowed_fwd = self.scene_allowed_fwd[scene_name]
        down = np.asarray(down_indices, dtype=np.int64)

        eligible = allowed_fwd.copy()
        if mode == "offset":
            # Partner index of each forward frame, i.e. the inverse of fwd_of.
            fwd_of = self.scene_fwd_of[scene_name]
            segments = self.scene_segments[scene_name]
            lo = int(down.min()) + self.forward_offset
            hi = int(down.max())
            keep = np.zeros_like(eligible)
            partners = np.arange(max(lo, 0), min(hi, len(fwd_of) - 1) + 1)
            partners = partners[segments[partners] == segments[int(down[0])]]
            f = fwd_of[partners]
            keep[f[f >= 0]] = True
            eligible &= keep

        # Score every eligible forward frame against the chosen downward frames.
        # Rows of the raw table, so no split mask and no renormalisation: the
        # cross-camera cells are already a fraction of the forward image.
        if not eligible.any():
            return []
        rows = np.stack([np.asarray(covis_raw[int(d), :]) for d in down])
        score = rows.max(0) * eligible
        picked = []
        for _ in range(n_fwd):
            ok = np.flatnonzero(score > self.forward_covis_thres)
            if len(ok) == 0:
                break
            w = score[ok]
            picked.append(int(self._rng.choice(ok, p=w / w.sum())))
            score[picked[-1]] = 0.0  # without replacement
        return picked

    def _sample_temporal_window(self, scene_name, num_views):
        """A window of consecutive frames at a random stride, within one segment.

        Segment-bounded on purpose: a window spanning a reconstruction
        discontinuity contains one frame pair whose relative pose is wrong by
        0.3-0.7 m, and training would take that as truth.
        """
        allowed = self.scene_allowed[scene_name] > 0
        segments = self.scene_segments[scene_name]
        for stride in self._rng.permutation(self.temporal_strides):
            span = (num_views - 1) * int(stride)
            if span >= len(allowed):
                continue
            starts = np.flatnonzero(
                allowed[: len(allowed) - span]
                & (segments[: len(segments) - span] == segments[span:])
            )
            # require every frame in the window to be in-split and in-segment
            ok = [i for i in starts if allowed[i : i + span + 1 : int(stride)].all()]
            if ok:
                start = int(self._rng.choice(ok))
                return list(range(start, start + span + 1, int(stride)))
        return None

    def _load_views(self, scene_name, view_indices, resolution):
        scene_root = self.ROOT / scene_name
        scene_meta = load_data(scene_root / "scene_meta.json", "scene_meta")
        depth_scale = scene_meta["divercam"]["depth_scale_to_metres"]
        file_names = list(scene_meta["frame_names"].keys())
        self.is_metric_scale = self.scene_metric[scene_name]

        views = []
        for view_index in view_indices:
            view_file_name = file_names[view_index]
            # A forward frame has no depth at all in the input-only arm, and
            # load_frame would raise on the missing modality. Its pose and ray
            # directions are still exact -- from the rig calibration and the forward
            # intrinsics -- so it is supervised on those and simply carries an empty
            # depthmap, which losses.py:2352 already drops from pts3d, cam_pts3d,
            # depth and pose_trans for the whole sample.
            frame_meta = scene_meta["frames"][view_index]
            is_fwd = frame_meta["frame_name"].endswith("_fwd")
            has_depth = "depth" in frame_meta
            view_data = load_frame(
                scene_root, view_file_name,
                modalities=["image", "depth"] if has_depth else ["image"],
                scene_meta=scene_meta,
            )

            image = view_data["image"].permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
            if has_depth:
                # EXRs hold raw chunk-internal depth; metres need the scene's factor
                depthmap = view_data["depth"].numpy().astype(np.float32) * depth_scale
                depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                depthmap = np.zeros(image.shape[:2], dtype=np.float32)
            intrinsics = view_data["intrinsics"].numpy().astype(np.float32)
            c2w_pose = view_data["extrinsics"].numpy().astype(np.float32)

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image=image, resolution=resolution, depthmap=depthmap,
                intrinsics=intrinsics,
            )

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=c2w_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    # no sky underwater, and a downward camera sees no water column:
                    # missing depth is missing MVS, not an ambiguous region. Kept as
                    # ones for BOTH cameras so every view carries the same key with
                    # the same shape, which default_collate requires; forward views
                    # are excluded via non_ambiguous_mask_known below instead.
                    non_ambiguous_mask=np.ones(depthmap.shape[:2], dtype=int),
                    # Whether the mask above is a real label. For a forward view it
                    # is not: we do not know which pixels are reconstructible until
                    # the fused cloud is rendered into the frame. A horizon test is
                    # not a substitute -- a ray a few degrees above horizontal still
                    # strikes a coral head taller than the camera within 5 m, so it
                    # mislabels real geometry as unknown as readily as ones mislabels
                    # water as known. NonAmbiguousMaskLoss skips views flagged False
                    # rather than training its head on either guess.
                    non_ambiguous_mask_known=np.array(not is_fwd),
                    dataset="DiverCam",
                    label=scene_name,
                    instance=os.path.join("images", f"{view_file_name}.jpg"),
                )
            )
        return views


def _default(rel):
    """Resolve a path relative to the loggerhead checkout containing this file."""
    return Path(__file__).resolve().parents[5] / rel
