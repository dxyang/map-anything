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
        self._ekf_cache = {}
        self._load_data()

        # overwritten per scene in _get_views; this is the default for a mixed split
        self.is_metric_scale = all(self.scene_metric.values())
        self.is_synthetic = False

    def _load_data(self):
        spec = yaml.safe_load((self.split_dir / f"{self.split_name}.yaml").read_text())
        dives = {d["label"]: d for d in yaml.safe_load(self.manifest.read_text())["dives"]}
        if self.part not in spec:
            raise ValueError(f"split {self.split_name} has no '{self.part}' part")

        self.scenes, self.scene_allowed, self.scene_metric = [], {}, {}
        self.scene_segments, self.scene_dive_root = {}, {}
        for member in spec[self.part]:
            label = member["survey"]
            scene_root = self.ROOT / label
            meta = json.loads((scene_root / "scene_meta.json").read_text())
            segments = np.array([f["segment"] for f in meta["frames"]])
            coverage = np.load(scene_root / "coverage.npy")

            keep = set(member.get("segments", range(segments.max() + 1)))
            keep -= set(member.get("exclude_segments", []))
            allowed = np.isin(segments, list(keep)) & (coverage >= self.min_coverage)
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
            self.scenes.append(label)
            self.scene_allowed[label] = allowed.astype(np.float32)
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
            csv_path = self.scene_dive_root[scene_name] / "left_ekf_sync.csv"
            self._ekf_cache[scene_name] = (
                _load_ekf_camera_poses(csv_path) if csv_path.is_file() else {})
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
                frame = Path(v["instance"]).stem
                T = self._ekf_poses(scene_name).get(frame)
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

        covis_dir = scene_root / "covisibility" / "v0"
        # WAI's mmap loader takes the shape from the filename; one npy per dir
        covis_name = next(f for f in os.listdir(covis_dir) if f.endswith(".npy"))
        covis = load_data(covis_dir / covis_name, "mmap")
        covis = _SplitMaskedCovisibility(covis, self.scene_allowed[scene_name])
        if self.forced_view_indices is not None:
            view_indices = self.forced_view_indices
        elif self.temporal_prob > 0 and self._rng.random() < self.temporal_prob:
            view_indices = self._sample_temporal_window(scene_name, num_views_to_sample)
            if view_indices is None:  # no window fits; fall back
                view_indices = self._sample_view_indices(
                    num_views_to_sample, len(file_names), covis)
        else:
            view_indices = self._sample_view_indices(
                num_views_to_sample, len(file_names), covis)
        return self._load_views(scene_name, view_indices, resolution)

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
            view_data = load_frame(
                scene_root, view_file_name, modalities=["image", "depth"],
                scene_meta=scene_meta,
            )

            image = view_data["image"].permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
            # EXRs hold raw chunk-internal depth; metres need the scene's factor
            depthmap = view_data["depth"].numpy().astype(np.float32) * depth_scale
            depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
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
                    # missing depth is missing MVS, not an ambiguous region
                    non_ambiguous_mask=np.ones(depthmap.shape[:2], dtype=int),
                    dataset="DiverCam",
                    label=scene_name,
                    instance=os.path.join("images", f"{view_file_name}.jpg"),
                )
            )
        return views


def _default(rel):
    """Resolve a path relative to the loggerhead checkout containing this file."""
    return Path(__file__).resolve().parents[5] / rel
