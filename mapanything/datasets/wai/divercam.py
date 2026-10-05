# DiverCam: underwater coral-reef surveys with Metashape pseudo-GT.
#
# Not a public dataset -- see loggerhead/docs and the manifest at
# third_party/Pi3/datasets/divercam_manifest.yaml for provenance.

"""
DiverCam underwater survey dataset using WAI format data.
"""

import json
import os
from pathlib import Path

import numpy as np
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
        overfit_num_sets=None,
        **kwargs,
    ):
        """
        Args:
            ROOT: Root directory of the WAI-format divercam dataset.
            split: name of a yaml in split_dir, or "train"/"val"/"test" within it.
            split_dir: directory of split specs (default: loggerhead/configs/splits).
            manifest: divercam_manifest.yaml (default: alongside the Pi3 datasets).
            min_coverage: drop frames whose valid-depth fraction is below this.
                          Metashape MVS leaves 40-50% of pixels empty on the worst
                          frames, which carry too little supervision to be worth a
                          view slot.
            overfit_num_sets: if set, truncate to this many sets.
        """
        super().__init__(*args, **kwargs)
        self.ROOT = Path(ROOT)
        self.split_name, self.part = split.rsplit(":", 1) if ":" in split else (split, "train")
        self.split_dir = Path(split_dir) if split_dir else _default("configs/splits")
        self.manifest = Path(manifest) if manifest else _default(
            "third_party/Pi3/datasets/divercam_manifest.yaml")
        self.min_coverage = min_coverage
        # when set, _get_views uses these indices instead of sampling. Evaluation
        # needs deployment-style consecutive windows as well as covisibility sets,
        # and going through the normal path keeps BaseDataset's post-processing
        # (transforms, pts3d, masks) identical between the two.
        self.forced_view_indices = None
        self.overfit_num_sets = overfit_num_sets
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
        for member in spec[self.part]:
            label = member["survey"]
            scene_root = self.ROOT / label
            meta = json.loads((scene_root / "scene_meta.json").read_text())
            segments = np.array([f["segment"] for f in meta["frames"]])
            coverage = np.load(scene_root / "coverage.npy")

            keep = set(member.get("segments", range(segments.max() + 1)))
            keep -= set(member.get("exclude_segments", []))
            allowed = np.isin(segments, list(keep)) & (coverage >= self.min_coverage)
            if not allowed.any():
                continue
            self.scenes.append(label)
            self.scene_allowed[label] = allowed.astype(np.float32)
            # metric supervision needs the dive scaled AND the split to trust it
            self.scene_metric[label] = bool(
                dives[label]["metric"] and member.get("metric_eval", True))

        self.num_of_scenes = len(self.scenes)
        if self.overfit_num_sets is not None:
            self.scenes = self.scenes[: self.overfit_num_sets]
            self.num_of_scenes = len(self.scenes)

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
        view_indices = (
            self.forced_view_indices
            if self.forced_view_indices is not None
            else self._sample_view_indices(num_views_to_sample, len(file_names), covis)
        )
        return self._load_views(scene_name, view_indices, resolution)

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
