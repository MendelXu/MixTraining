import mmcv
import numpy as np
import cv2
from PIL import Image, ImageOps
from mmcv.utils import build_from_cfg
from mmdet.datasets import PIPELINES
from mmdet.core.mask.structures import PolygonMasks, BitmapMasks
from copy import deepcopy

cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


def random_negative(value, random_negative_prob):
    """Randomly negate value based on random_negative_prob."""
    return -value if np.random.rand() < random_negative_prob else value


def random_factor(max_level, max_value, random_negative_prob, enhance=False):
    level = np.random.randint(1, max_level)
    value = level / max_level * max_value
    value = random_negative(value, random_negative_prob)
    if enhance:
        return value * 1.8 + 0.1
    return value


def _pair(obj):
    if obj is None:
        return obj
    elif isinstance(obj, (list, tuple)):
        assert len(obj) == 2
        return obj
    elif isinstance(obj, (int, float)):
        return (0, obj)


def bbox2fields():
    """The key correspondence from bboxes to labels, masks and
    segmentations."""
    bbox2label = {"gt_bboxes": "gt_labels", "gt_bboxes_ignore": "gt_labels_ignore"}
    bbox2mask = {"gt_bboxes": "gt_masks", "gt_bboxes_ignore": "gt_masks_ignore"}
    bbox2seg = {
        "gt_bboxes": "gt_semantic_seg",
    }
    return bbox2label, bbox2mask, bbox2seg


@PIPELINES.register_module()
class LazyGeoIdentity(object):
    def __call__(self, results):
        if "trans_matrix" not in results:
            results["trans_matrix"] = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        return results


@PIPELINES.register_module()
class Identity(object):
    def __call__(self, results):
        return results


@PIPELINES.register_module()
class LazyTranslate(object):
    def __init__(
        self,
        direction="horizontal",
        max_translate_offset=0.2,
        max_level=10,
        random_negative_prob=0.5,
    ):
        self.direction = direction
        self.max_translate_offset = max_translate_offset
        self.max_level = max_level
        self.random_negative_prob = random_negative_prob

    def __call__(self, results):
        offset = random_factor(
            self.max_level, self.max_translate_offset, self.random_negative_prob
        )
        dx, dy = 0, 0
        if self.direction == "horizontal":
            dx = offset * results["img_shape"][1]
        else:
            dy = offset * results["img_shape"][0]
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

        if "trans_matrix" not in results:
            results["trans_matrix"] = translation_matrix
        else:
            results["trans_matrix"] = translation_matrix @ results["trans_matrix"]
        return results


@PIPELINES.register_module()
class LazyRotate(object):
    def __init__(
        self, max_level=10, max_rotate_angle=30, random_negative_prob=0.5, center=None
    ):
        self.max_level = max_level
        self.max_rotate_angle = max_rotate_angle
        self.random_negative_prob = random_negative_prob
        self.center = center

    def __call__(self, results):

        assert (
            "img_shape" in results
        ), "You are trying to rotate the image, please provide the shape of the image. Maybe you can compute it before running."
        angle = random_factor(
            self.max_level, self.max_rotate_angle, self.random_negative_prob
        )
        h, w = results["img_shape"][:2]
        center = self.center
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        rotation_matrix = self.get_rotation_matrix(center, -angle)
        if "trans_matrix" not in results:
            results["trans_matrix"] = rotation_matrix
        else:
            results["trans_matrix"] = rotation_matrix @ results["trans_matrix"]
        return results

    def get_rotation_matrix(self, center, angle):
        radian = angle / 180 * np.pi
        alpha = np.cos(radian)
        beta = np.sin(radian)
        return np.float32(
            [
                [alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
                [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]],
                [0, 0, 1],
            ]
        )


@PIPELINES.register_module()
class LazyShear(object):
    def __init__(
        self,
        direction="horizontal",
        max_shear_magnitude=0.3,
        max_level=10,
        random_negative_prob=0.5,
    ):
        self.direction = direction
        self.max_shear_magnitude = max_shear_magnitude
        self.max_level = max_level
        self.random_negative_prob = random_negative_prob

    def __call__(self, results):
        magnitude = random_factor(
            self.max_level, self.max_shear_magnitude, self.random_negative_prob
        )

        shear_matrix = self.get_shear_matrix(magnitude)
        if "trans_matrix" not in results:
            results["trans_matrix"] = shear_matrix
        else:
            results["trans_matrix"] = shear_matrix @ results["trans_matrix"]
        return results

    def get_shear_matrix(self, magnitude):
        if self.direction == "horizontal":
            shear_matrix = np.float32([[1, magnitude, 0], [0, 1, 0], [0, 0, 1]])
        else:
            shear_matrix = np.float32([[1, 0, 0], [magnitude, 1, 0], [0, 0, 1]])
        return shear_matrix


@PIPELINES.register_module()
class LazyRandomFlip(object):
    def __init__(self, prob=0.5, direction="horizontal"):
        self.prob = prob
        self.direction = direction

    def __call__(self, results):

        if np.random.random() < self.prob:
            h, w = results["img_shape"][:2]
            if self.direction == "horizontal":
                flip_matrix = np.float32([[-1, 0, w], [0, 1, 0], [0, 0, 1]])
            else:
                flip_matrix = np.float32([[1, 0, 0], [0, h - 1, 0], [0, 0, 1]])
            if "trans_matrix" not in results:
                results["trans_matrix"] = flip_matrix
            else:
                results["trans_matrix"] = flip_matrix @ results["trans_matrix"]
            results["flip"] = True

        else:
            results["flip"] = False

        results["flip_direction"] = self.direction
        return results


@PIPELINES.register_module()
class LazyResize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(
        self,
        img_scale=None,
        multiscale_mode="range",
        ratio_range=None,
        keep_ratio=True,
        override=False,
    ):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ["value", "range"]

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long), max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short), max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range
            )
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == "range":
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == "value":
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results["scale"] = scale
        results["scale_idx"] = scale_idx

    def _lazy_resize_img(self, results):
        """Resize images with ``results['scale']``."""
        h, w = results["img_shape"][:2]
        if self.keep_ratio:
            new_size, scale_factor = mmcv.rescale_size(
                (w, h), results["scale"], return_scale=True
            )
            new_w, new_h = new_size
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            new_w, new_h = results["scale"]
            w_scale = new_w / w
            h_scale = new_h / h
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        results["img_shape"] = np.array(
            [new_h, new_w, results["img_shape"][-1]], dtype=np.int32
        )
        results["scale_factor"] = scale_factor
        results["keep_ratio"] = self.keep_ratio

        scale_matrix = np.float32([[w_scale, 0, 0], [0, h_scale, 0], [0, 0, 1]])

        if "trans_matrix" not in results:
            results["trans_matrix"] = scale_matrix
        else:
            results["trans_matrix"] = scale_matrix @ results["trans_matrix"]

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if "scale" not in results:
            if "scale_factor" in results:
                img_shape = results["img"].shape[:2]
                scale_factor = results["scale_factor"]
                assert isinstance(scale_factor, float)
                results["scale"] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1]
                )
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert (
                    "scale_factor" not in results
                ), "scale and scale_factor cannot be both set."
            else:
                results.pop("scale")
                if "scale_factor" in results:
                    results.pop("scale_factor")
                self._random_scale(results)

        self._lazy_resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(img_scale={self.img_scale}, "
        repr_str += f"multiscale_mode={self.multiscale_mode}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        repr_str += f"keep_ratio={self.keep_ratio})"
        return repr_str


@PIPELINES.register_module()
class LazyRandomCrop(object):
    """Random crop the image & bboxes & masks.
    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.
    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        crop_type (str, optional): one of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])]. Default "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(
        self,
        crop_size,
        crop_type="absolute",
        allow_negative_crop=False,
        bbox_clip_border=True,
    ):
        if crop_type not in [
            "relative_range",
            "relative",
            "absolute",
            "absolute_range",
        ]:
            raise ValueError(f"Invalid crop_type {crop_type}.")
        if crop_type in ["absolute", "absolute_range"]:
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(crop_size[1], int)
        else:
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            "gt_bboxes": "gt_labels",
            "gt_bboxes_ignore": "gt_labels_ignore",
        }
        self.bbox2mask = {
            "gt_bboxes": "gt_masks",
            "gt_bboxes_ignore": "gt_masks_ignore",
        }

    def _crop_data(self, results, crop_size):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get("img_fields", ["img"]):
            h, w = results["img_shape"][:2]
            margin_h = max(h - crop_size[0], 0)
            margin_w = max(w - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
            perspective_trans = cv2.getPerspectiveTransform(
                np.array(
                    [
                        [crop_x1, crop_y1],
                        [crop_x2, crop_y1],
                        [crop_x2, crop_y2],
                        [crop_x1, crop_y2],
                    ],
                    dtype=np.float32,
                ),
                np.array(
                    [
                        [0, 0],
                        [crop_size[1], 0],
                        [crop_size[1], crop_size[0]],
                        [0, crop_size[0]],
                    ],
                    dtype=np.float32,
                ),
            )
            img_shape = [crop_size[0], crop_size[1], 3]
            break
        results["img_shape"] = img_shape
        if "trans_matrix" not in results:
            results["trans_matrix"] = perspective_trans
        else:
            results["trans_matrix"] = perspective_trans @ results["trans_matrix"]
        return results

    def _get_crop_size(self, image_size):
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.
        Args:
            image_size (tuple): (h, w).
        Returns:
            crop_size (tuple): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            crop_h = np.random.randint(
                min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1
            )
            crop_w = np.random.randint(
                min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1
            )
            return crop_h, crop_w
        elif self.crop_type == "relative":
            crop_h, crop_w = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        image_size = results["img_shape"][:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(crop_size={self.crop_size}, "
        repr_str += f"crop_type={self.crop_type}, "
        repr_str += f"allow_negative_crop={self.allow_negative_crop}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border})"
        return repr_str


@PIPELINES.register_module()
class RandomApply(object):
    def __init__(self, policies, prob=1.0):
        if not isinstance(policies, list):
            policies = [policies]
        self.prob = prob
        self.policies = [build_from_cfg(p, PIPELINES) for p in policies]

    def __call__(self, results):
        if np.random.random() < self.prob:
            policy = np.random.choice(self.policies, 1)[0]
            results = policy(results)
        return results


@PIPELINES.register_module()
class MultiBranch(object):
    def __init__(self, policies, tags=None):
        if not isinstance(policies, list):
            policies = [policies]
        self.policies = [build_from_cfg(p, PIPELINES) for p in policies]
        self.tags = tags
        if self.tags is None:
            self.tags = [None for _ in policies]

    def __call__(self, results):
        multi_result = []
        for policy, tag in zip(self.policies, self.tags):
            res = policy(deepcopy(results))
            if res is None:
                return None
            if tag is not None:
                res["tag"] = tag
            multi_result.append(res)
        return multi_result


@PIPELINES.register_module()
class TransformAnnotation(object):
    def __init__(self, mask_fill_val=0, seg_fill_val=255, interpolation="bilinear"):
        self.mask_fill_val = mask_fill_val
        self.seg_fill_val = seg_fill_val
        self.interpolation = interpolation

    def __call__(self, results):
        trans_matrix = results["trans_matrix"]
        h, w = results["img_shape"][:2]
        for key in results.get("bbox_fields", []):
            results[key] = self._transform_bbox(results[key], trans_matrix, (h, w),)
        for key in results.get("mask_fields", []):
            results[key] = self._transform_mask(results[key], trans_matrix, (h, w))
        for key in results.get("seg_fields", []):
            results[key] = cv2.warpAffine(
                results[key],
                trans_matrix,
                (w, h),
                borderValue=self.seg_fill_val,
                flags=self.seg_fill_val,
            )
        self._filter_invalid(results)
        return results

    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        def box2points(box):
            min_x, min_y, max_x, max_y = (
                box[:, 0],
                box[:, 1],
                box[:, 2],
                box[:, 3],
            )
            cx = min_x * 0.5 + max_x * 0.5
            cy = min_y * 0.5 + max_y * 0.5
            return np.stack(
                [cx, min_y, max_x, cy, cx, max_y, min_x, cy], axis=1
            ).reshape(
                -1, 2
            )  # n*4,2

        def points2box(point):
            point = point.reshape((-1, 4, 2))
            if point.shape[0] > 0:
                # print(src_mapper.data,point)
                min_xy = point.min(axis=1)
                max_xy = point.max(axis=1)
                xmin = min_xy[:, 0].clip(min=0, max=max_shape[-1])
                ymin = min_xy[:, 1].clip(min=0, max=max_shape[-2])
                xmax = max_xy[:, 0].clip(min=0, max=max_shape[-1])
                ymax = max_xy[:, 1].clip(min=0, max=max_shape[-2])
                min_xy = np.stack([xmin, ymin], axis=1)
                max_xy = np.stack([xmax, ymax], axis=1)
                return np.concatenate([min_xy, max_xy], axis=1)  # n,4
            else:
                return np.zeros((0, 4))

        points = box2points(bboxes)  # n,2
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)  # n,3
        points = np.dot(trans_mat, points.T).T
        points = points[:, :2] / points[:, 2:3]

        return points2box(points).astype(np.float32)

    def _transform_mask(self, mask, trans_matrix, max_shape):
        if isinstance(mask, BitmapMasks):
            mask_trans = mask.masks.transpose((1, 2, 0))
            cv2.warpAffine(
                mask_trans,
                trans_matrix[:2],
                max_shape[::-1],
                borderValue=[self.mask_fill_val for _ in range(3)],
                flags=cv2_interp_codes[self.interpolation],
            )
            return BitmapMasks(mask_trans, *max_shape)
        elif isinstance(mask, PolygonMasks):
            trans_mask = []
            for poly_per_obj in mask.masks:
                trans_poly = []
                for p in poly_per_obj:
                    p = np.stack([p[0::2], p[1::2]], axis=0)  # [2, n]
                    new_coords = np.matmul(trans_matrix, p)  # [2, n]
                    new_coords[0, :] = np.clip(new_coords[0, :], 0, max_shape[1])
                    new_coords[1, :] = np.clip(new_coords[1, :], 0, max_shape[0])
                    trans_poly.append(new_coords.transpose((1, 0)).reshape(-1))
                trans_mask.append(trans_poly)
            return PolygonMasks(trans_mask, *max_shape)
        else:
            raise NotImplementedError()

    def _filter_invalid(self, results, min_bbox_size=0):
        """Filter bboxes and corresponding masks too small after shear
        augmentation."""
        bbox2label, bbox2mask, _ = bbox2fields()
        for key in results.get("bbox_fields", []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_inds = (bbox_w > min_bbox_size) & (bbox_h > min_bbox_size)
            valid_inds = np.nonzero(valid_inds)[0]
            results[key] = results[key][valid_inds]
            results["bbox_preserve"] = valid_inds
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_inds]


@PIPELINES.register_module()
class TransformImage(object):
    def __init__(self, img_fill_val=(0, 0, 0), mode="bilinear"):
        self.interp_flags = cv2_interp_codes[mode]
        self.img_fill_val = img_fill_val

    def __call__(self, results):
        h, w = results["img_shape"][:2]
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            img_trans = cv2.warpAffine(
                img,
                results["trans_matrix"][:2],
                (w, h),
                borderValue=self.img_fill_val,
                flags=self.interp_flags,
            )
            results[key] = img_trans.astype(img.dtype)
        return results


@PIPELINES.register_module()
class PosterizeV1(object):
    def __init__(self, max_level=10):
        self.max_level = max_level

    def _adjust_posterize(self, results, bits=0):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            results[key] = mmcv.posterize(img, bits)

    def __call__(self, results):
        """Call function for Posterize transformation.
        Args:
            results (dict): Results dict from loading pipeline.
        Returns:
            dict: Results after the transformation.
        """
        factor = int(4 - random_factor(self.max_level, 4, 0))
        self._adjust_posterize(results, factor)
        return results


@PIPELINES.register_module()
class RandomGrayScale(object):
    def __init__(self, max_level=10):
        self.max_level = max_level

    def _adjust_color_img(self, results, factor=1.0):
        """Apply Color transformation to image."""
        for key in results.get("img_fields", ["img"]):
            # NOTE defaultly the image should be BGR format
            img = results[key]
            results[key] = mmcv.adjust_color(img, factor).astype(img.dtype)

    def __call__(self, results):
        factor = random_factor(self.max_level, 1.0, 0, enhance=True)
        self._adjust_color_img(results, factor)
        return results


def adjust_hue(img, factor):
    gradient = cv2.Laplacian(img, cv2.CV_64F)
    return cv2.addWeighted(
        gradient.astype(np.float32), factor, img.astype(np.float32), 1 - factor, 0,
    )


@PIPELINES.register_module()
class Jitter(object):
    def __init__(self, contrast=None, brightness=None, hue=None, max_level=10):
        assert not (
            (contrast is None) and (brightness is None) and (hue is None)
        ), "set at least one attribute to change"
        self.contrast = _pair(contrast)
        self.brightness = _pair(brightness)
        self.hue = _pair(hue)
        self.max_level = max_level

    def _adjust_color_img(self, results, contrast, brightness, hue):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            if contrast is not None:
                img = mmcv.adjust_contrast(img, contrast)
            if brightness is not None:
                img = mmcv.adjust_brightness(img, brightness)
            if hue is not None:
                img = adjust_hue(img, hue)

    def __call__(self, results):
        contrast, brightness, hue = None, None, None
        if self.contrast is not None:
            contrast = (
                random_factor(
                    self.max_level, self.contrast[1] - self.contrast[0], 0, enhance=True
                )
                + self.contrast[0]
            )
        if self.brightness is not None:
            brightness = (
                random_factor(
                    self.max_level,
                    self.brightness[1] - self.brightness[0],
                    0,
                    enhance=True,
                )
                + self.brightness[0]
            )
        if self.hue is not None:
            hue = (
                random_factor(
                    self.max_level, self.hue[1] - self.hue[0], 0, enhance=True
                )
                + self.hue[0]
            )
        self._adjust_color_img(results, contrast, brightness, hue)
        return results


@PIPELINES.register_module()
class SolarizeV1(object):
    def __init__(self, max_level=10):
        self.max_level = max_level

    def _adjust_solarize(self, results, thr=128):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            results[key] = mmcv.solarize(img, thr)

    def __call__(self, results):
        """Call function for Posterize transformation.
        Args:
            results (dict): Results dict from loading pipeline.
        Returns:
            dict: Results after the transformation.
        """
        factor = 256 - random_factor(self.max_level, 256, 0.0)
        self._adjust_solarize(results, factor)
        return results


@PIPELINES.register_module()
class AutoContrast(object):
    def _auto_contrast(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            results[key] = np.asarray(
                ImageOps.autocontrast(Image.fromarray(img)), dtype=img.dtype
            )

    def __call__(self, results):
        self._auto_contrast(results)
        return results


@PIPELINES.register_module()
class Equalize(object):
    """Apply Equalize transformation to image. The bboxes, masks and
    segmentations are not modified.
    """

    def _imequalize(self, results):
        """Equalizes the histogram of one image."""
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            results[key] = mmcv.imequalize(img).astype(img.dtype)

    def __call__(self, results):
        """Call function for Equalize transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        self._imequalize(results)
        return results
