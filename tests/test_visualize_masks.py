import numpy as np
import cv2

from auto_labeling_tool.app.visualize_masks import (
    _find_rgb_image,
    find_aggregated_mask_paths,
    load_mask_stack,
    render_mask_stack,
)


def test_finds_only_aggregated_npy_masks(tmp_path):
    average = tmp_path / "episode-a" / "average_mask"
    version = tmp_path / "episode-a" / "version-a"
    average.mkdir(parents=True)
    version.mkdir(parents=True)
    np.save(average / "frame-2.npy", np.zeros((1, 2, 2), dtype=bool))
    np.save(average / "frame-1.npy", np.zeros((1, 2, 2), dtype=bool))
    np.save(version / "ignored.npy", np.zeros((1, 2, 2), dtype=bool))

    assert [path.name for path in find_aggregated_mask_paths(tmp_path)] == [
        "frame-1.npy",
        "frame-2.npy",
    ]


def test_finds_aggregated_masks_directly_under_root(tmp_path):
    average = tmp_path / "average_mask"
    average.mkdir()
    np.save(average / "frame.npy", np.zeros((1, 2, 2), dtype=bool))

    assert [path.name for path in find_aggregated_mask_paths(tmp_path)] == ["frame.npy"]


def test_loads_2d_mask_as_single_slice_and_renders_colors(tmp_path):
    path = tmp_path / "mask.npy"
    np.save(path, np.array([[0, 1], [1, 0]], dtype=np.uint8))

    stack = load_mask_stack(path)

    assert stack.shape == (1, 2, 2)
    rendered = render_mask_stack(stack)
    assert rendered.shape == (2, 2, 3)
    assert rendered[0, 1].any()
    assert not rendered[0, 0].any()


def test_finds_an_existing_rgb_extension_without_opening_missing_ones(tmp_path, monkeypatch):
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "frame.jpg"), image)
    opened_paths: list[str] = []
    original_imread = cv2.imread

    def record_imread(path, flags):
        opened_paths.append(path)
        return original_imread(path, flags)

    monkeypatch.setattr(
        "auto_labeling_tool.app.visualize_masks.cv2.imread", record_imread
    )

    assert _find_rgb_image(tmp_path, "episode", "frame") is not None
    assert opened_paths == [str(tmp_path / "frame.jpg")]
