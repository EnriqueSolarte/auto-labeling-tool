
import numpy as np


def uniform_mask_sampling(mask, num_samples):
    mask_bool = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask_bool)
    num_pixels = xs.size
    if num_pixels == 0 or num_samples <= 0:
        return []
    if num_samples >= num_pixels:
        indices = np.arange(num_pixels)
    else:
        indices = np.random.choice(num_pixels, size=num_samples, replace=False)
    xs_sel = xs[indices]
    ys_sel = ys[indices]
    return [(float(x), float(y)) for x, y in zip(xs_sel, ys_sel)]


def jitter_mask_sampling(mask, base_points, num_samples, noise_std=0.5):
    mask_bool = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask_bool)
    num_pixels = xs.size
    if num_pixels == 0 or num_samples <= 0:
        return []
    desired = min(num_samples, num_pixels)
    height, width = mask_bool.shape
    sampled = []
    sources = list(base_points) if base_points else [(float(x), float(y)) for x, y in zip(xs, ys)]
    if not sources:
        return []
    noise_std = max(float(noise_std), 1.0)
    attempts = 0
    max_attempts = desired * 20
    idx = 0
    while len(sampled) < desired and attempts < max_attempts:
        bx, by = sources[idx % len(sources)]
        idx += 1
        nx = np.random.normal(bx, noise_std)
        ny = np.random.normal(by, noise_std)
        xi = int(round(nx))
        yi = int(round(ny))
        if 0 <= xi < width and 0 <= yi < height and mask_bool[yi, xi]:
            sampled.append((float(xi), float(yi)))
        attempts += 1
    if len(sampled) < desired:
        remaining = desired - len(sampled)
        choose_idx = np.random.choice(num_pixels, size=remaining, replace=False)
        sampled.extend((float(xs[i]), float(ys[i])) for i in choose_idx)
    return sampled
