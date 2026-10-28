"""
Duplicate / near-duplicate image detection with a perceptual hash (pHash).

Public MRI datasets often contain the same scan more than once (re-saved,
resized or re-encoded). If copies end up in both the training and the test
set, the test score is inflated. Removing them BEFORE the split prevents it.

How it works:
  1) Each image is converted to grayscale, resized to 32x32 and transformed
     with a 2-D DCT. The 8x8 lowest frequencies are compared with their
     median, giving a 64-bit hash that is robust to resizing, compression
     and small brightness changes.
  2) Two images whose hashes differ in at most `max_distance` bits are
     treated as copies. With `match_flips`, an image is also compared in
     its left-right mirrored, top-bottom flipped and 180-degree rotated
     orientations, so flipped / rotated copies are caught too (Br35H has
     hundreds of them; 90-degree rotations are practically absent).
  3) Images are visited in order. An image that is a copy of an already
     kept image (its "representative") is removed; otherwise it is kept and
     becomes a new representative. Every removed image is therefore directly
     close to its representative (no transitive chaining, which would merge
     different scans through a series of small steps).
  4) A hash match between images with DIFFERENT labels is double-checked on
     the pixels (correlation of 32x32 grayscale versions, best orientation).
     Only a near-identical pair (>= `conflict_min_corr`) counts as the same
     scan; such a group is removed entirely, because its true label is
     ambiguous. Otherwise the match is treated as a coincidence. With two
     datasets (~6000 images, 4 orientations) such coincidences happen: all 8
     "conflicts" found without this check were different scans (correlation
     0.60-0.91), while genuine copies are near 1.0. Same-label matches are not
     double-checked: genuine copies with strong contrast changes can correlate
     as low as ~0.6, and missing them would leak copies into the test set.
"""

import numpy as np
from PIL import Image, ImageOps

HASH_SIZE = 8      # 8x8 low-frequency block -> 64-bit hash
DCT_SIZE  = 32     # image is resized to 32x32 before the DCT


def _dct_matrix(n):
    """Orthonormal DCT-II matrix, so that dct(x) = D @ x @ D.T."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    d = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    d[0] /= np.sqrt(2.0)
    return d


_DCT = _dct_matrix(DCT_SIZE)


def _phash_image(gray):
    """64-bit perceptual hash of a grayscale PIL image, as np.uint64."""
    small = gray.resize((DCT_SIZE, DCT_SIZE), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype="float64")
    low = (_DCT @ pixels @ _DCT.T)[:HASH_SIZE, :HASH_SIZE]
    bits = (low > np.median(low)).flatten()
    return np.packbits(bits).view(">u8")[0].astype(np.uint64)


# Orientations compared when match_flips is on (the image itself first).
ORIENTATIONS = (
    lambda g: g,                          # original
    ImageOps.mirror,                      # left-right
    ImageOps.flip,                        # top-bottom
    lambda g: g.rotate(180),              # 180-degree rotation
)


def phash(path):
    """Return the hashes of a file in all ORIENTATIONS (original first)."""
    with Image.open(path) as img:
        gray = img.convert("L")
    return [_phash_image(orient(gray)) for orient in ORIENTATIONS]


def _unit_pixels(gray):
    """32x32 grayscale pixels, zero-mean and unit-norm (for correlation)."""
    a = np.asarray(gray.resize((DCT_SIZE, DCT_SIZE), Image.Resampling.LANCZOS),
                   dtype="float64").ravel()
    a -= a.mean()
    return a / (np.linalg.norm(a) + 1e-9)


def pixel_similarity(path_a, path_b):
    """Pixel correlation of two images, maximised over ORIENTATIONS of b
    (1.0 = identical up to brightness / contrast / size / orientation)."""
    with Image.open(path_a) as a, Image.open(path_b) as b:
        va = _unit_pixels(a.convert("L"))
        gray_b = b.convert("L")
        return max(float(va @ _unit_pixels(orient(gray_b))) for orient in ORIENTATIONS)


def find_duplicates(paths, labels, max_distance, match_flips=True,
                    conflict_min_corr=0.95):
    """
    Find duplicate / near-duplicate images.

    Returns (keep, duplicate_groups, conflict_groups, unreadable):
      keep              - boolean mask, True for images to keep
      duplicate_groups  - lists of indices (same label); the first one is
                          the representative and is kept
      conflict_groups   - lists of indices with mixed labels; all removed
      unreadable        - indices of files that could not be decoded
    """
    n = len(paths)
    keep = np.ones(n, dtype=bool)
    unreadable = []
    reps, rep_hashes, groups = [], [], {}

    for i, path in enumerate(paths):
        try:
            hashes = phash(path)
        except OSError:
            unreadable.append(i)
            keep[i] = False
            continue
        h = hashes[0]

        if reps:
            known = np.array(rep_hashes, dtype=np.uint64)
            variants = hashes if match_flips else hashes[:1]
            dist = np.min([np.bitwise_count(v ^ known) for v in variants], axis=0)
            # Candidates within the distance, nearest first. A candidate with
            # another label must also pass the pixel check (see step 4).
            for c in np.flatnonzero(dist <= max_distance)[np.argsort(dist[dist <= max_distance], kind="stable")]:
                rep = reps[c]
                if (labels[rep] == labels[i]
                        or pixel_similarity(paths[rep], path) >= conflict_min_corr):
                    groups[rep].append(i)
                    keep[i] = False
                    break
            if not keep[i]:
                continue

        reps.append(i)
        rep_hashes.append(h)
        groups[i] = [i]

    duplicate_groups, conflict_groups = [], []
    for members in groups.values():
        if len(members) < 2:
            continue
        if len(set(labels[members])) > 1:
            keep[members] = False
            conflict_groups.append(members)
        else:
            duplicate_groups.append(members)

    return keep, duplicate_groups, conflict_groups, unreadable
