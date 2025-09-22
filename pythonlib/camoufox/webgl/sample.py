import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import orjson
from random import choices

from camoufox.pkgman import OS_ARCH_MATRIX

# Get database path relative to this file
DB_PATH = Path(__file__).parent / 'webgl_data.db'
OS_KEYS = tuple(OS_ARCH_MATRIX.keys())
OS_INDEX = {key: idx for idx, key in enumerate(OS_KEYS)}


@lru_cache(maxsize=1)
def _load_webgl_tables() -> Tuple[
    Dict[str, Tuple[Tuple[str, str, str, float], ...]],
    Dict[Tuple[str, str], Tuple[str, Tuple[float, ...]]],
]:
    """Load and cache WebGL rows grouped by OS and indexed by pair."""

    columns = ', '.join(OS_KEYS)
    query = f'SELECT vendor, renderer, data, {columns} FROM webgl_fingerprints'

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()

    per_os: Dict[str, List[Tuple[str, str, str, float]]] = {
        key: [] for key in OS_KEYS
    }
    index: Dict[Tuple[str, str], Tuple[str, Tuple[float, ...]]] = {}

    for vendor, renderer, data, *weights in rows:
        weight_tuple = tuple(float(weight) for weight in weights)
        index[(vendor, renderer)] = (data, weight_tuple)

        for os_key, weight in zip(OS_KEYS, weight_tuple):
            if weight > 0:
                per_os[os_key].append((vendor, renderer, data, float(weight)))

    for os_key, entries in per_os.items():
        entries.sort(key=lambda item: item[3], reverse=True)
        per_os[os_key] = tuple(entries)

    return per_os, index


@lru_cache(maxsize=None)
def _decode_webgl_payload(raw: str) -> Dict[str, str]:
    """Decode a WebGL JSON payload once and reuse it."""

    return orjson.loads(raw)


def sample_webgl(
    os: str, vendor: Optional[str] = None, renderer: Optional[str] = None
) -> Dict[str, str]:
    """
    Sample a random WebGL vendor/renderer combination and its data based on OS probabilities.
    Optionally use a specific vendor/renderer pair.

    Args:
        os: Operating system ('win', 'mac', or 'lin')
        vendor: Optional specific vendor to use
        renderer: Optional specific renderer to use (requires vendor to be set)

    Returns:
        Dict containing WebGL data including vendor, renderer and additional parameters

    Raises:
        ValueError: If invalid OS provided or no data found for OS/vendor/renderer
    """
    # Check that the OS is valid (avoid SQL injection)
    if os not in OS_ARCH_MATRIX:
        raise ValueError(f'Invalid OS: {os}. Must be one of: win, mac, lin')

    per_os, pair_index = _load_webgl_tables()

    if vendor and renderer:
        entry = pair_index.get((vendor, renderer))
        if not entry:
            raise ValueError(
                f'No WebGL data found for vendor "{vendor}" and renderer "{renderer}"'
            )

        data_blob, weights = entry
        os_weight = weights[OS_INDEX[os]]
        if os_weight <= 0:
            possible_pairs = [
                (pair_vendor, pair_renderer)
                for pair_vendor, pair_renderer, _, _ in per_os.get(os, ())
            ]
            raise ValueError(
                f'Vendor "{vendor}" and renderer "{renderer}" combination not valid for {os.title()}.\n'
                f'Possible pairs: {", ".join(str(pair) for pair in possible_pairs)}'
            )

        return dict(_decode_webgl_payload(data_blob))

    entries = per_os.get(os)
    if not entries:
        raise ValueError(f'No WebGL data found for OS: {os}')

    weights = [entry[3] for entry in entries]
    selected = choices(entries, weights=weights, k=1)[0]

    return dict(_decode_webgl_payload(selected[2]))


def get_possible_pairs() -> Dict[str, List[Tuple[str, str]]]:
    """
    Get all possible (vendor, renderer) pairs for all OS, where the probability is greater than 0.
    """
    per_os, _ = _load_webgl_tables()
    return {
        os_type: [(vendor, renderer) for vendor, renderer, _, _ in entries]
        for os_type, entries in per_os.items()
    }
