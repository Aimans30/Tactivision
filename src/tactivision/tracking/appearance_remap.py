"""Post-ByteTrack appearance-assisted identity remapping.

ByteTrack (Ultralytics) still owns motion association. This module only remaps
newly appeared track IDs onto recently lost IDs when spatial + appearance gates
pass — targeting short-gap fragmentation / rematch-with-new-ID failures.

Each raw ByteTrack ID is locked after its first assignment so appearance cannot
flip an already-bound ID every gap (that produced TrackEval ID-switch spikes).

Experimental. Not used by the default demo pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tactivision.tracking.appearance import (
    AppearanceParams,
    appearance_similarity,
    appearance_vector,
    bbox_center,
)
from tactivision.tracking.schemas import Track


@dataclass
class _GalleryEntry:
    feature: np.ndarray
    center: tuple[float, float]
    last_frame: int
    active: bool = True


@dataclass
class AppearanceRemapper:
    params: AppearanceParams = field(default_factory=AppearanceParams)
    gallery: dict[int, _GalleryEntry] = field(default_factory=dict)
    # Locked raw ByteTrack ID → canonical output ID (set once, never reopened).
    id_map: dict[int, int] = field(default_factory=dict)
    # Canonical IDs whose original raw owner was continued by a remapped successor.
    # If that raw ID later reappears, it gets a fresh locked identity (once).
    continued_canonicals: set[int] = field(default_factory=set)
    remap_events: list[dict] = field(default_factory=list)
    _next_fresh_id: int = 10_000

    def reset(self) -> None:
        self.gallery.clear()
        self.id_map.clear()
        self.continued_canonicals.clear()
        self.remap_events.clear()
        self._next_fresh_id = 10_000

    def _alloc_fresh(self, claimed: set[int]) -> int:
        while self._next_fresh_id in claimed or self._next_fresh_id in self.gallery:
            self._next_fresh_id += 1
        cid = self._next_fresh_id
        self._next_fresh_id += 1
        return cid

    def apply(
        self,
        frame_bgr: np.ndarray,
        frame_id: int,
        tracks: tuple[Track, ...] | list[Track],
    ) -> tuple[Track, ...]:
        """Return tracks with remapped ``track_id`` values (same boxes/scores).

        Guarantees unique canonical IDs within a frame (required by TrackEval).
        """
        present: list[tuple[int, Track, np.ndarray | None, tuple[float, float]]] = []
        for track in tracks:
            if track.track_id is None:
                continue
            raw_id = int(track.track_id)
            feat = appearance_vector(frame_bgr, track.bbox, self.params)
            center = bbox_center(track.bbox)
            present.append((raw_id, track, feat, center))

        locked: list[tuple[int, Track, np.ndarray | None, tuple[float, float]]] = []
        fresh_raw: list[tuple[int, Track, np.ndarray | None, tuple[float, float]]] = []
        for raw_id, track, feat, center in present:
            if raw_id in self.id_map:
                locked.append((raw_id, track, feat, center))
            else:
                fresh_raw.append((raw_id, track, feat, center))

        claimed: set[int] = set()
        assignments: dict[int, int] = {}

        # Phase 1: locked raw IDs keep their canonical binding.
        for raw_id, track, feat, center in locked:
            cid = self.id_map[raw_id]
            if cid in claimed:
                # Collision only if a prior bug; allocate and relock.
                cid = self._alloc_fresh(claimed)
                self.id_map[raw_id] = cid
            claimed.add(cid)
            assignments[raw_id] = cid

        # Phase 2: brand-new raw IDs — optional 1:1 rematch onto lost gallery.
        candidates: list[tuple[float, int, int]] = []  # sim, raw_id, lost_cid
        for raw_id, track, feat, center in fresh_raw:
            if feat is None:
                continue
            # If this raw ID equals a canonical that was already continued by a
            # remapped successor, do not reclaim that identity via appearance.
            if raw_id in self.continued_canonicals:
                continue
            for lost_cid, sim in self._lost_candidates(feat, center, frame_id, claimed):
                if lost_cid == raw_id:
                    continue  # no-op "remap"
                # Do not stitch onto synthetic IDs allocated for superseded reappearances.
                if lost_cid >= 10_000:
                    continue
                candidates.append((sim, raw_id, lost_cid))
        candidates.sort(key=lambda x: x[0], reverse=True)

        remapped_raw: set[int] = set()
        used_lost: set[int] = set()
        for sim, raw_id, lost_cid in candidates:
            if raw_id in remapped_raw or lost_cid in used_lost or lost_cid in claimed:
                continue
            remapped_raw.add(raw_id)
            used_lost.add(lost_cid)
            claimed.add(lost_cid)
            assignments[raw_id] = lost_cid
            self.id_map[raw_id] = lost_cid
            self.continued_canonicals.add(lost_cid)
            self.remap_events.append(
                {
                    "frame": frame_id,
                    "from_id": raw_id,
                    "to_id": lost_cid,
                    "similarity": round(float(sim), 4),
                }
            )

        # Phase 3: remaining new raw IDs lock to themselves (or fresh if taken).
        for raw_id, track, feat, center in fresh_raw:
            if raw_id in assignments:
                continue
            if raw_id in self.continued_canonicals or raw_id in claimed:
                cid = self._alloc_fresh(claimed)
            else:
                cid = raw_id
            claimed.add(cid)
            assignments[raw_id] = cid
            self.id_map[raw_id] = cid

        remapped: list[Track] = []
        active_canonical: set[int] = set()
        for raw_id, track, feat, center in present:
            cid = assignments[raw_id]
            active_canonical.add(cid)
            if feat is not None:
                self._update_gallery(cid, feat, center, frame_id)
            elif cid in self.gallery:
                entry = self.gallery[cid]
                entry.center = center
                entry.last_frame = frame_id
                entry.active = True
            remapped.append(
                Track(
                    track_id=cid,
                    class_name=track.class_name,
                    confidence=track.confidence,
                    bbox=track.bbox,
                )
            )

        for cid, entry in self.gallery.items():
            entry.active = cid in active_canonical

        for track in tracks:
            if track.track_id is None:
                remapped.append(track)

        ids = [t.track_id for t in remapped if t.track_id is not None]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"duplicate track IDs in frame {frame_id}: {ids}")

        return tuple(remapped)

    def _lost_candidates(
        self,
        feat: np.ndarray,
        center: tuple[float, float],
        frame_id: int,
        claimed: set[int],
    ) -> list[tuple[int, float]]:
        out: list[tuple[int, float]] = []
        for cid, entry in self.gallery.items():
            if entry.active or cid in claimed:
                continue
            gap = frame_id - entry.last_frame
            if gap <= 0 or gap > self.params.max_gap_frames:
                continue
            dx = center[0] - entry.center[0]
            dy = center[1] - entry.center[1]
            dist = float(np.hypot(dx, dy))
            if dist > self.params.max_center_distance_px:
                continue
            sim = appearance_similarity(feat, entry.feature)
            if sim < self.params.appearance_threshold:
                continue
            out.append((cid, sim))
        return out

    def _update_gallery(
        self,
        cid: int,
        feat: np.ndarray,
        center: tuple[float, float],
        frame_id: int,
    ) -> None:
        alpha = self.params.ema_alpha
        if cid not in self.gallery:
            self.gallery[cid] = _GalleryEntry(
                feature=feat.copy(),
                center=center,
                last_frame=frame_id,
                active=True,
            )
            return
        entry = self.gallery[cid]
        blended = (1.0 - alpha) * entry.feature + alpha * feat
        norm = float(np.linalg.norm(blended))
        if norm > 1e-8:
            blended = blended / norm
        entry.feature = blended
        entry.center = center
        entry.last_frame = frame_id
        entry.active = True
