# sort.py
# Lightweight IoU-based tracker (greedy matching).
# Drop-in replacement for very basic SORT behavior for counting tasks.

import numpy as np

class Track:
    def __init__(self, bbox, track_id, max_age=30):
        # bbox = [x1,y1,x2,y2]
        self.bbox = bbox
        self.id = track_id
        self.age = 0         # frames since last seen
        self.hits = 1        # total hits
        self.max_age = max_age

    def update(self, bbox):
        self.bbox = bbox
        self.age = 0
        self.hits += 1

    def mark_missed(self):
        self.age += 1
        return self.age > self.max_age

def iou(bb_test, bb_gt):
    # bb = [x1,y1,x2,y2]
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union

class Sort:
    def __init__(self, max_age=30, min_hits=1, iou_threshold=0.3):
        """
        max_age: frames to keep a track alive without matched detection
        min_hits: minimum hits to consider a track confirmed (not used heavily here)
        iou_threshold: IoU threshold for matching detections to existing tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []   # list of Track
        self._next_id = 1

    def update(self, dets=np.empty((0, 5))):
        """
        dets: numpy array of detections [[x1,y1,x2,y2,score], ...]
        returns: numpy array [[x1,y1,x2,y2,track_id], ...]
        """
        if dets is None:
            dets = np.empty((0, 5))
        dets = np.array(dets)
        matched_indices = []
        results = []

        if len(self.tracks) == 0:
            # create tracks for all detections
            for i, d in enumerate(dets):
                bbox = [int(d[0]), int(d[1]), int(d[2]), int(d[3])]
                tr = Track(bbox, self._next_id, max_age=self.max_age)
                self._next_id += 1
                self.tracks.append(tr)

        else:
            # compute IoU between each track and detection
            iou_matrix = np.zeros((len(self.tracks), len(dets)), dtype=np.float32)
            for t_idx, tr in enumerate(self.tracks):
                for d_idx, d in enumerate(dets):
                    iou_matrix[t_idx, d_idx] = iou(tr.bbox, d[:4])

            # Greedy matching: pick highest IoU pairs above threshold
            if iou_matrix.size > 0:
                # flatten indices sorted descending by IoU
                t_idx_flat, d_idx_flat = np.unravel_index(np.argsort(-iou_matrix, axis=None), iou_matrix.shape)
                matched_tracks = set()
                matched_dets = set()
                for ti, di in zip(t_idx_flat, d_idx_flat):
                    if iou_matrix[ti, di] < self.iou_threshold:
                        break
                    if ti in matched_tracks or di in matched_dets:
                        continue
                    # match
                    self.tracks[ti].update([int(dets[di,0]), int(dets[di,1]), int(dets[di,2]), int(dets[di,3])])
                    matched_tracks.add(ti)
                    matched_dets.add(di)
                    matched_indices.append((ti, di))

                # create new tracks for unmatched detections
                for d_idx in range(len(dets)):
                    if d_idx not in matched_dets:
                        d = dets[d_idx]
                        bbox = [int(d[0]), int(d[1]), int(d[2]), int(d[3])]
                        tr = Track(bbox, self._next_id, max_age=self.max_age)
                        self._next_id += 1
                        self.tracks.append(tr)

                # mark unmatched tracks as missed
                for t_idx in range(len(self.tracks)):
                    # if this track was not updated in matched_tracks, increase age
                    if t_idx not in matched_tracks:
                        # Note: careful — newly appended tracks shift indices; we use object property
                        # We'll instead iterate by object and increase age for those not updated this frame.
                        pass

            else:
                # no detections: increase age for all tracks
                for tr in self.tracks:
                    tr.age += 1

        # After matching logic above, we need a robust way to age tracks that were not updated this frame.
        # We'll do a simple approach: for each track, if its bbox was NOT equal to any detection bbox just now,
        # increment age. (This works for our simple tracker.)
        current_bboxes = []
        for d in dets:
            current_bboxes.append([int(d[0]), int(d[1]), int(d[2]), int(d[3])])

        for tr in self.tracks:
            if tr.bbox not in current_bboxes:
                tr.age += 1
            else:
                tr.age = 0  # just seen

        # remove dead tracks
        self.tracks = [tr for tr in self.tracks if tr.age <= self.max_age]

        # prepare results
        for tr in self.tracks:
            x1,y1,x2,y2 = tr.bbox
            results.append([x1, y1, x2, y2, tr.id])

        if len(results) == 0:
            return np.empty((0,5))
        return np.array(results)
