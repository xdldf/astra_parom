"""Bounded plate evidence from one continuous front-camera passage."""
import threading
from collections import deque


class FrontHistory:
    def __init__(self, seconds=30, max_frames=5, max_bytes=16*1024*1024):
        self.seconds, self.max_frames, self.max_bytes = seconds, max_frames, max_bytes
        self.lock = threading.Lock()
        self.last = None
        self.frames = []
        self.observations = deque(maxlen=512)
        self.passage = 0
        self.capture_delay = 5.

    def observe(self, packet, box, candidates, epoch):
        from web_app.plates import same_vehicle
        with self.lock:
            previous = self.last
            if previous and (previous['epoch'] != epoch or packet.stamp < previous['stamp']):
                self.frames = []
                self.observations.clear()
                previous = None
            if previous and packet.seq == previous['seq'] and packet.stamp == previous['stamp']:
                return
            boundary = (box is None or previous is None or previous['box'] is None
                        or not 0 < packet.stamp-previous['stamp'] <= 1.5
                        or not same_vehicle(box, previous['box']))
            if not boundary and candidates:
                # A different readable number may mean a following vehicle, even
                # when its box overlaps. Archive the previous passage for a
                # capture already in progress; never use it for the new one.
                texts = {c['text'] for c in candidates}
                current = [f for f in self.frames if f['passage'] == self.passage]
                boundary = bool(current) and not texts.intersection(c['text'] for f in current for c in f['candidates'])
            if boundary:
                self.passage += 1
            # Keep geometry/times without retaining a JPEG for every observation.
            self.last = dict(seq=packet.seq, stamp=packet.stamp, box=box, epoch=epoch, passage=self.passage)
            self.observations.append(self.last)
            cutoff = packet.stamp-self.seconds-self.capture_delay
            while self.observations and self.observations[0]['stamp'] < cutoff:
                self.observations.popleft()
            self.frames = [f for f in self.frames if f['packet'].stamp >= cutoff]
            if box is not None and candidates:
                self.frames.append(dict(packet=packet, box=box, candidates=candidates, passage=self.passage))
            groups = {}
            for frame in self.frames:
                groups.setdefault(frame['passage'], []).append(frame)
            # Three recent passages, five best readable frames each, sharing
            # one byte budget. A following cab cannot overwrite pending evidence.
            self.frames = []
            for passage in sorted(groups, reverse=True)[:3]:
                self.frames.extend(sorted(groups[passage], key=lambda f:(self.quality(f),f['packet'].stamp),
                                          reverse=True)[:self.max_frames])
            while sum(len(f['packet'].jpeg) for f in self.frames) > self.max_bytes:
                self.frames.pop()

    @staticmethod
    def quality(frame):
        return max((c.get('confidence', 0), c.get('detection_confidence', 0)) for c in frame['candidates'])

    def select(self, packet, box, epoch):
        from web_app.plates import same_vehicle
        with self.lock:
            last = self.last
            if box is None or last is None or last['epoch'] != epoch:
                return []
            before = next((o for o in reversed(self.observations) if o['stamp'] <= packet.stamp), None)
            after = next((o for o in self.observations if o['stamp'] >= packet.stamp), None)
            if (before is None or before['box'] is None
                    or not 0 <= packet.stamp-before['stamp'] <= 1.5
                    or not same_vehicle(box, before['box'])):
                return []
            if after and (after['passage'] != before['passage'] or after['box'] is None
                          or not same_vehicle(box, after['box'])):
                return []  # An unobserved boundary between two vehicles is ambiguous.
            frames = [f for f in self.frames if f['passage'] == before['passage']
                      and 0 <= packet.stamp-f['packet'].stamp <= self.seconds]
            return sorted(frames, key=self.quality, reverse=True)
