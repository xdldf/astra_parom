"""Bounded plate evidence from one continuous front-camera passage."""
import threading


class FrontHistory:
    def __init__(self, seconds=12, max_frames=5, max_bytes=16*1024*1024):
        self.seconds, self.max_frames, self.max_bytes = seconds, max_frames, max_bytes
        self.lock = threading.Lock()
        self.last = None
        self.frames = []

    def observe(self, packet, box, candidates, epoch):
        from web_app.plates import same_vehicle
        with self.lock:
            previous = self.last
            if (box is None or previous is None or previous['epoch'] != epoch
                    or not 0 < packet.stamp-previous['packet'].stamp <= 1.5
                    or not same_vehicle(box, previous['box'])):
                self.frames = []
            self.last = dict(packet=packet, box=box, epoch=epoch) if box is not None else None
            self.frames = [f for f in self.frames if packet.stamp-f['packet'].stamp <= self.seconds]
            if box is not None and candidates:
                # A different readable number may mean a following vehicle, even
                # when its box overlaps. Do not mix its evidence with the truck.
                texts = {c['text'] for c in candidates}
                if self.frames and not texts.intersection(c['text'] for f in self.frames for c in f['candidates']):
                    self.frames = []
                self.frames.append(dict(packet=packet, box=box, candidates=candidates))
                self.frames.sort(key=self.quality, reverse=True)
                self.frames = self.frames[:self.max_frames]
                while sum(len(f['packet'].jpeg) for f in self.frames) > self.max_bytes:
                    self.frames.pop()

    @staticmethod
    def quality(frame):
        return max((c.get('confidence', 0), c.get('detection_confidence', 0)) for c in frame['candidates'])

    def select(self, packet, box, epoch):
        from web_app.plates import same_vehicle
        with self.lock:
            last = self.last
            if (box is None or last is None or last['epoch'] != epoch
                    or not 0 <= packet.stamp-last['packet'].stamp <= 1.5
                    or not same_vehicle(box, last['box'])):
                return []
            return [f for f in self.frames if 0 <= packet.stamp-f['packet'].stamp <= self.seconds]
