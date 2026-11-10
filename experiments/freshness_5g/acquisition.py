"""Acquire exactly one control report per window at the selected boundary."""
import json


class Acquisition:
    def __init__(self, control, mode):
        if mode not in ('early', 'late'):
            raise ValueError(mode)
        self.control = control
        self.mode = mode
        self.pending = None
        self.observation = None

    def _capture(self):
        window, scenario = self.pending
        observation = self.control.observe(window, scenario)
        self.control.out.write(json.dumps(observation, sort_keys=True) + '\n')
        self.control.out.flush()
        self.observation = observation

    def at_slot(self, window, scenario):
        if self.pending is not None:
            raise RuntimeError('Previous control window has not been evaluated')
        self.pending = (window, scenario)
        if self.mode == 'early':
            self._capture()

    def at_decision(self, window):
        if self.pending is None or self.pending[0] != window:
            raise RuntimeError('Control observation belongs to a different window')
        if self.mode == 'late':
            self._capture()
        observation = self.observation
        self.pending = self.observation = None
        return observation
