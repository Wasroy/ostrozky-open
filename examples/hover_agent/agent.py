"""Stub agent: level attitude, hover thrust, no yaw. Useful to check the API plumbing."""


class Agent:
    def reset(self):
        pass

    def act(self, obs):
        return [0.0, 0.0, 0.0, 0.0]
