import numpy as np


class Quaternion:
    """Minimal quaternion class mirroring MATLAB's quaternion behavior."""

    def __init__(self, w=0.0, x=0.0, y=0.0, z=0.0):
        self.q = np.array([w, x, y, z], dtype=float)

    @classmethod
    def from_vector(cls, vec):
        """Create a quaternion with zero scalar part from a 3-vector."""
        v = np.asarray(vec, dtype=float)
        return cls(0.0, *v)

    @classmethod
    def from_axis_angle(cls, axis, angle):
        """Construct a unit quaternion from rotation axis and angle."""
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        half = angle / 2.0
        w = np.cos(half)
        xyz = axis * np.sin(half)
        return cls(w, *xyz)

    def conjugate(self):
        """Return the quaternion conjugate."""
        w, x, y, z = self.q
        return Quaternion(w, -x, -y, -z)

    def normalize(self):
        """Return a normalized quaternion."""
        n = np.linalg.norm(self.q)
        if n == 0:
            return Quaternion()
        return Quaternion(*(self.q / n))

    def to_rotation_matrix(self):
        """Convert the quaternion to a 3x3 rotation matrix."""
        w, x, y, z = self.normalize().q
        return np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ])

    def rotate_vector(self, vec):
        """Rotate a 3-vector using the quaternion."""
        vq = Quaternion.from_vector(vec)
        rq = self * vq * self.conjugate()
        return rq.q[1:]

    def __mul__(self, other):
        w1, x1, y1, z1 = self.q
        w2, x2, y2, z2 = other.q
        return Quaternion(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def __repr__(self):
        w, x, y, z = self.q
        return f"Quaternion({w}, {x}, {y}, {z})"
