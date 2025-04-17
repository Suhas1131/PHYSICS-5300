import numpy as np

class LagrangianSystem:
    """
    Class for simulating an N-body gravitational system using the Lagrangian formulation.
    Supports Euler, RK4, and Leapfrog integrators.

    Attributes
    ----------
    masses : ndarray
        Array of masses of the particles
    G : float
        Gravitational constant
    """

    def __init__(self, masses, G=1.0):
        self.masses = np.array(masses)
        self.N = len(masses)
        self.G = G

    def kinetic_energy(self, v):
        """Return the total kinetic energy of the system."""
        T = 0.5 * np.sum(self.masses[:, None] * v**2)
        return T

    def potential_energy(self, x):
        """Return the total gravitational potential energy of the system."""
        U = 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                r_ij = np.linalg.norm(x[i] - x[j])
                if r_ij != 0:
                    U -= self.G * self.masses[i] * self.masses[j] / r_ij
        return U

    def lagrangian(self, x, v):
        """Return the Lagrangian L = T - U."""
        return self.kinetic_energy(v) - self.potential_energy(x)

    def acceleration(self, x):
        """Compute acceleration using the gradient of the potential."""
        a = np.zeros_like(x)
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    r_ij = x[j] - x[i]
                    dist = np.linalg.norm(r_ij)
                    if dist != 0:
                        a[i] += self.G * self.masses[j] * r_ij / dist**3
        return a

    def euler_step(self, x, v, dt):
        """Euler method."""
        a = self.acceleration(x)
        x_new = x + dt * v
        v_new = v + dt * a
        return x_new, v_new

    def rk4_step(self, x, v, dt):
        """Fourth-order Runge-Kutta method."""
        a1 = self.acceleration(x)
        k1v = dt * a1
        k1x = dt * v

        a2 = self.acceleration(x + 0.5 * k1x)
        k2v = dt * a2
        k2x = dt * (v + 0.5 * k1v)

        a3 = self.acceleration(x + 0.5 * k2x)
        k3v = dt * a3
        k3x = dt * (v + 0.5 * k2v)

        a4 = self.acceleration(x + k3x)
        k4v = dt * a4
        k4x = dt * (v + k3v)

        x_new = x + (k1x + 2*k2x + 2*k3x + k4x) / 6
        v_new = v + (k1v + 2*k2v + 2*k3v + k4v) / 6

        return x_new, v_new

    def leapfrog_step(self, x, v, dt):
        """Leapfrog (velocity Verlet) integrator."""
        a = self.acceleration(x)
        v_half = v + 0.5 * dt * a
        x_new = x + dt * v_half
        a_new = self.acceleration(x_new)
        v_new = v_half + 0.5 * dt * a_new
        return x_new, v_new
