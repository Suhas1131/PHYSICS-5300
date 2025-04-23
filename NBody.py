import numpy as np
from scipy.integrate import solve_ivp

class NBody2D:
    """
    N-body gravitational system in 2D using Lagrangian mechanics.
    """
    def __init__(self, masses, radii, G=1., enable_collisions=False):
        """
        Saves values for the system.
            - masses (array-like): The masses of all of the bodies in the system.
            - radii (array-like): Array of radii of each mass.
            - G (float): Stores value of the gravitational constant.
            - N (int): The number of bodies in the system.
        """
        self.masses = np.array(masses)
        self.radii = np.array(radii)    # for collisions
        self.G = G
        self.N = len(masses)
        self.enable_collisions = enable_collisions

    def dy_dt(self, t, y):
        """
        This function returns the right-hand side of the diffeq: 
        [dx/dt dy/dt d^2x/dt^2 d^2y/dt^2]
        
        Parameters:
            - t (float): time. 
            - y (array-like): A 4N-component array with:
                    y[0] = x1(t), y[1] = x2(t), ...,
                    y[N] = y1(t), y[N+1] = y2(t), ...,
                    y[2N] = dx1/dt, y[2N+1] = dx2/dt, ...,
                    y[3N] = dy1/dt, and y[3N+1] = dy2/dt
        Returns:
            - a list/array?
        """
        # Extract positions and velocities
        pos_x = y[0:self.N]
        pos_y = y[self.N:2*self.N]
        v_x = y[2*self.N:3*self.N]
        v_y = y[3*self.N:4*self.N]

        # Creating arrays to store accelerations
        a_x = np.zeros(self.N)
        a_y = np.zeros(self.N)
        
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    delta_x = pos_x[j] - pos_x[i]
                    delta_y = pos_y[j] - pos_y[i]
                    r_cubed = np.power(delta_x**2 + delta_y**2, 3/2)
                    a_x[i] += self.G * self.masses[j] * delta_x / r_cubed
                    a_y[i] += self.G * self.masses[j] * delta_y / r_cubed

        t_deriv = np.zeros_like(y)    # Creating an array to store velocities and accelerations
        t_deriv[0:self.N] = v_x
        t_deriv[self.N:2*self.N] = v_y
        t_deriv[2*self.N:3*self.N] = a_x
        t_deriv[3*self.N:4*self.N] = a_y
        return t_deriv

    def solve_ode(self, t_pts, y0, method='RK23', abserr=1e-9, relerr=1e-9):
        """
        Uses scipy's solve_ivp to solve an ode using a RK23 integrator.

        Parameters:
            - t_pts (array-like): Array of time steps.
            - y0 (array-like):  Array of initial positions and speeds.
            - method (str): Type of integrator.
            - abserr (float): Absolute error.
            - relerr (float): Relative error.

        Returns:
            - solution: result of solve_ivp
        """
        solution = solve_ivp(self.dy_dt, (t_pts[0], t_pts[-1]), y0,
                        t_eval=t_pts, method=method, atol=abserr, rtol=relerr)
        return solution

    def euler(self, t_pts, y0):
        """
        Solves an ode using Euler method.

        Parameters:
            - t_pts (array-like): Array of time steps.
            - y0 (array-like): Array of initial conditions.

        Returns:
            - y (array-like): positions and velocities of bodies over time steps.
        """
        delta_t = t_pts[1] - t_pts[0]
        y = np.zeros((len(y0), len(t_pts)))
        y[:, 0] = y0
        for i in range(len(t_pts) - 1):
            t_deriv = self.dy_dt(t_pts[i], y[:, i])
            y[:, i+1] = y[:, i] + delta_t * t_deriv    # Update posititon and velocity at each time step
        return y

    def leapfrog(self, t_pts, y0):    # [Not completely sure if this is right but it works well]
        """
        Solves the ode using a leapfrog integrator. 

        Parameters:
            - t_pts (array-like): Array of time steps.
            - y0 (array-like): Array of initial conditions.

        Returns:
            - y (array-like): positions and velocities of bodies over all time steps.
        """
        delta_t = t_pts[1] - t_pts[0]
        y = np.zeros((len(y0), len(t_pts)))
        y[:, 0] = y0

        pos_x = y0[0:self.N].copy()
        pos_y = y0[self.N:2*self.N].copy()
        v_x = y0[2*self.N:3*self.N].copy()
        v_y = y0[3*self.N:4*self.N].copy()

        a_x, a_y = self.compute_a(pos_x, pos_y)

        for i in range(len(t_pts) - 1):

            # half time-step
            v_x += 0.5 * a_x * delta_t
            v_y += 0.5 * a_y * delta_t

            pos_x += v_x * delta_t
            pos_y += v_y * delta_t

            if self.enable_collisions:    # False by default
                pos_x, pos_y, v_x, v_y = self.collisions(pos_x, pos_y, v_x, v_y)

            a_x, a_y = self.compute_a(pos_x, pos_y)

            # second half time step
            v_x += 0.5 * a_x * delta_t
            v_y += 0.5 * a_y * delta_t

            y[0:self.N, i+1] = pos_x
            y[self.N:2*self.N, i+1] = pos_y
            y[2*self.N:3*self.N, i+1] = v_x
            y[3*self.N:4*self.N, i+1] = v_y

        return y

    def compute_a(self, pos_x, pos_y):
        """
        Calculates the accelerations of each body, to be used in the leapfrog integrator.

        Parameters:
            - pos_x (array-like): x positions of all bodies.
            - pos_y (array-like): y positions of all bodies.

        Returns:
            - a_x (array-like): accelerations of all bodies in the x direction.
            - a_y (array-like): accelerations of all bodies in the y direction.
        """
        a_x = np.zeros(self.N)
        a_y = np.zeros(self.N)
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    delta_x = pos_x[j] - pos_x[i]
                    delta_y = pos_y[j] - pos_y[i]
                    r_cubed = np.power(delta_x**2 + delta_y**2, 3/2)
                    # From the Lagrangian defined above:
                    a_x[i] += self.G * self.masses[j] * delta_x / r_cubed
                    a_y[i] += self.G * self.masses[j] * delta_y / r_cubed
        return a_x, a_y

    # I haven't been able to figure out the logic to change the length of N after a collision.
    def collisions(self, pos_x, pos_y, v_x, v_y):
        """
        Accounts for collisions and calculates the positions and velocities of the new masses.

        Parameters:
            - pos_x (array-like): x positions of all bodies.
            - pos_y (array-like): y positions of all bodies.
            - v_x (array-like): velocities of all bodies in the x direction.
            - v_y (array-like): velocities of all bodies in the y direction.

        Returns:
            - pos_x, 
        """
        i = 0
        while i < self.N:
            j = i + 1
            while j < self.N:
                delta_x = pos_x[i] - pos_x[j]
                delta_y = pos_y[i] - pos_y[j]
                dist = np.sqrt(delta_x**2 + delta_y**2)
                if dist < self.radii[i] + self.radii[j]:    # Check for collision
                    m1, m2 = self.masses[i], self.masses[j]
                    new_mass = m1 + m2
                    new_radius = (self.radii[i]**3 + self.radii[j]**3)**(1/3)

                    pos_x[i] = (m1 * pos_x[i] + m2 * pos_x[j]) / new_mass
                    pos_y[i] = (m1 * pos_y[i] + m2 * pos_y[j]) / new_mass
                    v_x[i] = (m1 * v_x[i] + m2 * v_x[j]) / new_mass
                    v_y[i] = (m1 * v_y[i] + m2 * v_y[j]) / new_mass

                    # Delete object j
                    self.masses = np.delete(self.masses, j)
                    self.radii = np.delete(self.radii, j)
                    pos_x = np.delete(pos_x, j)
                    pos_y = np.delete(pos_y, j)
                    v_x = np.delete(v_x, j)
                    v_y = np.delete(v_y, j)

                    # Update object i
                    self.masses[i] = new_mass
                    self.radii[i] = new_radius
                    self.N -= 1
                else:
                    j += 1
            i += 1
        return pos_x, pos_y, v_x, v_y

    def kinetic(self, y):
        """
        Compute kinetic energy of each body.
        
        Parameters:
            - y (array-like): Contains the positions and velocities of all of the 
            bodies in the system.
        
        Returns:
            - Array of kinetic energies for each body.
        """
        v_x = y[2*self.N:3*self.N]
        v_y = y[3*self.N:4*self.N]
        return 0.5 * self.masses * (v_x**2 + v_y**2)

    def potential(self, y):
        """
        Compute the total potential energy of the system.

        Parameters:
            - y (array-like): Contains the psitions and velocities of all of the
            bodies in the system.

        Returns:
            - U (float): Total potential energy of the system.
        """
        pos_x = y[0:self.N]
        pos_y = y[self.N:2*self.N]
        U = 0.0
        for i in range(self.N):
            for j in range(i):
                dx = pos_x[j] - pos_x[i]
                dy = pos_y[j] - pos_y[i]
                r = np.sqrt(dx**2 + dy**2) 
                U -= self.G * self.masses[i] * self.masses[j] / r
        return U

    def energy(self, y_traj, t_pts):
        """
        Compute total energy over all time steps.
        
        Parameters:
            - y_traj (array-like): Contains the positions and velocities of all of the
            bodies in the system at every time step.
            - t_pts (array-like): Array of time steps.

        Returns:
            - E (array-like): The total energy of the system.
        """
        E = np.zeros(len(t_pts))

        # Energy at each time step
        for i in range(len(t_pts)):
            y_i = y_traj[:, i]
            T = self.kinetic(y_i)     
            U = self.potential(y_i) 
            E[i] = np.sum(T) + U
            
        return E
