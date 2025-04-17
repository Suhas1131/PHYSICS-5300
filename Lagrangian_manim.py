from manim import *
import numpy as np

# Include the LagrangianSystem class definition from earlier here or import it from a file
# For brevity, I'm assuming it's already defined in the script or imported

class LagrangianOrbitAnimation(Scene):
    def construct(self):
        # Define system: one massive body (star) and one light body (planet)
        masses = [1000.0, 1.0]  # Star, Planet
        system = LagrangianSystem(masses)

        # Initial positions and velocities (2D)
        x = np.array([[0.0, 0.0], [4.0, 0.0]])  # Star at origin, Planet at x=4
        v = np.array([[0.0, 0.0], [0.0, 1.6]])  # Planet velocity for stable-ish orbit

        dt = 0.05
        steps = 300

        # Store position history
        trajectory = []

        for _ in range(steps):
            x, v = system.leapfrog_step(x, v, dt)
            trajectory.append(x.copy())

        trajectory = np.array(trajectory)
        planet_path = trajectory[:, 1, :]  # Extract planet's positions

        # Set up dots
        star_dot = Dot(point=ORIGIN, color=YELLOW)
        planet_dot = Dot(point=planet_path[0], color=BLUE)
        trail = VMobject(color=BLUE)
        trail.set_points_as_corners([planet_path[0]])

        self.add(star_dot, planet_dot, trail)

        # Animate motion
        for pos in planet_path[1:]:
            self.play(planet_dot.animate.move_to(pos),
                      trail.animate.set_points_as_corners([*trail.get_points(), pos]),
                      run_time=dt, rate_func=linear)
