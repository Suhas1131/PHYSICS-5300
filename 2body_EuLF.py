'''
This  simulation shows us the difference between energy conservation for the Euler and 
Lagrange methods, I also used it as a template for the other simulations.
'''

# Run using : 
#         manim -pql 2body_EuLF.py TwoBody_EuLF

from manim import *
import numpy as np
from NBody import NBody2D

# Prevent Manim from auto-opening rendered media, otherwise displays error
config.preview = False
config.show_in_file_browser = False

class TwoBody_EuLF(Scene):
    def construct(self):

        N = 2
        masses = [1.0] * N
        radii = [0.05] * N
        
        t_start = 0.0
        t_end = 30.0    # simulated seconds
        duration = t_end - t_start     # actual playback duration
        delta_t = 1 / config.frame_rate
        t_pts = np.arange(t_start, t_end, delta_t)
        t_pts = np.append(t_pts, t_end)
        print("t_pts shape =", t_pts.shape)

        # Initial conditions
        y0 = np.array([-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, -0.5])

        # Trajectories
        system = NBody2D(masses, radii, G=1.0)
        sol_EU = system.euler(t_pts, y0)
        sol_LF = system.leapfrog(t_pts, y0)

        # Create dots
        dot_EU_1 = Dot(radius=radii[0], color=BLUE)
        dot_EU_2 = Dot(radius=radii[1], color=BLUE)
        dot_LF_1 = Dot(radius=radii[0], color=RED)
        dot_LF_2 = Dot(radius=radii[1], color=RED)

        # Set initial positions
        dot_EU_1.set_x(sol_EU[0, 0]).set_y(sol_EU[0 + N, 0])
        dot_EU_2.set_x(sol_EU[1, 0]).set_y(sol_EU[1 + N, 0])
        dot_LF_1.set_x(sol_LF[0, 0]).set_y(sol_LF[0 + N, 0])
        dot_LF_2.set_x(sol_LF[1, 0]).set_y(sol_LF[1 + N, 0])

        # Create tracers
        trace_EU_1 = TracedPath(dot_EU_1.get_center, stroke_color=BLUE, stroke_width=1)
        trace_EU_2 = TracedPath(dot_EU_2.get_center, stroke_color=BLUE, stroke_width=1)
        trace_LF_1 = TracedPath(dot_LF_1.get_center, stroke_color=RED, stroke_width=1)
        trace_LF_2 = TracedPath(dot_LF_2.get_center, stroke_color=RED, stroke_width=1)

        # Add tracers and bodies to the simulation
        self.add(trace_EU_1, trace_EU_2, trace_LF_1, trace_LF_2,
                 dot_EU_1, dot_EU_2, dot_LF_1, dot_LF_2)
        self.wait()

        # Store time index
        time_tracker = ValueTracker(0)

        # Attach updaters to each dot
        dot_EU_1.add_updater(lambda m: m.set_x(sol_EU[0, int(time_tracker.get_value())])
                        .set_y(sol_EU[0 + N, int(time_tracker.get_value())]))
        dot_EU_2.add_updater(lambda m: m.set_x(sol_EU[1, int(time_tracker.get_value())])
                        .set_y(sol_EU[1 + N, int(time_tracker.get_value())]))
        dot_LF_1.add_updater(lambda m: m.set_x(sol_LF[0, int(time_tracker.get_value())])
                        .set_y(sol_LF[0 + N, int(time_tracker.get_value())]))
        dot_LF_2.add_updater(lambda m: m.set_x(sol_LF[1, int(time_tracker.get_value())])
                        .set_y(sol_LF[1 + N, int(time_tracker.get_value())]))

        # Play animation
        num_steps = len(t_pts)
        self.play(
            time_tracker.animate.set_value(num_steps - 1),    # Runs time_tracker from 0 to last index of t_pts
            run_time=duration,
            rate_func=linear
        )
        self.wait()
