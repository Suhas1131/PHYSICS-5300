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

        # --- 2) Time setup (DDP style) ---
        t_start = 0.0
        t_end = 30.0                   # simulated seconds
        duration = t_end - t_start     # actual playback duration
        delta_t = 1 / config.frame_rate
        t_pts = np.arange(t_start, t_end, delta_t)
        t_pts = np.append(t_pts, t_end)
        print("t_pts shape =", t_pts.shape)

        #Initial conditions for circular orbit
        y0 = np.array([-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, -0.5])

        # Trajectories
        system = NBody2D(masses, radii, G=1.0)
        sol_EU = system.euler(t_pts, y0)
        sol_LF = system.leapfrog(t_pts, y0)

        # Create dots
        bodies_EU = VGroup()
        bodies_LF = VGroup()
        
        for i in range(N):
            
            # Place Euler dot at its initial position
            dot_EU = Dot(radius=radii[i], color=BLUE).move_to([
                sol_EU[i, 0], sol_EU[i + N, 0], 0
            ])
            bodies_EU.add(dot_EU)
            
            # Place Leapfrog dot at its initial position
            dot_LF = Dot(radius=radii[i], color=RED).move_to([
                sol_LF[i, 0], sol_LF[i + N, 0], 0
            ])
            bodies_LF.add(dot_LF)

        # Create tracers
        traces_EU = VGroup(*[
            TracedPath(dot.get_center, stroke_color=BLUE, stroke_width=1)
            for dot in bodies_EU
        ])
        traces_LF = VGroup(*[
            TracedPath(dot.get_center, stroke_color=RED, stroke_width=1)
            for dot in bodies_LF
        ])

        # Add tracers and bodies to the simulation
        self.add(traces_EU, traces_LF, bodies_EU, bodies_LF)
        self.wait()

        # Store time index
        time_tracker = ValueTracker(0)

        # Attach different bodies to different time values
        for i, dot in enumerate(bodies_EU):
            dot.add_updater(lambda m, i=i: m.move_to([
                sol_EU[i, int(time_tracker.get_value())],    # x-coord
                sol_EU[i + N, int(time_tracker.get_value())],    # y-coord
                0    # z-coord
            ]))
        for i, dot in enumerate(bodies_LF):
            dot.add_updater(lambda m, i=i: m.move_to([
                sol_LF[i, int(time_tracker.get_value())],
                sol_LF[i + N, int(time_tracker.get_value())],
                0
            ]))

        # Play animation
        num_steps = len(t_pts)
        self.play(
            time_tracker.animate.set_value(num_steps - 1),    # Runs time_tracker from 0 to last index of t_pts
            run_time=duration,
            rate_func=linear
        )
        self.wait()