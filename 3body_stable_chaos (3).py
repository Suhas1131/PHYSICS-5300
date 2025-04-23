from manim import *
import numpy as np
from NBody import NBody2D

# Prevent Manim from auto‐opening rendered media since it led to errors
config.preview = False
config.show_in_file_browser = False

class ThreeBody_chaos(Scene):
    def construct(self):

        N = 3
        masses = [1.0] * N
        radii = [0.05] * N

        t_start = 0.
        t_end = 70.
        delta_t = 1 / config.frame_rate
        t_pts = np.arange(t_start, t_end, delta_t)
        t_pts = np.append(t_pts, t_end)

        # figure‐8 initial conditions
        y0 = np.array([
             0.97000436, -0.97000436,  0.0,
            -0.24308753,  0.24308753,  0.0,
             0.466203685,  0.466203685, -0.93240737,
             0.43236573,   0.43236573,  -0.86473146])

        # Slight perturbation
        y0p = y0.copy()
        y0p[0] += 1e-6

        system = NBody2D(masses, radii, G=1.0, enable_collisions=False)
        sol_og = system.leapfrog(t_pts, y0)
        sol_pert = system.leapfrog(t_pts, y0p)

        # Create and place dots
        dot_1 = Dot(radius=radii[0], color=RED)
        dot_2 = Dot(radius=radii[1], color=RED)
        dot_3 = Dot(radius=radii[2], color=RED)
        dot_1.set_x(sol_og[0, 0]).set_y(sol_og[0 + N, 0])
        dot_2.set_x(sol_og[1, 0]).set_y(sol_og[1 + N, 0])
        dot_3.set_x(sol_og[2, 0]).set_y(sol_og[2 + N, 0])

        dot_p_1 = Dot(radius=radii[0], color=BLUE)
        dot_p_2 = Dot(radius=radii[1], color=BLUE)
        dot_p_3 = Dot(radius=radii[2], color=BLUE)
        dot_p_1.set_x(sol_pert[0, 0]).set_y(sol_pert[0 + N, 0])
        dot_p_2.set_x(sol_pert[1, 0]).set_y(sol_pert[1 + N, 0])
        dot_p_3.set_x(sol_pert[2, 0]).set_y(sol_pert[2 + N, 0])

        # Tracers
        trace_1 = TracedPath(dot_1.get_center, stroke_color=RED, stroke_width=1)
        trace_2 = TracedPath(dot_2.get_center, stroke_color=RED, stroke_width=1)
        trace_3 = TracedPath(dot_3.get_center, stroke_color=RED, stroke_width=1)

        trace_p_1 = TracedPath(dot_p_1.get_center, stroke_color=BLUE, stroke_width=1)
        trace_p_2 = TracedPath(dot_p_2.get_center, stroke_color=BLUE, stroke_width=1)
        trace_p_3 = TracedPath(dot_p_3.get_center, stroke_color=BLUE, stroke_width=1)

        # Add everything
        self.add(trace_1, trace_2, trace_3,
                 trace_p_1, trace_p_2, trace_p_3,
                 dot_1, dot_2, dot_3,
                 dot_p_1, dot_p_2, dot_p_3)
        self.wait()

        # Updaters
        tracker = ValueTracker(0)
        dot_o_1.add_updater(lambda m: m.set_x(sol_og[0, int(tracker.get_value())])
                        .set_y(sol_og[0 + N, int(tracker.get_value())]))
        dot_o_2.add_updater(lambda m: m.set_x(sol_og[1, int(tracker.get_value())])
                        .set_y(sol_og[1 + N, int(tracker.get_value())]))
        dot_o_3.add_updater(lambda m: m.set_x(sol_og[2, int(tracker.get_value())])
                        .set_y(sol_og[2 + N, int(tracker.get_value())]))

        dot_p_1.add_updater(lambda m: m.set_x(sol_pert[0, int(tracker.get_value())])
                        .set_y(sol_pert[0 + N, int(tracker.get_value())]))
        dot_p_2.add_updater(lambda m: m.set_x(sol_pert[1, int(tracker.get_value())])
                        .set_y(sol_pert[1 + N, int(tracker.get_value())]))
        dot_p_3.add_updater(lambda m: m.set_x(sol_pert[2, int(tracker.get_value())])
                        .set_y(sol_pert[2 + N, int(tracker.get_value())]))

        steps = len(t_pts)
        self.play(
            tracker.animate.set_value(steps - 1),
            run_time = 30,
            rate_func = linear
        )
        self.wait()
