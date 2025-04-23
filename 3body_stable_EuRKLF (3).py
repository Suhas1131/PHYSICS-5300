# Run using:
#         manim -pql 3body_stable_EuRKLF.py ThreeBody_stable

from manim import *
import numpy as np
from NBody import NBody2D

# Prevent Manim from auto-opening rendered media
config.preview = False
config.show_in_file_browser = False

class ThreeBody_stable(Scene):
    def construct(self):

        N = 3
        masses = [1.0] * N
        radii = [0.05] * N

        t_start = 0.
        t_end = 50.
        duration = t_end - t_start
        delta_t = 1 / config.frame_rate
        t_pts = np.arange(t_start, t_end, delta_t)
        t_pts = np.append(t_pts, t_end)

        # Initial conditions for figure-8
        y0 = np.array([
             0.97000436, -0.97000436,  0.0,
            -0.24308753,  0.24308753,  0.0,
             0.466203685,  0.466203685, -0.93240737,
             0.43236573,   0.43236573,  -0.86473146])

        # Integrate with three methods
        system = NBody2D(masses, radii, G=1.0, enable_collisions=False)
        sol_eu = system.euler(t_pts, y0)
        sol_lf = system.leapfrog(t_pts, y0)
        ode_rk = system.solve_ode(
            t_pts, y0,
            method='RK23',
            abserr=1e-9,
            relerr=1e-9)
        sol_rk = ode_rk.y

        # Create and place dots
        dot_EU_1 = Dot(radius=radii[0], color=BLUE)
        dot_EU_2 = Dot(radius=radii[1], color=BLUE)
        dot_EU_3 = Dot(radius=radii[2], color=BLUE)
        dot_EU_1.set_x(sol_eu[0, 0]).set_y(sol_eu[0 + N, 0])
        dot_EU_2.set_x(sol_eu[1, 0]).set_y(sol_eu[1 + N, 0])
        dot_EU_3.set_x(sol_eu[2, 0]).set_y(sol_eu[2 + N, 0])

        dot_LF_1 = Dot(radius=radii[0], color=RED)
        dot_LF_2 = Dot(radius=radii[1], color=RED)
        dot_LF_3 = Dot(radius=radii[2], color=RED)
        dot_LF_1.set_x(sol_lf[0, 0]).set_y(sol_lf[0 + N, 0])
        dot_LF_2.set_x(sol_lf[1, 0]).set_y(sol_lf[1 + N, 0])
        dot_LF_3.set_x(sol_lf[2, 0]).set_y(sol_lf[2 + N, 0])

        dot_RK_1 = Dot(radius=radii[0], color=GREEN)
        dot_RK_2 = Dot(radius=radii[1], color=GREEN)
        dot_RK_3 = Dot(radius=radii[2], color=GREEN)
        dot_RK_1.set_x(sol_rk[0, 0]).set_y(sol_rk[0 + N, 0])
        dot_RK_2.set_x(sol_rk[1, 0]).set_y(sol_rk[1 + N, 0])
        dot_RK_3.set_x(sol_rk[2, 0]).set_y(sol_rk[2 + N, 0])

        # Tracers
        trace_EU_1 = TracedPath(dot_EU_1.get_center, stroke_color=BLUE, stroke_width=1)
        trace_EU_2 = TracedPath(dot_EU_2.get_center, stroke_color=BLUE, stroke_width=1)
        trace_EU_3 = TracedPath(dot_EU_3.get_center, stroke_color=BLUE, stroke_width=1)

        trace_LF_1 = TracedPath(dot_LF_1.get_center, stroke_color=RED, stroke_width=1)
        trace_LF_2 = TracedPath(dot_LF_2.get_center, stroke_color=RED, stroke_width=1)
        trace_LF_3 = TracedPath(dot_LF_3.get_center, stroke_color=RED, stroke_width=1)

        trace_RK_1 = TracedPath(dot_RK_1.get_center, stroke_color=GREEN, stroke_width=1)
        trace_RK_2 = TracedPath(dot_RK_2.get_center, stroke_color=GREEN, stroke_width=1)
        trace_RK_3 = TracedPath(dot_RK_3.get_center, stroke_color=GREEN, stroke_width=1)

        # Add everything to the scene
        self.add(trace_EU_1, trace_EU_2, trace_EU_3,
                 trace_LF_1, trace_LF_2, trace_LF_3,
                 trace_RK_1, trace_RK_2, trace_RK_3,
                 dot_EU_1, dot_EU_2, dot_EU_3,
                 dot_LF_1, dot_LF_2, dot_LF_3,
                 dot_RK_1, dot_RK_2, dot_RK_3)
        self.wait()

        # Updaters
        time_tracker = ValueTracker(0)
        num_steps = len(t_pts)

        dot_EU_1.add_updater(lambda m: m.set_x(sol_eu[0, int(time_tracker.get_value())])
                            .set_y(sol_eu[0 + N, int(time_tracker.get_value())]))
        dot_EU_2.add_updater(lambda m: m.set_x(sol_eu[1, int(time_tracker.get_value())])
                            .set_y(sol_eu[1 + N, int(time_tracker.get_value())]))
        dot_EU_3.add_updater(lambda m: m.set_x(sol_eu[2, int(time_tracker.get_value())])
                            .set_y(sol_eu[2 + N, int(time_tracker.get_value())]))

        dot_LF_1.add_updater(lambda m: m.set_x(sol_lf[0, int(time_tracker.get_value())])
                            .set_y(sol_lf[0 + N, int(time_tracker.get_value())]))
        dot_LF_2.add_updater(lambda m: m.set_x(sol_lf[1, int(time_tracker.get_value())])
                            .set_y(sol_lf[1 + N, int(time_tracker.get_value())]))
        dot_LF_3.add_updater(lambda m: m.set_x(sol_lf[2, int(time_tracker.get_value())])
                            .set_y(sol_lf[2 + N, int(time_tracker.get_value())]))

        dot_RK_1.add_updater(lambda m: m.set_x(sol_rk[0, int(time_tracker.get_value())])
                            .set_y(sol_rk[0 + N, int(time_tracker.get_value())]))
        dot_RK_2.add_updater(lambda m: m.set_x(sol_rk[1, int(time_tracker.get_value())])
                            .set_y(sol_rk[1 + N, int(time_tracker.get_value())]))
        dot_RK_3.add_updater(lambda m: m.set_x(sol_rk[2, int(time_tracker.get_value())])
                            .set_y(sol_rk[2 + N, int(time_tracker.get_value())]))

        # Animate
        self.play(
            time_tracker.animate.set_value(num_steps - 1),
            run_time=duration,
            rate_func=linear
        )
        self.wait()
