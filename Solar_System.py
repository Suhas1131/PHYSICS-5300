# manim -pql Solar_System.py SolarSystem

from manim import *
import numpy as np
from NBody import NBody2D

config.preview = False
config.show_in_file_browser = False

class SolarSystem(Scene):
    def construct(self):
        # System parameters (Sun + 8 planets)
        N = 9
        m = np.array([0.330, 4.87, 5.97, 0.642,
                           1898, 568, 86.8, 102])    # (10^24 kg)
        M_sun = 1988400.    # Sun mass (10^24 kg)
        masses = [1.0] + (m / M_sun).tolist()    # Converting to Solar masses

        # radii
        radii_km = np.array([
            696340, 2439.7, 6051.8, 6371.0,
            3389.5, 69911.0, 58232.0, 25362.0, 24622.0])
        base_dot = 0.02
        radii = (radii_km / radii_km[0] * base_dot).tolist()

        G_sim = 4 * np.pi**2    # from P² = (4 * pi^2) a³ / GM
        system = NBody2D(masses, radii, G = G_sim)

        t_start, t_end = 0.0, 50.
        delta_t_anim = 1 / config.frame_rate
        delta_t_sim = delta_t_anim / 10    # integration step to maintain accuracy
        t_sim = np.arange(t_start, t_end + delta_t_sim, delta_t_sim)
        t_pts = np.arange(t_start, t_end + delta_t_anim, delta_t_anim)

        # Initial condition
        dist = np.array([57.9, 108.2, 149.6, 228.0, 778.5, 1432.0, 2867.0, 4515.0])    # 10^6 km
        AU = 149597870.7    # km
        a = (dist * 1e6) / (AU)    # semimajor axes in AU
        # From v = (GM/r)0.5
        speeds = 2 * np.pi / np.sqrt(a)    # (AU/yr)

        # scale distances to make simulation fit on the screen
        distance_scale = 6.0 / np.max(a)
        
        y0 = np.zeros(4 * N)
        for i, (ai, vi) in enumerate(zip(a, speeds), start=1):
            y0[i] = ai
            y0[N+i] = 0.0
            y0[2*N+i] = 0.0
            y0[3*N+i] = vi

        # to keep physics accurate while keeping frames low
        sol_sim = system.leapfrog(t_sim, y0)
        ratio = int(delta_t_anim / delta_t_sim)    # If ratio = n
        frames_idx = np.arange(0, sol_sim.shape[1], ratio)    # Pick every nth step
        if frames_idx[-1] != sol_sim.shape[1] - 1:
            frames_idx = np.append(frames_idx, sol_sim.shape[1] - 1)
        sol = sol_sim[:, frames_idx]    # Every nth step of the trajectory

        colors = [YELLOW, GREY, ORANGE, BLUE, RED, GOLD, PURPLE, TEAL, GREEN]
        dot_0 = Dot(radius=radii[0], color=colors[0])
        dot_0.set_x(sol[0, 0] * distance_scale).set_y(sol[0 + N, 0] * distance_scale)
        dot_1 = Dot(radius=radii[1], color=colors[1])
        dot_1.set_x(sol[1, 0] * distance_scale).set_y(sol[1 + N, 0] * distance_scale)
        dot_2 = Dot(radius=radii[2], color=colors[2])
        dot_2.set_x(sol[2, 0] * distance_scale).set_y(sol[2 + N, 0] * distance_scale)
        dot_3 = Dot(radius=radii[3], color=colors[3])
        dot_3.set_x(sol[3, 0] * distance_scale).set_y(sol[3 + N, 0] * distance_scale)
        dot_4 = Dot(radius=radii[4], color=colors[4])
        dot_4.set_x(sol[4, 0] * distance_scale).set_y(sol[4 + N, 0] * distance_scale)
        dot_5 = Dot(radius=radii[5], color=colors[5])
        dot_5.set_x(sol[5, 0] * distance_scale).set_y(sol[5 + N, 0] * distance_scale)
        dot_6 = Dot(radius=radii[6], color=colors[6])
        dot_6.set_x(sol[6, 0] * distance_scale).set_y(sol[6 + N, 0] * distance_scale)
        dot_7 = Dot(radius=radii[7], color=colors[7])
        dot_7.set_x(sol[7, 0] * distance_scale).set_y(sol[7 + N, 0] * distance_scale)
        dot_8 = Dot(radius=radii[8], color=colors[8])
        dot_8.set_x(sol[8, 0] * distance_scale).set_y(sol[8 + N, 0] * distance_scale)

        # Create tracers for planets
        trace_1 = TracedPath(dot_1.get_center, stroke_color=colors[1], stroke_width=1)
        trace_2 = TracedPath(dot_2.get_center, stroke_color=colors[2], stroke_width=1)
        trace_3 = TracedPath(dot_3.get_center, stroke_color=colors[3], stroke_width=1)
        trace_4 = TracedPath(dot_4.get_center, stroke_color=colors[4], stroke_width=1)
        trace_5 = TracedPath(dot_5.get_center, stroke_color=colors[5], stroke_width=1)
        trace_6 = TracedPath(dot_6.get_center, stroke_color=colors[6], stroke_width=1)
        trace_7 = TracedPath(dot_7.get_center, stroke_color=colors[7], stroke_width=1)
        trace_8 = TracedPath(dot_8.get_center, stroke_color=colors[8], stroke_width=1)

        # Add everything to the scene
        self.add(trace_1, trace_2, trace_3, trace_4, trace_5, trace_6, trace_7, trace_8,
                 dot_0, dot_1, dot_2, dot_3, dot_4, dot_5, dot_6, dot_7, dot_8)
        self.wait()

        # Attach updater
        tracker = ValueTracker(0)
        dot_0.add_updater(lambda m: m.set_x(sol[0, int(tracker.get_value())] * distance_scale)
                          .set_y(sol[0 + N, int(tracker.get_value())] * distance_scale))
        dot_1.add_updater(lambda m: m.set_x(sol[1, int(tracker.get_value())] * distance_scale)
                        .set_y(sol[1 + N, int(tracker.get_value())] * distance_scale))
        dot_2.add_updater(lambda m: m.set_x(sol[2, int(tracker.get_value())] * distance_scale)
                        .set_y(sol[2 + N, int(tracker.get_value())] * distance_scale))
        dot_3.add_updater(lambda m: m.set_x(sol[3, int(tracker.get_value())] * distance_scale)
                        .set_y(sol[3 + N, int(tracker.get_value())] * distance_scale))
        dot_4.add_updater(lambda m: m.set_x(sol[4, int(tracker.get_value())] * distance_scale)
                        .set_y(sol[4 + N, int(tracker.get_value())] * distance_scale))
        dot_5.add_updater(lambda m: m.set_x(sol[5, int(tracker.get_value())] * distance_scale)
                        .set_y(sol[5 + N, int(tracker.get_value())] * distance_scale))
        dot_6.add_updater(lambda m: m.set_x(sol[6, int(tracker.get_value())] * distance_scale)
                        .set_y(sol[6 + N, int(tracker.get_value())] * distance_scale))
        dot_7.add_updater(lambda m: m.set_x(sol[7, int(tracker.get_value())] * distance_scale)
                        .set_y(sol[7 + N, int(tracker.get_value())] * distance_scale))
        dot_8.add_updater(lambda m: m.set_x(sol[8, int(tracker.get_value())] * distance_scale)
                        .set_y(sol[8 + N, int(tracker.get_value())] * distance_scale))

        num_frames = len(t_pts)
        self.play(
            tracker.animate.set_value(num_frames - 1),
            run_time=t_end,
            rate_func=linear
        )
        self.wait()