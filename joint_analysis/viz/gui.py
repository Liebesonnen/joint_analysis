"""
GUI for joint analysis using DearPyGUI.
"""

import os
import time
import threading
import numpy as np
import dearpygui.dearpygui as dpg
from typing import Dict, List, Optional, Callable, Any, Union


class JointAnalysisGUI:
    """
    GUI for joint analysis using DearPyGUI.
    """

    def __init__(self):
        """Initialize the GUI."""
        # Initialize DearPyGUI
        dpg.create_context()

        # State variables
        self.stop_refresh = False
        self.modes = []

        # Data storage
        self.velocity_profile = {}
        self.angular_velocity_profile = {}
        self.col_score_profile = {}
        self.cop_score_profile = {}
        self.rad_score_profile = {}
        self.zp_score_profile = {}
        self.position_error_profile = {}
        self.angular_error_profile = {}
        self.joint_prob_profile = {}

        # Window configuration
        self.window_width = 1200
        self.window_height = 800

        # Analysis results callback
        self.result_callback = None

    def setup_modes(self, modes: List[str]) -> None:
        """
        Set up the available modes.

        Args:
            modes (List[str]): List of mode names
        """
        self.modes = modes

        # Initialize data structures for each mode
        for mode in self.modes:
            self.velocity_profile[mode] = []
            self.angular_velocity_profile[mode] = []
            self.col_score_profile[mode] = []
            self.cop_score_profile[mode] = []
            self.rad_score_profile[mode] = []
            self.zp_score_profile[mode] = []
            self.position_error_profile[mode] = []
            self.angular_error_profile[mode] = []
            self.joint_prob_profile[mode] = {
                "prismatic": [],
                "planar": [],
                "revolute": [],
                "screw": [],
                "ball": []
            }

    def clear_data(self) -> None:
        """Clear all data."""
        for mode in self.modes:
            self.velocity_profile[mode].clear()
            self.angular_velocity_profile[mode].clear()
            self.col_score_profile[mode].clear()
            self.cop_score_profile[mode].clear()
            self.rad_score_profile[mode].clear()
            self.zp_score_profile[mode].clear()
            self.position_error_profile[mode].clear()
            self.angular_error_profile[mode].clear()
            for jt_name in self.joint_prob_profile[mode]:
                self.joint_prob_profile[mode][jt_name].clear()

    def set_result_callback(self, callback: Callable) -> None:
        """
        Set the callback function for analysis results.

        Args:
            callback (Callable): Callback function
        """
        self.result_callback = callback

    def update_plots(self) -> None:
        """Update all plot data."""
        if self.stop_refresh:
            return

        for mode in self.modes:
            # Update velocity plot
            if dpg.does_item_exist(f"{mode}_vel_series"):
                vel_data = self.velocity_profile[mode]
                x_data = list(range(len(vel_data)))
                dpg.set_value(f"{mode}_vel_series", [x_data, vel_data])

            # Update angular velocity plot
            if dpg.does_item_exist(f"{mode}_omega_series"):
                omega_data = self.angular_velocity_profile[mode]
                dpg.set_value(f"{mode}_omega_series", [list(range(len(omega_data))), omega_data])

            # Update basic scores plots
            if dpg.does_item_exist(f"{mode}_col_series"):
                col_data = self.col_score_profile[mode]
                dpg.set_value(f"{mode}_col_series", [list(range(len(col_data))), col_data])

            if dpg.does_item_exist(f"{mode}_cop_series"):
                cop_data = self.cop_score_profile[mode]
                dpg.set_value(f"{mode}_cop_series", [list(range(len(cop_data))), cop_data])

            if dpg.does_item_exist(f"{mode}_rad_series"):
                rad_data = self.rad_score_profile[mode]
                dpg.set_value(f"{mode}_rad_series", [list(range(len(rad_data))), rad_data])

            if dpg.does_item_exist(f"{mode}_zp_series"):
                zp_data = self.zp_score_profile[mode]
                dpg.set_value(f"{mode}_zp_series", [list(range(len(zp_data))), zp_data])

            # Update error plots
            if dpg.does_item_exist(f"{mode}_pos_err_series"):
                pos_err_data = self.position_error_profile[mode]
                dpg.set_value(f"{mode}_pos_err_series", [list(range(len(pos_err_data))), pos_err_data])

            if dpg.does_item_exist(f"{mode}_ang_err_series"):
                ang_err_data = self.angular_error_profile[mode]
                dpg.set_value(f"{mode}_ang_err_series", [list(range(len(ang_err_data))), ang_err_data])

            # Update joint probability plots
            for jt in ["prismatic", "planar", "revolute", "screw", "ball"]:
                if dpg.does_item_exist(f"{mode}_prob_{jt}_series"):
                    p_data = self.joint_prob_profile[mode][jt]
                    dpg.set_value(f"{mode}_prob_{jt}_series", [list(range(len(p_data))), p_data])
    def add_analysis_results(self, mode: str, analysis_data: Dict[str, Any]) -> None:
        """
        Add analysis results for a mode.

        Args:
            mode (str): Mode name
            analysis_data (Dict[str, Any]): Analysis results
        """
        if mode not in self.modes:
            return

        # Extract data
        basic_scores = analysis_data.get("basic_score_avg", {})
        joint_probs = analysis_data.get("joint_probs", {})
        position_error = analysis_data.get("position_error", 0.0)
        angular_error = analysis_data.get("angular_error", 0.0)

        # Update basic scores
        col_m = basic_scores.get("col_mean", 0.0)
        cop_m = basic_scores.get("cop_mean", 0.0)
        rad_m = basic_scores.get("rad_mean", 0.0)
        zp_m = basic_scores.get("zp_mean", 0.0)

        self.col_score_profile[mode].append(col_m)
        self.cop_score_profile[mode].append(cop_m)
        self.rad_score_profile[mode].append(rad_m)
        self.zp_score_profile[mode].append(zp_m)

        # Update joint probabilities
        for jt in ["prismatic", "planar", "revolute", "screw", "ball"]:
            prob = joint_probs.get(jt, 0.0)
            self.joint_prob_profile[mode][jt].append(prob)

        # Update errors
        self.position_error_profile[mode].append(position_error)
        self.angular_error_profile[mode].append(angular_error)

        # Update plots
        self.update_plots()

    def add_velocity_data(self, mode: str, velocity: float, angular_velocity: float) -> None:
        """
        Add velocity data for a mode.

        Args:
            mode (str): Mode name
            velocity (float): Linear velocity
            angular_velocity (float): Angular velocity
        """
        if mode not in self.modes:
            return

        self.velocity_profile[mode].append(velocity)
        self.angular_velocity_profile[mode].append(angular_velocity)

        # Update plots
        self.update_plots()

    def create_gui(self) -> None:
        """Create the GUI layout."""
        with dpg.window(label="Joint Analysis", width=self.window_width, height=self.window_height):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Pause Refresh", callback=self._toggle_refresh, tag="refresh_button")
                dpg.add_button(label="Clear Plots", callback=self._clear_plots)

                with dpg.collapsing_header(label="Select Modes to Display", default_open=False,
                                           tag="mode_selector_group"):
                    for mode in self.modes:
                        dpg.add_checkbox(label=mode, default_value=(mode == self.modes[0]),
                                         callback=self._on_mode_change, tag=f"checkbox_{mode}")

            with dpg.tab_bar():
                # Velocity and Omega tab
                with dpg.tab(label="Velocity and Omega"):
                    with dpg.group(horizontal=True):
                        # Linear velocity plot
                        with dpg.plot(label="Weighted linear velocity in different time frames", height=300, width=450):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="Time frames", tag="x_axis_vel")
                            y_axis_vel = dpg.add_plot_axis(dpg.mvYAxis, label="Weighted velocity", tag="y_axis_vel")

                            for mode in self.modes:
                                tag_line = f"{mode}_vel_series"
                                dpg.add_line_series([], [], label=mode, parent=y_axis_vel, tag=tag_line,
                                                    show=(mode == self.modes[0]))

                        # Angular velocity plot
                        with dpg.plot(label="Weighted angular velocity in different time frames", height=300,
                                      width=450):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="Time frames", tag="x_axis_omega")
                            y_axis_omega = dpg.add_plot_axis(dpg.mvYAxis, label="Weighted omega", tag="y_axis_omega")

                            for mode in self.modes:
                                tag_line = f"{mode}_omega_series"
                                dpg.add_line_series([], [], label=mode, parent=y_axis_omega, tag=tag_line,
                                                    show=(mode == self.modes[0]))

                # Basic Scores tab
                with dpg.tab(label="Basic Scores"):
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            for label, tag_y, suffix in [("Collinearity", "y_axis_col", "col_series"),
                                                         ("Radius Consistency", "y_axis_rad", "rad_series")]:
                                with dpg.plot(label=f"{label} Score in different time frames", height=250, width=400):
                                    dpg.add_plot_legend()
                                    dpg.add_plot_axis(dpg.mvXAxis, label="Time frames")
                                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label=f"{label} Score", tag=tag_y)
                                    dpg.set_axis_limits(tag_y, -0.1, 1.1)

                                    for mode in self.modes:
                                        tag_line = f"{mode}_{suffix}"
                                        dpg.add_line_series([], [], label=mode, parent=y_axis, tag=tag_line,
                                                            show=(mode == self.modes[0]))

                        with dpg.group():
                            for label, tag_y, suffix in [("Coplanarity", "y_axis_cop", "cop_series"),
                                                         ("Zero Pitch", "y_axis_zp", "zp_series")]:
                                with dpg.plot(label=f"{label} Score in different time frames", height=250, width=400):
                                    dpg.add_plot_legend()
                                    dpg.add_plot_axis(dpg.mvXAxis, label="Time frames")
                                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label=f"{label} Score", tag=tag_y)
                                    dpg.set_axis_limits(tag_y, -0.1, 1.1)

                                    for mode in self.modes:
                                        tag_line = f"{mode}_{suffix}"
                                        dpg.add_line_series([], [], label=mode, parent=y_axis, tag=tag_line,
                                                            show=(mode == self.modes[0]))

                # Joint Errors tab
                with dpg.tab(label="Joint Errors"):
                    with dpg.group(horizontal=True):
                        # Position error plot
                        with dpg.plot(label="Position Error in different time frames", height=300, width=450):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="Time frames", tag="x_axis_pos_err")
                            y_axis_pos_err = dpg.add_plot_axis(dpg.mvYAxis, label="Position Error",
                                                               tag="y_axis_pos_err")

                            for mode in self.modes:
                                tag_line = f"{mode}_pos_err_series"
                                dpg.add_line_series([], [], label=mode, parent=y_axis_pos_err, tag=tag_line,
                                                    show=(mode == self.modes[0]))

                        # Angular error plot
                        with dpg.plot(label="Angular Error in different time frames", height=300, width=450):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="Time frames", tag="x_axis_ang_err")
                            y_axis_ang_err = dpg.add_plot_axis(dpg.mvYAxis, label="Angular Error", tag="y_axis_ang_err")

                            for mode in self.modes:
                                tag_line = f"{mode}_ang_err_series"
                                dpg.add_line_series([], [], label=mode, parent=y_axis_ang_err, tag=tag_line,
                                                    show=(mode == self.modes[0]))

                # Joint Probabilities tab
                with dpg.tab(label="Joint Probabilities"):
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            for label, tag_y, suffix in [("Prismatic", "y_axis_prob_pm", "prob_prismatic_series"),
                                                         ("Planar", "y_axis_prob_pl", "prob_planar_series"),
                                                         ("Revolute", "y_axis_prob_rv", "prob_revolute_series")]:
                                with dpg.plot(label=f"{label} Probability in different time frames", height=250,
                                              width=400):
                                    dpg.add_plot_legend()
                                    dpg.add_plot_axis(dpg.mvXAxis, label="Time frames")
                                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label=f"{label} Probability", tag=tag_y)
                                    dpg.set_axis_limits(tag_y, -0.1, 1.1)

                                    for mode in self.modes:
                                        tag_line = f"{mode}_{suffix}"
                                        dpg.add_line_series([], [], label=mode, parent=y_axis, tag=tag_line,
                                                            show=(mode == self.modes[0]))

                        with dpg.group():
                            for label, tag_y, suffix in [("Screw", "y_axis_prob_sc", "prob_screw_series"),
                                                         ("Ball", "y_axis_prob_ba", "prob_ball_series")]:
                                with dpg.plot(label=f"{label} Probability in different time frames", height=250,
                                              width=400):
                                    dpg.add_plot_legend()
                                    dpg.add_plot_axis(dpg.mvXAxis, label="Time frames")
                                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label=f"{label} Probability", tag=tag_y)
                                    dpg.set_axis_limits(tag_y, -0.1, 1.1)

                                    for mode in self.modes:
                                        tag_line = f"{mode}_{suffix}"
                                        dpg.add_line_series([], [], label=mode, parent=y_axis, tag=tag_line,
                                                            show=(mode == self.modes[0]))

    def _toggle_refresh(self, sender, app_data) -> None:
        """
        Toggle the refresh state.

        Args:
            sender: Sender item
            app_data: Application data
        """
        self.stop_refresh = not self.stop_refresh
        if self.stop_refresh:
            dpg.set_item_label(sender, "Resume Refresh")
        else:
            dpg.set_item_label(sender, "Pause Refresh")
            self.update_plots()

    def _clear_plots(self) -> None:
        """Clear all plots."""
        self.clear_data()
        self.update_plots()

        if self.result_callback:
            self.result_callback("clear_plots")

    def _on_mode_change(self, sender, app_data) -> None:
        """
        Handle mode selection change.

        Args:
            sender: Sender item
            app_data: Application data
        """
        selected_modes = []
        for mode in self.modes:
            if dpg.get_value(f"checkbox_{mode}"):
                selected_modes.append(mode)

        for mode in self.modes:
            for suffix in ["vel_series", "omega_series", "col_series", "rad_series",
                           "cop_series", "zp_series",
                           "pos_err_series", "ang_err_series",
                           "prob_prismatic_series", "prob_planar_series",
                           "prob_revolute_series", "prob_screw_series", "prob_ball_series"]:
                tag = f"{mode}_{suffix}"
                if dpg.does_item_exist(tag):
                    dpg.configure_item(tag, show=(mode in selected_modes))

    def start(self, width: int = 1250, height: int = 900) -> None:
        """
        Start the GUI.

        Args:
            width (int): Viewport width
            height (int): Viewport height
        """
        # Create viewport
        dpg.create_viewport(title='Joint Analysis - Plots', width=width, height=height)
        dpg.setup_dearpygui()

        # Create GUI layout
        self.create_gui()

        # Initialize mode selection state
        self._on_mode_change(None, None)

        # Show viewport
        dpg.show_viewport()

    def run(self) -> None:
        """Run the GUI loop."""
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
            time.sleep(0.033)  # ~30 FPS

            if not self.stop_refresh:
                self.update_plots()

    def start_in_thread(self) -> threading.Thread:
        """
        Start the GUI in a separate thread.

        Returns:
            threading.Thread: The GUI thread
        """
        self.start()

        # Create and start thread
        gui_thread = threading.Thread(target=self.run, daemon=True)
        gui_thread.start()

        return gui_thread
    def shutdown(self) -> None:
        """Shutdown the GUI."""
        if dpg.is_dearpygui_running():
            dpg.destroy_context()