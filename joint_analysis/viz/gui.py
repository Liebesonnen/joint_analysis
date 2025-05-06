"""
GUI for joint analysis using DearPyGUI.
"""

import os
import time
import threading
import numpy as np
import dearpygui.dearpygui as dpg
from typing import Dict, List, Optional, Callable, Any, Union
import matplotlib.pyplot as plt
from .plot_saver import PlotSaver

class PlotSaver:
    """
    Utility class to save plot data from JointAnalysisGUI to image files.
    """

    def __init__(self, output_dir="plot_images"):
        """
        Initialize the PlotSaver.

        Args:
            output_dir (str): Directory to save plot images
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_velocity_plots(self, mode, velocity_data, angular_velocity_data):
        """
        Save velocity and angular velocity plots.

        Args:
            mode (str): Joint mode name
            velocity_data (list): Linear velocity data
            angular_velocity_data (list): Angular velocity data
        """
        # Create mode-specific directory
        mode_dir = os.path.join(self.output_dir, mode.replace(" ", "_"))
        os.makedirs(mode_dir, exist_ok=True)

        # Plot linear velocity
        plt.figure(figsize=(10, 6))
        x_data = list(range(len(velocity_data)))
        plt.plot(x_data, velocity_data, 'b-', linewidth=2)
        plt.title(f'Linear Velocity over Time - {mode}')
        plt.xlabel('Time Frames')
        plt.ylabel('Linear Velocity Magnitude(m/s)')
        plt.grid(True)
        plt.savefig(os.path.join(mode_dir, 'linear_velocity.png'), dpi=50)
        plt.close()

        # Plot angular velocity
        plt.figure(figsize=(10, 6))
        x_data = list(range(len(angular_velocity_data)))
        plt.plot(x_data, angular_velocity_data, 'r-', linewidth=2)
        plt.title(f'Angular Velocity over Time - {mode}')
        plt.xlabel('Time Frames')
        plt.ylabel('Angular Velocity Magnitude(rad/s)')
        plt.grid(True)
        plt.savefig(os.path.join(mode_dir, 'angular_velocity.png'), dpi=50)
        plt.close()

    def save_basic_scores_plots(self, mode, col_data, cop_data, rad_data, zp_data):
        """
        Save basic scores plots.

        Args:
            mode (str): Joint mode name
            col_data (list): Collinearity scores
            cop_data (list): Coplanarity scores
            rad_data (list): Radius consistency scores
            zp_data (list): Zero pitch scores
        """
        # Create mode-specific directory
        mode_dir = os.path.join(self.output_dir, mode.replace(" ", "_"))
        os.makedirs(mode_dir, exist_ok=True)

        # Plot basic scores
        plt.figure(figsize=(12, 8))
        x_data = list(range(len(col_data)))

        plt.subplot(2, 2, 1)
        plt.plot(x_data, col_data, 'g-', linewidth=2)
        plt.title('Collinearity Score')
        plt.xlabel('Time Frames')
        plt.ylabel('Score')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(x_data, cop_data, 'b-', linewidth=2)
        plt.title('Coplanarity Score')
        plt.xlabel('Time Frames')
        plt.ylabel('Score')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(x_data, rad_data, 'm-', linewidth=2)
        plt.title('Radius Consistency Score')
        plt.xlabel('Time Frames')
        plt.ylabel('Score')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(x_data, zp_data, 'c-', linewidth=2)
        plt.title('Zero Pitch Score')
        plt.xlabel('Time Frames')
        plt.ylabel('Score')
        plt.grid(True)

        plt.tight_layout()
        plt.suptitle(f'Basic Scores over Time - {mode}', y=1.02, fontsize=16)
        plt.savefig(os.path.join(mode_dir, 'basic_scores.png'), dpi=50, bbox_inches='tight')
        plt.close()

    def save_joint_probability_plots(self, mode, joint_probs):
        """
        Save joint probability plots.

        Args:
            mode (str): Joint mode name
            joint_probs (dict): Dictionary mapping joint types to probability data
        """
        # Create mode-specific directory
        mode_dir = os.path.join(self.output_dir, mode.replace(" ", "_"))
        os.makedirs(mode_dir, exist_ok=True)

        # Plot joint probabilities
        plt.figure(figsize=(12, 8))

        # Get common x axis
        max_len = 0
        for jt_name, probs in joint_probs.items():
            max_len = max(max_len, len(probs))

        x_data = list(range(max_len))

        # Map joint types to colors for consistency
        colors = {
            'prismatic': 'b',
            'planar': 'g',
            'revolute': 'r',
            'screw': 'm',
            'ball': 'c'
        }

        for jt_name, probs in joint_probs.items():
            if len(probs) > 0:
                # Pad with last value if needed
                padded_probs = probs + [probs[-1]] * (max_len - len(probs)) if len(probs) < max_len else probs
                plt.plot(x_data, padded_probs, f'{colors.get(jt_name, "k")}-', linewidth=2, label=jt_name.capitalize())

        plt.title(f'Joint Type Probabilities over Time - {mode}')
        plt.xlabel('Time Frames')
        plt.ylabel('Probability')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(mode_dir, 'joint_probabilities.png'), dpi=50)
        plt.close()

    def save_error_plots(self, mode, position_error_data, angular_error_data):
        """
        Save error plots.

        Args:
            mode (str): Joint mode name
            position_error_data (list): Position error data
            angular_error_data (list): Angular error data
        """
        # Create mode-specific directory
        mode_dir = os.path.join(self.output_dir, mode.replace(" ", "_"))
        os.makedirs(mode_dir, exist_ok=True)

        # Plot errors
        plt.figure(figsize=(12, 5))

        x_data = list(range(len(position_error_data)))

        plt.subplot(1, 2, 1)
        plt.plot(x_data, position_error_data, 'b-', linewidth=2)
        plt.title('Position Error')
        plt.xlabel('Time Frames')
        plt.ylabel('Error(m)')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(x_data, angular_error_data, 'r-', linewidth=2)
        plt.title('Angular Error')
        plt.xlabel('Time Frames')
        plt.ylabel('Error(rad)')
        plt.grid(True)

        plt.tight_layout()
        plt.suptitle(f'Error Metrics over Time - {mode}', y=1.05, fontsize=16)
        plt.savefig(os.path.join(mode_dir, 'error_metrics.png'), dpi=50, bbox_inches='tight')
        plt.close()

    def save_all_plots(self, gui_instance):
        """
        Save all plots for all modes in the GUI.

        Args:
            gui_instance: Instance of JointAnalysisGUI
        """
        for mode in gui_instance.modes:
            # Only save plots if there's data
            if mode in gui_instance.velocity_profile and len(gui_instance.velocity_profile[mode]) > 0:
                self.save_velocity_plots(
                    mode,
                    gui_instance.velocity_profile[mode],
                    gui_instance.angular_velocity_profile[mode]
                )

                self.save_basic_scores_plots(
                    mode,
                    gui_instance.col_score_profile[mode],
                    gui_instance.cop_score_profile[mode],
                    gui_instance.rad_score_profile[mode],
                    gui_instance.zp_score_profile[mode]
                )

                self.save_joint_probability_plots(
                    mode,
                    gui_instance.joint_prob_profile[mode]
                )

                self.save_error_plots(
                    mode,
                    gui_instance.position_error_profile[mode],
                    gui_instance.angular_error_profile[mode]
                )
class JointAnalysisGUI:
    """
    GUI for joint analysis using DearPyGUI.
    """

    def __init__(self, output_dir="plot_images"):
        """Initialize the GUI."""
        # Initialize DearPyGUI
        dpg.create_context()

        # State variables
        self.stop_refresh = False
        self.modes = []
        self.output_dir = output_dir
        self.app_instance = None  # 存储应用程序实例的引用
        # Create plot saver
        self.plot_saver = PlotSaver(output_dir=output_dir)

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
        self.window_width = 1250
        self.window_height = 900

        # Analysis results callback
        self.result_callback = None

    def set_app_instance(self, app_instance):
        """
        Set the reference to the main application instance.

        Args:
            app_instance: Reference to the JointAnalysisApp instance
        """
        self.app_instance = app_instance

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

    # In joint_analysis/viz/gui.py
    def add_velocity_data(self, mode: str, velocity: float, angular_velocity: float) -> None:
        """
        Add velocity data for a mode.

        Args:
            mode (str): Mode name
            velocity (float): Linear velocity (instantaneous)
            angular_velocity (float): Angular velocity (instantaneous)
        """
        if mode not in self.modes:
            return

        # Store the raw instantaneous velocity values
        # Check if we've reached a reasonable limit of data points
        max_data_points = 150  # Adjust this value as needed

        if mode in self.velocity_profile and len(self.velocity_profile[mode]) >= max_data_points:
            # Shift data by removing oldest point
            self.velocity_profile[mode] = self.velocity_profile[mode][1:] + [velocity]
            self.angular_velocity_profile[mode] = self.angular_velocity_profile[mode][1:] + [angular_velocity]
        else:
            # Just append the new values
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

                # Add a Save Plots button
                dpg.add_button(label="Save Plots", callback=self._save_plots)

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
                            y_axis_vel = dpg.add_plot_axis(dpg.mvYAxis, label="Weighted velocity(m/s)", tag="y_axis_vel")

                            for mode in self.modes:
                                tag_line = f"{mode}_vel_series"
                                dpg.add_line_series([], [], label=mode, parent=y_axis_vel, tag=tag_line,
                                                    show=(mode == self.modes[0]))

                        # Angular velocity plot
                        with dpg.plot(label="Weighted angular velocity in different time frames", height=300,
                                      width=450):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="Time frames", tag="x_axis_omega")
                            y_axis_omega = dpg.add_plot_axis(dpg.mvYAxis, label="Weighted omega(rad/s)", tag="y_axis_omega")

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
                            y_axis_pos_err = dpg.add_plot_axis(dpg.mvYAxis, label="Position Error(m)",
                                                               tag="y_axis_pos_err")

                            for mode in self.modes:
                                tag_line = f"{mode}_pos_err_series"
                                dpg.add_line_series([], [], label=mode, parent=y_axis_pos_err, tag=tag_line,
                                                    show=(mode == self.modes[0]))

                        # Angular error plot
                        with dpg.plot(label="Angular Error in different time frames", height=300, width=450):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="Time frames", tag="x_axis_ang_err")
                            y_axis_ang_err = dpg.add_plot_axis(dpg.mvYAxis, label="Angular Error(m)", tag="y_axis_ang_err")

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

    def _save_plots(self) -> None:
        """Save all plots to image files."""
        # Create a message to show to the user
        with dpg.window(label="Saving Plots", modal=True, no_close=True, width=400, height=100,
                        pos=(int(self.window_width / 2 - 200), int(self.window_height / 2 - 50))):
            dpg.add_text("Saving plots, please wait...")
            progress_bar_tag = dpg.add_progress_bar(default_value=0.0, width=-1)

            # Create a separate thread to save plots to avoid blocking the GUI
            save_thread = threading.Thread(target=self._save_plots_thread, args=(progress_bar_tag,))
            save_thread.daemon = True
            save_thread.start()

    def _save_plots_thread(self, progress_bar_tag) -> None:
        """
        Thread function to save plots.

        Args:
            progress_bar_tag: Tag of the progress bar to update
        """
        try:
            # Get selected modes
            selected_modes = []
            for mode in self.modes:
                if dpg.does_item_exist(f"checkbox_{mode}") and dpg.get_value(f"checkbox_{mode}"):
                    selected_modes.append(mode)

            # If no modes are selected, use all modes
            if not selected_modes:
                selected_modes = self.modes

            # Save plots for each selected mode
            for i, mode in enumerate(selected_modes):
                # Update progress
                progress = (i + 1) / len(selected_modes)
                dpg.set_value(progress_bar_tag, progress)

                # Skip modes with no data
                if len(self.velocity_profile.get(mode, [])) == 0:
                    continue

                # Save velocity plots
                self.plot_saver.save_velocity_plots(
                    mode,
                    self.velocity_profile[mode],
                    self.angular_velocity_profile[mode]
                )

                # Save basic scores plots
                self.plot_saver.save_basic_scores_plots(
                    mode,
                    self.col_score_profile[mode],
                    self.cop_score_profile[mode],
                    self.rad_score_profile[mode],
                    self.zp_score_profile[mode]
                )

                # Save joint probability plots
                self.plot_saver.save_joint_probability_plots(
                    mode,
                    self.joint_prob_profile[mode]
                )

                # Save error plots
                self.plot_saver.save_error_plots(
                    mode,
                    self.position_error_profile[mode],
                    self.angular_error_profile[mode]
                )

                # Small delay to allow GUI to update
                time.sleep(0.1)

            # Wait a bit before closing the modal
            time.sleep(1.0)

        finally:
            # Close the modal window
            if dpg.does_item_exist(progress_bar_tag):
                parent = dpg.get_item_parent(progress_bar_tag)
                if dpg.does_item_exist(parent):
                    dpg.delete_item(parent)