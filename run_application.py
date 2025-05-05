#!/usr/bin/env python
"""
Script to run the joint analysis application.
"""

import argparse
from joint_analysis import run_application

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run Joint Analysis Application")
    parser.add_argument("--output-dir", type=str, default="exported_pointclouds",
                        help="Directory to save exported data and plots")
    parser.add_argument("--save-plots", action="store_true",
                        help="Automatically save plots for all modes with data")
    parser.add_argument("--no-gui", action="store_true",
                        help="Run without GUI (useful for batch processing)")

    # Parse arguments
    args = parser.parse_args()

    # Run the application
    run_application(
        use_gui=not args.no_gui,
        output_dir=args.output_dir,
        auto_save_plots=args.save_plots
    )