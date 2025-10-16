
# ui/plotter.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_trajectory_3d(df, show_3d=True, show_2d=True):
    """
    df must contain columns: 'north', 'up', 'east' (and optionally 'time', 'speed').
    This function will:
      - create a 3D ENU plot (East, North, Up)
      - create a 2D ground-track (East vs North) and altitude vs range subplot
    """

    # Ensure aliases exist
    if "east" not in df.columns and "x" in df.columns:
        df["east"] = df["x"]
    if "north" not in df.columns and "y" in df.columns:
        df["north"] = df["y"]
    if "up" not in df.columns and "z" in df.columns:
        df["up"] = df["z"]

    east = df["east"].to_numpy()
    north = df["north"].to_numpy()
    up = df["up"].to_numpy()

    if show_3d:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(east, north, up, label="Trajectory")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_zlabel("Altitude / Up (m)")
        ax.set_title("3D Trajectory (East, North, Up)")
        ax.legend()
        ax.grid(True)
        plt.show()

    if show_2d:
        # Ground track + altitude vs slant-range
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Ground track (East vs North)
        axes[0].plot(east, north)
        axes[0].set_xlabel("East (m)")
        axes[0].set_ylabel("North (m)")
        axes[0].set_title("Ground Track (East vs North)")
        axes[0].grid(True)
        axes[0].axis('equal')

        # Altitude vs horizontal range (slant along ground)
        # compute cumulative ground-range for x-axis (distance along ground)
        ground_range = ( (east - east[0])**2 + (north - north[0])**2 )**0.5
        axes[1].plot(ground_range, up)
        axes[1].set_xlabel("Ground range (m)")
        axes[1].set_ylabel("Altitude (m)")
        axes[1].set_title("Altitude vs Ground Range")
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
