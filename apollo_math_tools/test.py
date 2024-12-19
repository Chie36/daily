import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def rotate_point(x, y, angle):
    """Rotate a point (x, y) by 'angle' degrees counter-clockwise around the origin."""
    angle_rad = np.radians(angle)
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return x_rot, y_rot


def plot_rotated_rectangle(ax, center, width, height, angle, label):
    """Plot a rotated rectangle centered at 'center' with given 'width', 'height' and 'angle'."""
    # Half dimensions
    half_w = width / 2
    half_h = height / 2

    # Define the four corners of the rectangle relative to the center
    corners = np.array([[-half_w, -half_h],
                        [half_w, -half_h],
                        [half_w, half_h],
                        [-half_w, half_h]])

    # Rotate each corner
    rotated_corners = np.array([rotate_point(x, y, angle) for x, y in corners])

    # Shift the corners by the center coordinates
    rotated_corners += center

    # Create a Polygon for the rectangle
    rect = Polygon(rotated_corners, closed=True, fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(rect)

    # Label the rectangle
    ax.text(center[0], center[1], label, color='blue', ha='center', va='center')

    return rotated_corners


def plot_projection(ax, rect1_center, rect2_center, angle1, angle2, width1, height1, width2, height2):
    """Plot the projections of one rectangle's edges onto the other rectangle's coordinate system."""
    # Plot the first rectangle (Rect 1)
    rect1_corners = plot_rotated_rectangle(ax, rect1_center, width1, height1, angle1, "Rect 1")

    # Plot the second rectangle (Rect 2)
    rect2_corners = plot_rotated_rectangle(ax, rect2_center, width2, height2, angle2, "Rect 2")

    # Mark dx, dy, a, b, c, d for the first rectangle
    for i, corner in enumerate(rect1_corners):
        ax.annotate(f'dx{i + 1}, dy{i + 1}', corner, textcoords="offset points", xytext=(5, 5), ha='center',
                    color='red')

    # Mark q, w, e, r for the projection bounds (example for Rect1's projection on Rect2's axes)
    ax.text(rect1_center[0] + 2, rect1_center[1] + 3, f'q', color='green', ha='center', va='center')
    ax.text(rect1_center[0] + 2, rect1_center[1] - 3, f'w', color='green', ha='center', va='center')

    # Mark t, u, i, o for another set of projections (example for Rect2)
    ax.text(rect2_center[0] + 2, rect2_center[1] + 3, f't', color='purple', ha='center', va='center')
    ax.text(rect2_center[0] + 2, rect2_center[1] - 3, f'u', color='purple', ha='center', va='center')

    # Additional projection bounds
    ax.text(rect1_center[0] - 2, rect1_center[1], f'a', color='orange', ha='center', va='center')
    ax.text(rect2_center[0] - 2, rect2_center[1], f'b', color='orange', ha='center', va='center')
    ax.text(rect1_center[0] + 3, rect1_center[1], f'c', color='blue', ha='center', va='center')
    ax.text(rect2_center[0] + 3, rect2_center[1], f'd', color='blue', ha='center', va='center')

    # Add axis lines for visualization of coordinate systems
    ax.plot([rect1_center[0] - 1, rect1_center[0] + 1], [rect1_center[1], rect1_center[1]], 'r--',
            label="Rect 1 X-axis")
    ax.plot([rect1_center[0], rect1_center[0]], [rect1_center[1] - 1, rect1_center[1] + 1], 'g--',
            label="Rect 1 Y-axis")
    ax.plot([rect2_center[0] - 1, rect2_center[0] + 1], [rect2_center[1], rect2_center[1]], 'b--',
            label="Rect 2 X-axis")
    ax.plot([rect2_center[0], rect2_center[0]], [rect2_center[1] - 1, rect2_center[1] + 1], 'y--',
            label="Rect 2 Y-axis")


# Set up the figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')

# Coordinates and dimensions of two rotated rectangles
rect1_center = (2, 3)
rect2_center = (4, 5)
width1, height1 = 4, 2
width2, height2 = 3, 5
angle1 = 30  # Angle for Rect 1
angle2 = -45  # Angle for Rect 2

# Plot the rectangles and projections
plot_projection(ax, rect1_center, rect2_center, angle1, angle2, width1, height1, width2, height2)

# Label axes and grid
ax.set_xlim(-5, 10)
ax.set_ylim(-5, 10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.grid(True)

# Show the plot
plt.legend()
plt.show()
