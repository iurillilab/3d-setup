"""Generate publication-quality figure explaining triangulation geometry."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def normalize(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def closest_point_on_line(C, d, x):
    t = np.dot(d, (x - C))
    return C + t * d


def main(save_path=None, dpi=300):
    # --- Geometry ---
    P_true = np.array([0.20, 0.10, 0.35])
    C1 = np.array([-0.60, -0.20, 0.15])
    C2 = np.array([0.70, -0.10, 0.20])
    C3 = np.array([0.00, 0.85, 0.25])

    d1 = normalize((P_true - C1) + np.array([0.00, 0.01, -0.005]))
    d2 = normalize((P_true - C2) + np.array([-0.01, 0.00, 0.006]))
    d3 = normalize((P_true - C3) + np.array([0.008, -0.01, 0.00]))

    Cs = np.stack([C1, C2, C3], axis=0)
    ds = np.stack([d1, d2, d3], axis=0)

    # Least-squares triangulation
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for C, d in zip(Cs, ds):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ C
    P_est = np.linalg.solve(A, b)

    rays = [
        C[None, :] + np.linspace(0, 1.6, 50)[:, None] * d[None, :]
        for C, d in zip(Cs, ds)
    ]

    # --- Plotting ---
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11})
    fig = plt.figure(figsize=(10, 5))
    ax2d = fig.add_subplot(121)
    ax3d = fig.add_subplot(122, projection="3d")

    colors = ["#0173b2", "#de8f05", "#029e73"]  # Tol palette

    # --- Panel A: 2D principle (xy cross-section) ---
    # Two cameras in 2D
    c1_2d, c2_2d = C1[:2], C2[:2]
    r1_2d = normalize((P_true[:2] - c1_2d) + np.array([0.01, -0.005]))
    r2_2d = normalize((P_true[:2] - c2_2d) + np.array([-0.01, 0.006]))
    tmax = 1.2
    ax2d.plot(
        [c1_2d[0], c1_2d[0] + tmax * r1_2d[0]],
        [c1_2d[1], c1_2d[1] + tmax * r1_2d[1]],
        color=colors[0], lw=2.5, label=r"Back-projected ray 1"
    )
    ax2d.plot(
        [c2_2d[0], c2_2d[0] + tmax * r2_2d[0]],
        [c2_2d[1], c2_2d[1] + tmax * r2_2d[1]],
        color=colors[1], lw=2.5, label=r"Back-projected ray 2"
    )
    ax2d.scatter(*c1_2d, s=120, marker="^", c=colors[0], edgecolor="k", linewidth=0.8, zorder=5)
    ax2d.scatter(*c2_2d, s=120, marker="^", c=colors[1], edgecolor="k", linewidth=0.8, zorder=5)
    ax2d.scatter(*P_true[:2], s=80, c="k", edgecolor="w", linewidth=1, zorder=5, label=r"3D point $\mathbf{P}$")
    ax2d.annotate(r"$\mathbf{C}_1$", c1_2d - 0.12, fontsize=11, fontweight="bold")
    ax2d.annotate(r"$\mathbf{C}_2$", c2_2d + np.array([0.08, 0]), fontsize=11, fontweight="bold")
    ax2d.annotate(r"$\mathbf{P}$", P_true[:2] + 0.08, fontsize=11, fontweight="bold")
    ax2d.set_xlabel(r"$x$ (m)")
    ax2d.set_ylabel(r"$y$ (m)")
    ax2d.set_aspect("equal")
    ax2d.legend(loc="lower right", fontsize=9)
    ax2d.grid(True, alpha=0.3)
    ax2d.set_title(r"(a) Principle: intersection of rays")
    ax2d.set_xlim(-0.9, 0.9)
    ax2d.set_ylim(-0.5, 0.5)

    # --- Panel B: 3D multi-camera setup ---

    # Floor grid (arena)
    xg = np.linspace(-0.8, 0.8, 9)
    yg = np.linspace(-0.4, 0.95, 6)
    for x in xg:
        ax3d.plot([x, x], [yg[0], yg[-1]], [0, 0], "gray", lw=0.5, alpha=0.4)
    for y in yg:
        ax3d.plot([xg[0], xg[-1]], [y, y], [0, 0], "gray", lw=0.5, alpha=0.4)

    # Rays
    for i, (ray, col) in enumerate(zip(rays, colors)):
        ax3d.plot(ray[:, 0], ray[:, 1], ray[:, 2], color=col, lw=2, alpha=0.9)

    # Residuals
    for C, d, col in zip(Cs, ds, colors):
        q = closest_point_on_line(C, d, P_est)
        ax3d.plot(
            [P_est[0], q[0]], [P_est[1], q[1]], [P_est[2], q[2]],
            "--", color=col, lw=1, alpha=0.6
        )

    ax3d.scatter(
        Cs[:, 0], Cs[:, 1], Cs[:, 2],
        s=100, marker="^", c=colors, edgecolors="black", linewidths=0.8, zorder=5
    )
    ax3d.scatter(
        *P_true, s=80, c="black", edgecolors="white", linewidths=1,
        marker="o", zorder=5, label=r"True $\mathbf{P}$"
    )
    ax3d.scatter(
        *P_est, s=120, c="crimson", marker="x", linewidths=3, zorder=5,
        label=r"Estimate $\hat{\mathbf{P}}$"
    )

    for i, C in enumerate(Cs):
        ax3d.text(C[0], C[1], C[2] + 0.05, rf"$\mathbf{{C}}_{{i+1}}$", fontsize=10, fontweight="bold")

    ax3d.set_xlabel(r"$x$ (m)")
    ax3d.set_ylabel(r"$y$ (m)")
    ax3d.set_zlabel(r"$z$ (m)")
    ax3d.set_title(r"(b) Multi-camera triangulation (3 views)")
    ax3d.view_init(elev=22, azim=-60)
    ax3d.set_box_aspect([1.6, 1.4, 1])
    ax3d.legend(loc="upper left", fontsize=9, framealpha=0.95)
    for p in ["x", "y", "z"]:
        getattr(ax3d, f"{p}axis").pane.fill = False
    ax3d.grid(True, alpha=0.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    return P_true, P_est


if __name__ == "__main__":
    P_true, P_est = main(save_path="triangulation_schematic.pdf")
    print("P_true:", P_true)
    print("P_est :", P_est)
