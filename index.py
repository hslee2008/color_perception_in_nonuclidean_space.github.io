# Improved 3D chromaticity surface
# Now: **less-perceived (hard-to-discriminate) colors rise up**, while highly perceived regions stay low.
# - Height H = +alpha * (1 - S)^gamma  (S = relative sensitivity)
# - Face colors = true sRGB converted from (x,y) chromaticity
# - Same Top/Side/Iso camera buttons and sliders for alpha/gamma
#
# If S_base, X, Y, and the xy->sRGB converter exist from earlier, we reuse them. Otherwise we rebuild.

import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib as mpl

# CIE를 RGB로 변환하기
def xyY_to_sRGB(x, y, Y=1.0):
    if y == 0:
        return np.zeros(3)
      
    Xv = (x * Y) / y
    Zv = ((1 - x - y) * Y) / y
    M = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689,  1.8758,  0.0415],
                  [0.0557, -0.2040,  1.0570]])
    rgb = M @ np.array([Xv, Y, Zv])
    rgb = np.clip(rgb, 0, None)
    mask = rgb <= 0.0031308
    rgb[mask] *= 12.92
    rgb[~mask] = 1.055 * (rgb[~mask] ** (1 / 2.4)) - 0.055
    
    return np.clip(rgb, 0, 1)

# 2D Gaussian function for synthetic sensitivity peaks
def gaussian2d(X, Y, xc, yc, sx, sy, theta):
    ct, st = np.cos(theta), np.sin(theta)
    Xc, Yc = X - xc, Y - yc
    Xr = ct*Xc + st*Yc
    Yr = -st*Xc + ct*Yc
    
    return np.exp(-0.5*((Xr/sx)**2 + (Yr/sy)**2))


# ---------- Rebuild domain + sensitivity field if needed ----------
rebuild = False
try:
    S_base
    X
    Y
except NameError:
    rebuild = True

if rebuild:
    xmin, xmax, ymin, ymax = 0.10, 0.70, 0.10, 0.70
    N = 220
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    X, Y = np.meshgrid(x, y)

    # Approximate sRGB triangle mask to avoid unrealistic corners
    sRGB_triangle = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]])

    def point_in_triangle(px, py, tri):
        (x1, y1), (x2, y2), (x3, y3) = tri
        den = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
        w1 = ((y2 - y3)*(px - x3) + (x3 - x2)*(py - y3)) / den
        w2 = ((y3 - y1)*(px - x3) + (x1 - x3)*(py - y3)) / den
        w3 = 1 - w1 - w2
        return (w1 >= 0) & (w2 >= 0) & (w3 >= 0)
    mask = point_in_triangle(X, Y, sRGB_triangle)

    # Synthetic peaks (you can replace by MacAdam ellipse-based strengths)
    centers = [
        (0.33, 0.56, 0.020, 0.015, 20*np.pi/180, 1.00),
        (0.36, 0.54, 0.022, 0.014, -25*np.pi/180, 0.95),
        (0.40, 0.52, 0.020, 0.016, 10*np.pi/180, 0.90),
        (0.42, 0.50, 0.018, 0.015, -15*np.pi/180, 0.85),
        (0.54, 0.40, 0.030, 0.018, 20*np.pi/180, 0.55),
        (0.60, 0.34, 0.028, 0.022, -10*np.pi/180, 0.50),
        (0.22, 0.35, 0.026, 0.022, 25*np.pi/180, 0.60),
        (0.17, 0.28, 0.024, 0.020, -15*np.pi/180, 0.55),
        (0.15, 0.20, 0.022, 0.022, 10*np.pi/180, 0.50),
    ]
    rng = np.random.default_rng(7)
    for k in range(18):
        xc = float(rng.uniform(0.12, 0.65))
        yc = float(rng.uniform(0.12, 0.65))
        sx = float(rng.uniform(0.012, 0.028))
        sy = float(rng.uniform(0.012, 0.028))
        th = float(rng.uniform(0, 2*np.pi))
        w = float(rng.uniform(0.25, 0.45))
        centers.append((xc, yc, sx, sy, th, w))

    S = np.zeros_like(X)
    for (xc, yc, sx, sy, th, w) in centers:
        S += w * gaussian2d(X, Y, xc, yc, sx, sy, th)
    S -= np.nanmin(S)
    S /= np.nanmax(S)
    S_base = np.where(mask, S, np.nan)

# ---------- Map to height: 색깔 구분 능력 떨어짐 rises up ----------
alpha0, gamma0 = 1.0, 1.3


def height_from_S(S, alpha, gamma):
    # S = sensitivity in [0,1]; we want height high when sensitivity is low.
    return alpha * np.power(1 - S, gamma)


Z = height_from_S(S_base, alpha0, gamma0)

# ---------- Face colors from true chromaticity (x,y) ----------
colors = np.zeros((*X.shape, 3))
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        colors[i, j] = xyY_to_sRGB(X[i, j], Y[i, j], Y=1.0)
# Transparent outside mask
alpha_face = np.where(np.isfinite(S_base), 1.0, 0.0)

# ---------- Plot ----------
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    X, Y, Z,
    facecolors=np.dstack([colors, alpha_face]),
    linewidth=0, antialiased=True, shade=False
)

# Outline approximate sRGB triangle for reference
sRGB_triangle = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]])
triX = np.append(sRGB_triangle[:, 0], sRGB_triangle[0, 0])
triY = np.append(sRGB_triangle[:, 1], sRGB_triangle[0, 1])
ax.plot(triX, triY, np.nanmax(Z[np.isfinite(Z)])
        * 0 + 0.02, linewidth=2.0, color="k", alpha=0.5)

ax.set_title("3D Chromaticity Surface — **Less-perceived colors rise up**")
ax.set_xlabel("CIE x")
ax.set_ylabel("CIE y")
ax.set_zlabel("색깔 구분 능력 떨어짐 ↑")
ax.set_xlim(np.nanmin(X), np.nanmax(X))
ax.set_ylim(np.nanmin(Y), np.nanmax(Y))
ax.set_zlim(0.0, 1.25)
ax.set_box_aspect((1, 1, 0.6))

# ---------- Sliders for alpha, gamma ----------
ax_alpha = fig.add_axes([0.15, 0.05, 0.50, 0.03])
ax_gamma = fig.add_axes([0.15, 0.01, 0.50, 0.03])
s_alpha = Slider(ax=ax_alpha, label="Height scale α",
                 valmin=0.0, valmax=2.0, valinit=alpha0, valstep=0.02)
s_gamma = Slider(ax=ax_gamma, label="Nonlinearity γ",
                 valmin=0.5, valmax=2.5, valinit=gamma0, valstep=0.05)


def on_change(val):
    global surf
    a = s_alpha.val
    g = s_gamma.val
    Z2 = height_from_S(S_base, a, g)
    surf.remove()
    surf = ax.plot_surface(
        X, Y, Z2,
        facecolors=np.dstack([colors, alpha_face]),
        linewidth=0, antialiased=True, shade=False
    )
    fig.canvas.draw_idle()


s_alpha.on_changed(on_change)
s_gamma.on_changed(on_change)

# ---------- View buttons ----------
btn_top_ax = fig.add_axes([0.70, 0.045, 0.08, 0.045])
btn_side_ax = fig.add_axes([0.79, 0.045, 0.08, 0.045])
btn_iso_ax = fig.add_axes([0.88, 0.045, 0.08, 0.045])
btn_top = Button(btn_top_ax,  "Top")
btn_side = Button(btn_side_ax, "Side")
btn_iso = Button(btn_iso_ax,  "Iso")


def set_top(event):  ax.view_init(elev=90, azim=-90); fig.canvas.draw_idle()
def set_side(event): ax.view_init(elev=0,  azim=-90); fig.canvas.draw_idle()
def set_iso(event):  ax.view_init(elev=28, azim=45);  fig.canvas.draw_idle()


btn_top.on_clicked(set_top)
btn_side.on_clicked(set_side)
btn_iso.on_clicked(set_iso)

plt.show()

# Export an interactive HTML where the surface is colored by the *actual hue* (sRGB from CIE x,y),
# not by perceptual sensitivity. We use Plotly's Mesh3d since it supports per-vertex colors.


# ---------- Rebuild grid and Z if not present ----------
try:
    X, Y, S_base
except NameError:
    xmin, xmax, ymin, ymax = 0.10, 0.70, 0.10, 0.70
    N = 200
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    X, Y = np.meshgrid(x, y)

    # Simple synthetic "sensitivity" field just to define Z; replace with real data if available
    def gaussian2d(X, Y, xc, yc, sx, sy, theta):
        ct, st = np.cos(theta), np.sin(theta)
        Xc, Yc = X - xc, Y - yc
        Xr = ct*Xc + st*Yc
        Yr = -st*Xc + ct*Yc
        return np.exp(-0.5*((Xr/sx)**2 + (Yr/sy)**2))

    centers = [
        (0.33, 0.56, 0.020, 0.015, 0.0, 1.0),
        (0.40, 0.52, 0.020, 0.016, 0.2, 0.9),
        (0.22, 0.35, 0.026, 0.022, 0.5, 0.6),
        (0.60, 0.34, 0.028, 0.022, -0.1, 0.5),
    ]
    S = np.zeros_like(X)
    for (xc, yc, sx, sy, th, w) in centers:
        S += w * gaussian2d(X, Y, xc, yc, sx, sy, th)
    S -= S.min()
    S /= S.max()
    S_base = S

# Height: poor-discrimination up (you can swap to your preferred mapping)
alpha0, gamma0 = 1.0, 1.3
Z = (1 - S_base)**gamma0 * alpha0

# ---------- Convert (x,y) to sRGB for vertex colors ----------


def xyY_to_sRGB(x, y, Yval=1.0):
    if y == 0:
        return np.array([0.0, 0.0, 0.0])
    Xv = (x * Yval) / y
    Zv = ((1 - x - y) * Yval) / y
    XYZ = np.array([Xv, Yval, Zv])
    M = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689,  1.8758,  0.0415],
                  [0.0557, -0.2040,  1.0570]])
    rgb = M @ XYZ
    rgb = np.clip(rgb, 0, None)
    mask = rgb <= 0.0031308
    rgb[mask] *= 12.92
    rgb[~mask] = 1.055 * (rgb[~mask] ** (1/2.4)) - 0.055
    return np.clip(rgb, 0, 1)


# Build per-vertex colors
nrows, ncols = X.shape
x_flat = X.ravel()
y_flat = Y.ravel()
z_flat = Z.ravel()

vertex_colors = []
for xv, yv in zip(x_flat, y_flat):
    srgb = xyY_to_sRGB(xv, yv, Yval=1.0)
    # Convert to 0-255 and to 'rgb(r,g,b)' string that Mesh3d accepts
    r, g, b = (srgb * 255).astype(int)
    vertex_colors.append(f"rgb({r},{g},{b})")

# ---------- Build triangulation (i,j,k) ----------
i_idx = []
j_idx = []
k_idx = []
for r in range(nrows - 1):
    for c in range(ncols - 1):
        v0 = r * ncols + c
        v1 = v0 + 1
        v2 = v0 + ncols
        v3 = v2 + 1
        # two triangles per quad
        i_idx.extend([v0, v1])
        j_idx.extend([v1, v3])
        k_idx.extend([v2, v2])

mesh = go.Mesh3d(
    x=x_flat, y=y_flat, z=z_flat,
    i=i_idx, j=j_idx, k=k_idx,
    vertexcolor=vertex_colors,
    flatshading=True,
    lighting=dict(ambient=0.9, diffuse=0.2, specular=0.2),
    name="Chromaticity surface"
)

fig = go.Figure(data=[mesh])
fig.update_layout(
    title="색 인지도 능력",
    scene=dict(
        xaxis_title='CIE x',
        yaxis_title='CIE y',
        zaxis_title='색깔 구분 능력 떨어짐 ↑',
        aspectmode='cube'
    ),
    margin=dict(l=0, r=0, t=40, b=0)
)

out_path = "./chromaticity_surface_hue.html"
fig.write_html(out_path, include_plotlyjs='cdn')
out_path
