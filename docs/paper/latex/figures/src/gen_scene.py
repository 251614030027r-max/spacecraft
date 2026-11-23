"""Generate scene_body.tex (included by fig_scene.tex): oblique 3-D schematic of the pre-capture scene.
All geometry follows docs/paper/facts/01 (port, capture point, cone, sphere,
target box); the chaser position is illustrative. Output: 2-D TikZ coordinates."""
import numpy as np, sys

S = 0.41                       # cm per meter
v = np.array([-0.40, -0.84, 0.38]); v /= np.linalg.norm(v)          # toward viewer
right = np.cross([0, 0, 1], v); right /= np.linalg.norm(right)
up = np.cross(v, right)
P = lambda p: (S * float(np.dot(p, right)), S * float(np.dot(p, up)))
f = lambda xy: f"({xy[0]:.3f},{xy[1]:.3f})"
out = []
w = out.append

def circle3(c, n, rho, k=180):
    n = np.asarray(n, float); n /= np.linalg.norm(n)
    a = np.cross(n, [0, 0, 1]) if abs(n[2]) < .9 else np.cross(n, [0, 1, 0]); a /= np.linalg.norm(a)
    b = np.cross(n, a)
    th = np.linspace(0, 2*np.pi, k, endpoint=False)
    return [np.asarray(c) + rho*(np.cos(t)*a + np.sin(t)*b) for t in th], a, b

def cone(apex, axis, half, length):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    c = np.asarray(apex) + length*axis; rho = length*np.tan(np.radians(half))
    pts, a, b = circle3(c, axis, rho, 720)
    th = np.linspace(0, 2*np.pi, 720, endpoint=False)
    A = np.array(P(apex))
    vals = []
    for t, p in zip(th, pts):
        d = np.array(P(p)) - A
        dp = np.array(P(rho*(-np.sin(t)*a + np.cos(t)*b))) - np.array(P(np.zeros(3)))
        vals.append(d[0]*dp[1] - d[1]*dp[0])
    vals = np.array(vals)
    idx = [i for i in range(len(vals)) if np.sign(vals[i]) != np.sign(vals[(i+1) % len(vals)])]
    sil = [pts[i] for i in idx[:2]]
    return pts, sil, c

# ---------------- target box with grapple rod
hx, hy, hz = 0.62*1.7, 0.51*1.7, 0.37*1.7   # box enlarged for legibility (schematic)
V = {(i, j, k): np.array([i*hx, j*hy, k*hz]) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)}
faces = [((1, 0, 0), [(1,-1,-1),(1,1,-1),(1,1,1),(1,-1,1)]), ((-1, 0, 0), [(-1,-1,-1),(-1,-1,1),(-1,1,1),(-1,1,-1)]),
         ((0, 1, 0), [(-1,1,-1),(-1,1,1),(1,1,1),(1,1,-1)]), ((0, -1, 0), [(-1,-1,-1),(1,-1,-1),(1,-1,1),(-1,-1,1)]),
         ((0, 0, 1), [(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]), ((0, 0, -1), [(-1,-1,-1),(-1,1,-1),(1,1,-1),(1,-1,-1)])]
shade = {0: 'gray!22', 1: 'gray!35', 2: 'gray!12'}

# ---------------- approach cone (apex at the port, axis n_p = -x, half-angle 35 deg)
port = np.array([-1.5, 0, 0]); cap = np.array([-3.0, 0, 0])
cpts, csil, cbase = cone(port, [-1, 0, 0], 35, 5.2)
w(r"\begin{tikzpicture}[>=Stealth,font=\footnotesize,line cap=round]")
# operation sphere (silhouette) and equator
w(rf"\draw[gray!70,densely dashed] (0,0) circle ({6*S:.3f});")
eq, _, _ = circle3([0, 0, 0], [0, 0, 1], 6.0, 180)
w(r"\draw[gray!45,densely dotted] " + " -- ".join(f(P(p)) for p in eq) + " -- cycle;")
# cone body
w(r"\fill[blue!9] " + f(P(port)) + " -- " + f(P(csil[0])) + " -- " + f(P(csil[1])) + " -- cycle;")
w(r"\fill[blue!9] " + " -- ".join(f(P(p)) for p in cpts) + " -- cycle;")
w(r"\draw[blue!55] " + " -- ".join(f(P(p)) for p in cpts) + " -- cycle;")
for s in csil:
    w(r"\draw[blue!55] " + f(P(port)) + " -- " + f(P(s)) + ";")
# mouth region: band of the cone between the operation sphere and the sphere of radius rho_s + l_m
def cone_sphere(rho, k=240):
    pts = []
    _, a1, b1 = circle3(port, [-1, 0, 0], 1.0, 4)
    for th in np.linspace(0, 2*np.pi, k, endpoint=False):
        d = np.cos(np.radians(35))*np.array([-1.0, 0, 0]) + np.sin(np.radians(35))*(np.cos(th)*a1 + np.sin(th)*b1)
        pb = float(np.dot(port, d)); tt = -pb + np.sqrt(pb*pb - (np.dot(port, port) - rho*rho))
        pts.append(port + tt*d)
    return pts
m0, m1 = cone_sphere(6.0), cone_sphere(6.5)
w(r"\fill[blue!22,even odd rule] " + " -- ".join(f(P(p)) for p in m1) + " -- cycle " + " -- ".join(f(P(p)) for p in m0) + " -- cycle;")
w(r"\draw[blue!60,thin] " + " -- ".join(f(P(p)) for p in m0) + " -- cycle;")
w(r"\draw[blue!60,thin] " + " -- ".join(f(P(p)) for p in m1) + " -- cycle;")
# half-angle of the cone: upper generator and arc at the port
gen = np.cos(np.radians(35))*np.array([-1.0, 0, 0]) + np.sin(np.radians(35))*np.array([0, 0, -1.0])
arc = [port + 3.0*(np.cos(np.radians(s))*np.array([-1.0, 0, 0]) + np.sin(np.radians(s))*np.array([0, 0, -1.0])) for s in np.linspace(0, 35, 20)]
w(r"\draw[blue!70!black,densely dashed] " + f(P(port)) + " -- " + f(P(port + 5.4*gen)) + ";")
w(r"\draw[blue!70!black] " + " -- ".join(f(P(p)) for p in arc) + ";")
ax_, ay_ = P(port + 3.45*(np.cos(np.radians(19))*np.array([-1.0, 0, 0]) + np.sin(np.radians(20))*np.array([0, 0, -1.0])))
w(rf"\node[blue!70!black,inner sep=1pt] at ({ax_:.3f},{ay_:.3f}) {{$\vartheta_{{\mathrm{{c}}}}$}};")
# port axis and points
w(r"\draw[->,thick,blue!70!black] " + f(P(port)) + " -- " + f(P(port + np.array([-2.9, 0, 0]))) + r" node[anchor=south east,inner sep=1pt] {$\bnp$};")
mouth = np.array([-6.3, 0, 0])
w(r"\fill[blue!70!black] " + f(P(mouth)) + " circle (1.6pt);")
# target box
for n, quad in faces:
    if np.dot(n, v) > 0:
        k = 0 if n[2] else (1 if n[0] else 2)
        w(rf"\filldraw[fill={shade[k]},draw=black,line join=round] " + " -- ".join(f(P(V[q])) for q in quad) + " -- cycle;")
w(r"\draw[line width=1.2pt] " + f(P(np.array([-hx, 0, 0]))) + " -- " + f(P(port)) + ";")
w(r"\fill " + f(P(port)) + " circle (1.5pt);")
w(r"\fill[red!75!black] " + f(P(cap)) + " circle (1.7pt);")
# target body axes and spin axis
O = np.zeros(3)
w(r"\draw[->,thick] " + f(P(np.array([hx, 0, 0]))) + " -- " + f(P(np.array([2.3, 0, 0]))) + r" node[anchor=west,inner sep=1pt] {$x_{\mathrm{T}}$};")
w(r"\draw[->,thick] " + f(P(np.array([0, 0, hz]))) + " -- " + f(P(np.array([0, 0, 2.2]))) + r" node[anchor=east,inner sep=1pt] {$z_{\mathrm{T}}$};")
sh = np.array([np.sin(np.radians(20))*np.cos(np.radians(-25)), np.sin(np.radians(20))*np.sin(np.radians(-25)), np.cos(np.radians(20))])
w(r"\draw[->,thick,orange!85!black] " + f(P(sh*hz/sh[2])) + " -- " + f(P(3.4*sh)) + r" node[anchor=south west,inner sep=1pt] {$\bs$};")
ring, _, _ = circle3(2.4*sh, sh, 0.45, 60)
w(r"\draw[->,orange!85!black] " + " -- ".join(f(P(p)) for p in ring[5:52]) + ";")
# chaser with boresight and field-of-view cone
ch = np.array([1.5, -10.5, -2.6])
bs = port - ch; bs /= np.linalg.norm(bs)
cx, cy = P(ch); bx, by = np.array(P(ch + bs)) - np.array(P(ch)); ang = np.degrees(np.arctan2(by, bx))
w(rf"\fill[green!12] ({cx:.3f},{cy:.3f}) -- ++({ang-38:.1f}:1.35) arc[start angle={ang-38:.1f},end angle={ang+38:.1f},radius=1.35] -- cycle;")
w(rf"\draw[green!45!black] ({cx:.3f},{cy:.3f}) -- ++({ang-38:.1f}:1.35) ({cx:.3f},{cy:.3f}) -- ++({ang+38:.1f}:1.35);")
w(r"\draw[->,densely dotted,shorten >=4pt] " + f(P(O)) + " -- " + f(P(ch)) + ";")
w(r"\filldraw[fill=white,draw=black] " + f(P(O)) + " circle (1.3pt);")
w(r"\draw[->,thick] " + f(P(ch)) + " -- " + f(P(ch + 2.2*bs)) + r" node[anchor=south east,inner sep=1pt] {$\vect{e}_{\mathrm{C}}$};")
w(rf"\filldraw[fill=gray!25,draw=black,rotate around={{{ang:.1f}:({cx:.3f},{cy:.3f})}}] ({cx-0.17:.3f},{cy-0.13:.3f}) rectangle ({cx+0.17:.3f},{cy+0.13:.3f});")
# labels (screen coordinates relative to projected anchors)
def lab(p, txt, anchor, dx=0, dy=0, style=""):
    x, y = P(p); w(rf"\node[{style}anchor={anchor},inner sep=1pt,align=center] at ({x+dx:.3f},{y+dy:.3f}) {{{txt}}};")
lab(port, r"$\bpport$", "north", 0.12, -0.20)
lab(cap, r"$\bpcap$", "south", 0.0, 0.10, "red!75!black,")
lab(mouth, r"$\brhom$", "south", 0.05, 0.08, "blue!70!black,")
lab(cbase + np.array([0, 0, 4.9]), r"approach cone $\Kset$", "south", 0.0, 0.0, "blue!70!black,")
lab(m1[60], r"mouth region $\Km$\\(width $\ell_{\mathrm{m}}$)", "north", 0.0, -0.08, "blue!70!black,")
lab(np.array([0, 0, 0]), r"operation sphere\\(radius $\rhos$)", "south west", 1.45, 1.95, "gray!80!black,")
lab(O, r"$\FT$", "north west", 0.55, -0.42)
lab(ch, r"chaser, $\FC$", "north", 0.0, -0.14)
w(rf"\node[green!40!black,anchor=east,inner sep=1pt] at ({cx-1.15:.3f},{cy+0.15:.3f}) {{FOV}};")
lab((O + ch)/2, r"$\brho$", "north east", -0.02, 0.0)
# inertial frame (screen corner)
x0, y0 = P(np.array([-7.6, 3.0, -7.0]))
w(rf"\begin{{scope}}[shift={{({x0:.3f},{y0:.3f})}}]")
for e, name, anc in ((np.array([1, 0, 0]), r"X_{\mathrm{I}}", "west"), (np.array([0, 1, 0]), r"Y_{\mathrm{I}}", "east"), (np.array([0, 0, 1]), r"Z_{\mathrm{I}}", "south")):
    w(r"\draw[->] (0,0) -- " + f(P(2.0*e)) + rf" node[anchor={anc},inner sep=1pt] {{${name}$}};")
w(r"\node[anchor=north,inner sep=2pt] at (0,0) {$\FI$};")
w(r"\end{scope}")
w(r"\end{tikzpicture}")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
