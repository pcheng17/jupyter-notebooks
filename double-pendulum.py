import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

# Constants
g = 9.81
l1, l2 = 1.0, 0.5
m1, m2 = 1.0, 1.0

# Time settings
fps = 240
dt = 1/fps
T = 30 
time = np.arange(0, T, dt)
timesteps = len(time)
  
# Number of segments to show, adjust as needed
max_path_length = 800

# Derivative function for ODE
def derivatives(t1, t2, o1, o2):
    delta_theta = t1 - t2
    a1 = (-g*(2*m1+m2)*np.sin(t1) - m2*g*np.sin(t1-2*t2) - 2*np.sin(t1-t2)*m2*(o2**2*l2 + o1**2*l1*np.cos(delta_theta)))/(l1*(2*m1+m2-m2*np.cos(2*delta_theta)))
    a2 = (2*np.sin(delta_theta)*(o1**2*l1*(m1+m2) + g*(m1+m2)*np.cos(t1) + o2**2*l2*m2*np.cos(delta_theta)))/(l2*(2*m1+m2-m2*np.cos(2*delta_theta)))
    return o1, o2, a1, a2

def rk4(t1, t2, o1, o2):
    k1_t1, k1_t2, k1_o1, k1_o2 = derivatives(t1, t2, o1, o2)
    k2_t1, k2_t2, k2_o1, k2_o2 = derivatives(t1+0.5*dt*k1_t1, t2+0.5*dt*k1_t2, o1+0.5*dt*k1_o1, o2+0.5*dt*k1_o2)
    k3_t1, k3_t2, k3_o1, k3_o2 = derivatives(t1+0.5*dt*k2_t1, t2+0.5*dt*k2_t2, o1+0.5*dt*k2_o1, o2+0.5*dt*k2_o2)
    k4_t1, k4_t2, k4_o1, k4_o2 = derivatives(t1+dt*k3_t1, t2+dt*k3_t2, o1+dt*k3_o1, o2+dt*k3_o2)

    t1_new = t1 + (1/6)*(k1_t1 + 2*k2_t1 + 2*k3_t1 + k4_t1)*dt
    t2_new = t2 + (1/6)*(k1_t2 + 2*k2_t2 + 2*k3_t2 + k4_t2)*dt
    o1_new = o1 + (1/6)*(k1_o1 + 2*k2_o1 + 2*k3_o1 + k4_o1)*dt
    o2_new = o2 + (1/6)*(k1_o2 + 2*k2_o2 + 2*k3_o2 + k4_o2)*dt

    return t1_new, t2_new, o1_new, o2_new
    
def solve(t1_0, t2_0, n):
    t1 = np.zeros(n)
    t2 = np.zeros(n)
    o1 = np.zeros(n)
    o2 = np.zeros(n)
    
    t1[0] = t1_0
    t2[0] = t2_0

    for i in range(1, timesteps):
        t1[i], t2[i], o1[i], o2[i] = rk4(t1[i-1], t2[i-1], o1[i-1], o2[i-1])

    return t1, t2, o1, o2

def get_positions(t1, t2):
    x1 = l1 * np.sin(t1)
    y1 = -l1 * np.cos(t1)
    
    x2 = x1 + l2 * np.sin(t2)
    y2 = y1 - l2 * np.cos(t2)
    
    return x1, y1, x2, y2

def main(initial_conditions, colors):
    assert(len(initial_conditions) == len(colors))
    solutions = [solve(t1_0, t2_0, timesteps) for t1_0, t2_0 in initial_conditions]

    fig, ax = plt.subplots()
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 1.75])
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    
    lines = []
    paths = []
    path_segs = []
    bobs1 = []
    bobs2 = []

    for (t1, t2, _, _), c in zip(solutions, colors):
        x1, y1, x2, y2 = get_positions(t1[0], t2[0])
        
        line, = ax.plot([0, x1, x2], [0, y1, y2], color='slategrey', lw=2, alpha=0.5)
        path, = ax.plot([], [], linewidth=1, color=c, alpha=0.9)

        bob1 = plt.Circle((x1, y1), 0.08, fc='b', alpha=0.0)
        bob2 = plt.Circle((x2, y2), 0.08, fc='g', alpha=0.0)

        ax.add_patch(bob1)
        ax.add_patch(bob2)
        
        lines.append(line)
        paths.append(path)
        path_segs.append([(x2, y2)])
        bobs1.append(bob1)
        bobs2.append(bob2)

    def init():
        for line in lines:
            line.set_data([], [])
        for path in paths:
            path.set_data([], [])
        for bob1, bob2 in zip(bobs1, bobs2):
            ax.add_patch(bob1)
            ax.add_patch(bob2)
        return lines + paths + bobs1 + bobs2

    def animate(i):
        for (line, path, path_seg, bob1, bob2, solution) in zip(lines, paths, path_segs, bobs1, bobs2, solutions):
            t1, t2, _, _ = solution
            x1, y1, x2, y2 = get_positions(t1[i], t2[i])

            line.set_data([0, x1, x2], [0, y1, y2])
            bob1.center = (x1, y1)
            bob2.center = (x2, y2)

            path_seg.append((x2, y2))
            if max_path_length is not None:
                path.set_data(*zip(*path_seg[-max_path_length:]))
            else:
                path.set_data(*zip(*path_seg))

        return lines + paths + bobs1 + bobs2

    anim = animation.FuncAnimation(fig, animate, frames=timesteps, init_func=init, blit=True, interval=dt*1000)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    anim.save('double-pendulum.mp4', writer=writer, dpi=300)  # Change dpi to set the resolution

# Call the function with initial conditions and colors
initial_conditions = [
    (np.pi/2, np.pi - 0.0000),
    (np.pi/2, np.pi - 0.0001),
    (np.pi/2, np.pi - 0.0002),
    (np.pi/2, np.pi - 0.0003),
    (np.pi/2, np.pi - 0.0004),
    (np.pi/2, np.pi - 0.0005),
    (np.pi/2, np.pi - 0.0006),
]
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
main(initial_conditions, colors)
