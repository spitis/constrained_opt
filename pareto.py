from distutils.config import DEFAULT_PYPIRC
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

import jax
import jax.numpy as np
import numpy as onp

xmin, xmax, xstep = 0, 3, 0.1
ymin, ymax, ystep = 0, 3, 0.1

# initial parameter. roughly: (angle, radius)
INIT_POINT = (0.1, 1.7)

# Target loss
ALPHAS = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])

# def loss_concave(x, y):
#   r = 2 + (y**2)**0.6 # radius of loss
#   theta = jax.nn.sigmoid(x) * np.pi/2 # angle of loss
#   return r*np.cos(theta), r*np.sin(theta)

MOGUL_CENTERS = np.concatenate([
  np.linspace(np.array([0.7, -0.2]), np.array([-0.2, 0.7]), 9),
  np.linspace(np.array([0.9, -0.2]), np.array([-0.2, 0.9]), 12),
  np.linspace(np.array([1.1, -0.2]), np.array([-0.2, 1.1]), 13)
])

RADIUS = 0.045
DEFAULT_TARGET = np.array([0,0])

def loss_moguls(xy, target=DEFAULT_TARGET):


  base_loss = np.linalg.norm(xy - target, axis=-1)
  mogul_centers = MOGUL_CENTERS + (target / 3.)
  max_bonus = np.linalg.norm(mogul_centers - target, axis=1) / 3


  # Jagged bonus
  dists = np.linalg.norm(xy[None] - mogul_centers, axis=-1)
  dists = np.where(dists > RADIUS, 1000., dists)
  bonus = 0.1 / (dists + 0.1) # batch x num_jagged_points
  
  bonus *= max_bonus

  return base_loss - np.maximum(0., bonus.max())


def loss_convex(x, y):
  r = 2 + y**2 #radius of loss
  theta = jax.nn.sigmoid(x) * np.pi/2 # angle of loss
  vx, vy = np.cos(theta), np.sin(theta)
  norm = (vx**0.5 + vy**0.5)**2
  return 2.2 * r * vx / norm, 2.2 * r * vy / norm

def loss_concave(x, y):
  r = 1.5 + y**2 # radius of loss
  theta_prop = jax.nn.sigmoid(x) 
  theta = theta_prop * np.pi/2 # angle of loss

  vx, vy = np.cos(theta), np.sin(theta)
  
  r_old = r
  r = np.where(0.3 < theta_prop, np.where(theta_prop < 0.7, r+np.abs(theta_prop - 0.5)*2+0.2/(y**2 + 1e-4), r), r)

  norm = (vx**0.5 + vy**0.5)**2

  loss1 = 2.2 * r * vx / norm
  loss2 = 2.2 * r * vy / norm

  r = np.where(y**2 < 0.3, np.where(0.3 < theta_prop, np.where(theta_prop < 0.7, r_old, r), r), r)

  loss1 = 2.2 * r * vx / norm
  loss2 = 2.2 * r * vy / norm

  return loss1, loss2
  
def plot_paths(paths, titles=None):
    path_liness = []
    path_pointss = []
    num_row = len(paths) // 2 or 1
    num_col = (len(paths)+1) % 2 + 1
    fig, axes = plt.subplots(num_row, num_col, figsize = (2+num_col*5, 6*num_row))
    if len(paths) > 1:
      axes = list(axes.flat)
    else:
      axes = [axes]
    
    for ax, path in zip(axes, paths):
      path_lines, path_points = plot_loss(ax, path)
      path_liness.append(path_lines)
      path_pointss.append(path_points)
    
    if titles is not None:
      for ax, title in zip(axes, titles):
        ax.set_title(title)


    def init():
      res = []
      for path_lines, path_points, path in zip(path_liness, path_pointss, paths):
        for line, point, p in zip(path_lines, path_points, path):
          line.set_data([], [])
          point.set_data([], [])
          res.append(line)
          res.append(point)
      return res
      
    def animate(i):
      res = []
      for path_lines, path_points, path in zip(path_liness, path_pointss, paths):
        for line, point, p in zip(path_lines, path_points, path):
          pathx, pathy = p
          line.set_data(pathx[:i], pathy[:i])
          point.set_data(pathx[i-1:i], pathy[i-1:i])
          res.append(line)
          res.append(point)
      return res

    plt.tight_layout()
    plt.show()
      
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=paths[0].shape[-1], interval=30, 
                                  repeat_delay=5, blit=True)

    return fig, anim

def plot_loss(ax, paths=None):
    x, y = np.meshgrid(np.arange(-2, 2, 0.01), np.arange(-3, 3, 0.01))
    loss_vec = jax.jit(jax.vmap(loss_concave, in_axes=[0,0]))
    a , b = loss_vec(x, y)
    z = a + b
    #z = x + y
    ax.contourf(a, b, z, alpha=0.15, levels=np.linspace(0., 10., 100), cmap=plt.cm.jet, antialiased=True, extend='both')
    path_lines = []
    path_points = []
    if paths is not None:
      for path, alpha in zip(paths, ALPHAS):
        pathx, pathy = path
        ax.quiver(pathx[:-1], pathy[:-1], pathx[1:]-pathx[:-1], pathy[1:]-pathy[:-1], 
              scale_units='xy', angles='xy', scale=1, color='gray', alpha=0.75)
        
        line, = ax.plot([], [], '', label=alpha, lw=2, alpha=0.5, color='darkviolet')
        point, = ax.plot([], [], 'o', markersize=12, alpha=0.7, color='darkviolet')
        path_lines.append(line)
        path_points.append(point)

    ax.set_xlabel('$J_x$')
    ax.set_ylabel('$J_y$')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    return path_lines, path_points