import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm 
from mpl_toolkits.mplot3d import Axes3D

def get_sphere_surface(radius:float=1.0, color='lightblue', alpha=0.5, antialiased=True, plot=False)-> np.ndarray:
  phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
  x_blank_sphere = radius*np.sin(phi)*np.cos(theta)
  y_blank_sphere = radius*np.sin(phi)*np.sin(theta)
  z_blank_sphere = radius*np.cos(phi)
  sphere_surface = np.array(([x_blank_sphere,
                              y_blank_sphere,
                              z_blank_sphere]))
  if plot:
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(sphere_surface[0], sphere_surface[1], sphere_surface[2],
                    rstride=1, cstride=1, linewidth=0,
                    antialiased=antialiased, color=color, alpha=alpha)

    ax.set_aspect('auto')
    ax.view_init(elev=10)
    ax.set_xticks([-radius,0,radius])
    ax.set_yticks([-radius,0,radius])
    ax.set_zticks([-radius,0,radius])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Surface Plot: 2-sphere')
    plt.show()

  return sphere_surface

def get_cylinder_surface(radius:float=1.0, height:float=1.0, color='cyan', alpha=0.5, antialiased=True, plot=False)-> np.ndarray:
  p0 = np.array([0, 0, height]) #point at one end
  p1 = np.array([0, 0, -height]) #point at other end
  #vector in direction of axis
  v = p1 - p0
  #find magnitude of vector
  mag = np.linalg.norm(v)
  #unit vector in direction of axis
  v = v / mag
  #make some vector not in the same direction as v
  not_v = np.array([1, 0, 0])
  if (v == not_v).all():
      not_v = np.array([0, 1, 0])
  #make vector perpendicular to v
  n1 = np.cross(v, not_v)
  #normalize n1
  n1 /= np.linalg.norm(n1)
  #make unit vector perpendicular to v and n1
  n2 = np.cross(v, n1)
  #surface ranges over t from 0 to length of axis and 0 to 2*pi
  t = np.linspace(0, mag, 2)
  theta = np.linspace(0, 2 * np.pi, 100)
  rsample = np.linspace(0, radius, 2)
  #use meshgrid to make 2d arrays
  t, theta2 = np.meshgrid(t, theta)
  rsample,theta = np.meshgrid(rsample, theta)
  # "Finite Cylinder" surface
  x_blank_cylinder, y_blank_cylinder, z_blank_cylinder = \
                                              [p0[i] + v[i] * t + radius * 
                                              np.sin(theta2) * n1[i]+radius* 
                                              np.cos(theta2)*
                                              n2[i] for i in [0, 1, 2]] 
  cylinder_surface = np.array(([x_blank_cylinder,
                                y_blank_cylinder,
                                z_blank_cylinder]))
  if plot:
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(cylinder_surface[0], cylinder_surface[1], cylinder_surface[2],
                rstride=1, cstride=1, linewidth=0,
                antialiased=antialiased, color=color, alpha=alpha)
    ax.set_aspect('equal')
    ax.view_init(azim=-0.0005)
    ax.set_xlim(-height,height)
    ax.set_ylim(-height,height)
    ax.set_zlim(-height,height)
    ax.set_xticks([-height,0,height])
    ax.set_yticks([-height,0,height])
    ax.set_zticks([-height,0,height])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Surface Plot: Finite Cylinder')
    plt.show()
  return cylinder_surface

# matrix exponential
def projection(A:np.ndarray)->np.ndarray:
  """
  compute matrix exponential
  """
  # A: N,m,m  
  # return exp(A): N,m,m 
  return expm(A)

def linear_transform_to_hyperbolic_space(d:int) -> np.ndarray:
  """
  # linear map from Euclidean space to hyperbolic space
  """
  m = d + 1
  M = np.zeros((d, m, m))
  for i in range(d):
    M[i, m-1, i] = 1
    M[i, i, m-1] = 1
  return M
   
def compute_hyperbolic_pathdev(X:np.ndarray) -> np.ndarray: 
  N,T,d = X.shape
  M = linear_transform_to_hyperbolic_space(d) # d,m,m
  dX = X[:,1:,:] - X[:,:-1,:] # N,T-1,d
  identity_matrix = np.eye(d+1,dtype=float)
  I = np.repeat(identity_matrix[np.newaxis, :, :], N, axis=0)
  hyper_path = [I]
  for i in range(T-1):
    M_dX = np.matmul(np.transpose(M,(1,-1,0)), dX[:,i].T) # m,m,d matmul d,N -> m,m,N -> N,m,m
    M_dX = np.transpose(M_dX,(-1,0,1)) # m,m,N -> N,m,m
    exp_M_dX = projection(M_dX)  # N,m,m
    if i == 0:
      cumulative_product = exp_M_dX
    else:
      cumulative_product = np.matmul(cumulative_product, exp_M_dX) 
    hyper_path.append(cumulative_product)
  return np.stack(hyper_path, axis=1) # N,T,m,m

def test_H2_pathdev(N:int=1, T:float=0.5, n_steps:int=100, rng=np.random.default_rng(1635134)):
  dt = T / n_steps
  increments = rng.normal(size=(N, n_steps, 2), scale=np.sqrt(dt))  # N, n_steps, 2
  trajectories_2d = np.cumsum(increments, axis=1)  # N, n_steps, 2
  initial_position = np.zeros((N, 1, 2))
  trajectories_2d = np.concatenate((initial_position, trajectories_2d), axis=1)  # N, n_steps+1, 2

  # fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,6))
  fig = plt.figure(figsize=(10,8))
  ax1 = fig.add_subplot(121)
  for i in range(N):
    ax1.plot(trajectories_2d[i, :, 0], trajectories_2d[i, :, 1])
  ax1.set_xlabel('X')
  ax1.set_ylabel('Y')
  ax1.set_title('2D Brownian Motion Paths')
  
  hyperbolic_pathdev = compute_hyperbolic_pathdev(trajectories_2d) # N,n_steps+1,m,m where m=d+1
  start = np.array([0,0,1]).reshape(3,-1) # m,1
  hyperbolic_pathdev = np.matmul(hyperbolic_pathdev,start)  # N,n_steps+1,m,1

  def hyperbolic_plane(x1, x2):
    return np.sqrt((x1 ** 2 + x2 ** 2) + 1)

  x1 = np.linspace(-1, 1, 100)
  x2 = np.linspace(-1, 1, 100)
  X1, X2 = np.meshgrid(x1, x2)
  X3 = hyperbolic_plane(X1, X2)

  ax2 = fig.add_subplot(122, projection='3d',proj_type='persp')
  ax2.plot_surface(X1, X2, X3, color='lightblue',alpha=0.5)

  for path in hyperbolic_pathdev:
    x_coords = path[:,0]
    y_coords = path[:,1]
    z_coords = path[:,2]
    ax2.plot(x_coords,y_coords,z_coords,color='r',linewidth=1)
    
  ax2.view_init(elev=40)
  ax2.set_xlabel('x1')
  ax2.set_ylabel('x2')
  ax2.set_zlabel('x3')
  plt.tight_layout()
  plt.show()