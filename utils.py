import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm 
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from IPython.display import display 

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

def compute_piecewise_normed_directions(cart_trajectories:np.ndarray)->np.ndarray:
  """
  cart_trajectories: N,nsteps+1,d
  return: N,nsteps,d
  """
  N, nsteps_plus1, d = cart_trajectories.shape
  piecewise_directions = cart_trajectories[:, 1:, :] - cart_trajectories[:, :-1, :]  # N, nsteps, d
  piecewise_norms = np.linalg.norm(piecewise_directions, axis=2, keepdims=True)  # N, nsteps, 1
  piecewise_normed_directions = piecewise_directions / piecewise_norms  # N, nsteps, d
  return piecewise_normed_directions


def compute_scaled_trajectories(cart_trajectories:np.ndarray, lam:float)->np.ndarray:
  """
  cart_trajectories: N,n_steps+1,d
  lam: float

  return: N,n_steps+1,d
  """
  piecewise_directions = cart_trajectories[:,1:,]-cart_trajectories[:,:-1,] # N,n_steps,d
  scaled_trajectories = [cart_trajectories[:,0,:]] # [N,d]
  for i in range(cart_trajectories.shape[1]-1):
    newpoint = scaled_trajectories[-1] + lam*piecewise_directions[:,i] # N,d
    scaled_trajectories.append(newpoint)
  return np.stack(scaled_trajectories, axis=1) # N,n_steps+1,d


def compute_scaled_hyperbolic_pathdev(cart_trajectories:np.ndarray, lam:float)->np.ndarray:
  """
  cart_trajectories: N,n_steps+1,d
  lam: float

  return: N,n_steps,d
  """
  scaled_trajectories = compute_scaled_trajectories(cart_trajectories, lam) # N,n_steps+1,d
  scaled_hyperbolic_pathdev = compute_hyperbolic_pathdev(scaled_trajectories) # N,n_steps+1,m,m
  return scaled_hyperbolic_pathdev # N,n_steps+1,m,m


def test_if_on_hyperboloid(pathdev_in_hyperboloid:np.ndarray):
  """
  pathdev_in_hyperboloid: N,nsteps+1,m,1
  """
  points = pathdev_in_hyperboloid[..., 0]  # N, nsteps+1, m
  hyperboloid_values = -points[..., -1]**2 + np.sum(points[..., :-1]**2, axis=-1)
  print('hyperboloid_values', hyperboloid_values)
  is_on_hyperboloid = np.isclose(hyperboloid_values, -1, atol=1e-1)
  if np.all(is_on_hyperboloid):
      print("All points lie on the H^3 hyperboloid.")
  else:
      print("Some points do not lie on the H^3 hyperboloid.")
      print("Points not on hyperboloid:", points[~is_on_hyperboloid])


def length_conj(cart_trajectories:np.ndarray, lam:float)->np.array:
  """
  applicable to any path in C^2.
  for experiment with linear piecewise embedding, the true length is regarded as sum of each linear piece
  ----------------------------------
  Parameters

  cart_trajectories: N,n_steps+1,d
  lam: float
  ----------------------------------
  Return

  length_of_trajectories: N,
  """
  N,nsteps_plus1,d = cart_trajectories.shape 
  scaled_hyperbolic_pathdev = compute_scaled_hyperbolic_pathdev(cart_trajectories, lam) # N,n_steps+1,m,m
  last_element = scaled_hyperbolic_pathdev[:,-1,-1,-1].reshape(-1,1) # N,1
  rho_final = np.arccosh(last_element) # N,1
  conjectured_lengths = rho_final/lam # N,1

  return conjectured_lengths # N,1


def test_length_conj(lam_list:list, n_paths:int=10, rng=np.random.default_rng(1635134),
                     display_df=True, plot_results=True):
  """
  Validate the length_conj function using randomly generated 3d piecewise linear paths.

  ----------------------------------------------------------------
  Parameter:

  n_paths: number of paths to be generated
  lam_list: a list of amplification factor values to be applied to each path

  ----------------------------------------------------------------
  Return:

  results_dict: dictionary
  results_df: pd.DataFrame
  """
  # randomly generate 3d piecewise linear paths
  T = 10
  random_uni_samples = rng.uniform(low=0., high=1., size=(n_paths, T, 3)) # [N, T, 3]
  paths = np.cumsum(random_uni_samples, axis=1) # [N, T, 3]
  initial_position = np.zeros((n_paths, 1, 3))
  paths = np.concatenate((initial_position, paths), axis=1) # [N, T+1, 3]
  
  results_dict = {}
  results_df = []

  for lam in lam_list:
    conjectured_lengths = length_conj(paths, lam) # N, 1
    true_piecewise_lengths = np.linalg.norm(paths[:, 1:, :] - paths[:, :-1, :], axis=2) # N, T
    true_lengths = np.sum(true_piecewise_lengths, axis=1).reshape(-1, 1) # N, 1
    relative_errors = np.abs(conjectured_lengths - true_lengths)*100/true_lengths # N, 1

    results_dict[lam] = {
        'Path Indices': np.arange(paths.shape[0]), 
        'Conjectured Lengths': conjectured_lengths, # N, 1
        'True Lengths': true_lengths, # N, 1
        'Relative Errors': relative_errors # N, 1
    }

    for i in range(paths.shape[0]):
      results_df.append({
          'lam': lam,
          'Path Index': i,
          'Conjectured Length': conjectured_lengths[i, 0], # scalar
          'True Length': true_lengths[i, 0], # scalar
          'Relative Error': relative_errors[i, 0], # scalar
      })

  results_df = pd.DataFrame(results_df)
  
  if display_df:
    display(display_df)
  
  if plot_results:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
    
    # Plot True Length vs Conjectured Length
    for lam in lam_list:
      ax1.scatter(results_dict[lam]['True Lengths'], results_dict[lam]['Conjectured Lengths'], label=f'λ={lam}',s=10)
    ax1.plot([min(results_df['True Length']), max(results_df['True Length'])], [min(results_df['True Length']), max(results_df['True Length'])], 'r--')
    ax1.set_xlabel('True Length')
    ax1.set_ylabel('Conjectured Length')
    ax1.set_title('True vs Conjectured Length')
    ax1.legend()
    
    # Plot Relative Error vs Lambda with color gradient for True Length
    all_lams = []
    all_relative_errors = []
    all_true_lengths = []
    for lam in lam_list:
      all_lams.extend([lam] * len(results_dict[lam]['Relative Errors']))
      all_relative_errors.extend(results_dict[lam]['Relative Errors'])
      all_true_lengths.extend(results_dict[lam]['True Lengths'])
    
    sc = ax2.scatter(all_lams, all_relative_errors, c=all_true_lengths, cmap='viridis', marker='o', s=10)
    ax2.set_xlabel('Lambda')
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title('Relative Error vs Lambda')
    plt.colorbar(sc, ax=ax2, label='True Length')
    
    plt.tight_layout()
    plt.show()
  
  return results_dict, results_df




