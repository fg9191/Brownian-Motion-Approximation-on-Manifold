import torch
# import signatory
import roughpy as rp
import math
import numpy as np
import seaborn as sns
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *

class manifold(object):
  def __init__(self,
              manifold='sphere',
              radius_sphere:float=1.0,
              radius_cylinder:float=1,
              height_cylinder:float=10,
              plt_interactive=True):
    
    self.manifold = manifold
    self.radius_sphere = radius_sphere
    self.radius_cylinder = radius_cylinder
    self.height_cylinder = height_cylinder
    self.sphere_surface = get_sphere_surface(self.radius_sphere)
    self.cylinder_surface = get_cylinder_surface(self.radius_cylinder,self.height_cylinder)
    self.half_injective_sphere = self.radius_sphere*np.pi/2

    # if plt_interactive == True:
    #     plt.ion()
    # else:
    #     plt.ioff()
    
    if self.manifold not in ('sphere', 'cylinder'):
      raise NameError('{0} is not a recognized manifold!'.format(self.manifold))


  def __repr__(self):
    """An internal representation"""

    return "{0}(manifold='{1}', radius_sphere = {2},\n\
    radius_cylinder = {3}, height_cylinder = {4})".format(
          self.__class__.__name__,
          self.manifold,
          self.radius_sphere,
          self.radius_cylinder,
          self.height_cylinder)


  def __str__(self):
    return "The manifold is a {0}!".format(self.manifold)


  def is_on_sphere(self, position:np.array):
    return position[-1] == self.radius_sphere

  def Riemannian_distance_on_2sphere(self, position1:np.array, position2:np.array):
    """
    calculate the Riemannian distance between two points x and v on a 2-sphere
    x, v: 3D sphere coordinates of two points on the 2-sphere with positive value of radius
    """

    theta_1, phi_1 = position1[0], position1[1]
    theta_2, phi_2 = position2[0], position2[1]
    # Riemannian distance is the angle between two points
    return self.radius_sphere*np.arccos(np.cos(phi_1)*np.cos(phi_2)+np.sin(phi_1)*np.sin(phi_2)*np.cos(theta_1-theta_2))

  def compute_last_step_distances(self, cart_trajs_all:np.ndarray, target_position)->list:
    """
    cart_trajs_all:N,n_steps+1,d
    """
    if not self.is_on_sphere(target_position):
      raise ValueError('Given target position({0}) is not on the sphere(r={1}).'.format(target_position,self.radius_sphere))
    distances = []
    for i in range(cart_trajs_all.shape[0]):
      last_step = cart_trajs_all[i,-1]
      distance = self.Riemannian_distance_on_2sphere(cartesian_to_spherical(*last_step), target_position)
      distances.append(distance)
    return distances

  @staticmethod
  def spherical_to_cartesian(theta, phi, radius) -> np.array:
    """
    convert (theta, phi, radius) to Cartesian coordinates on the sphere
    """
    x = radius*np.sin(phi)*np.cos(theta)
    y = radius*np.sin(phi)*np.sin(theta)
    z = radius*np.cos(phi)
    return np.array([x, y, z])

  @staticmethod
  def cartesian_to_spherical(x, y, z) -> np.array:
    """
    convert Cartesian coordinates to (theta, phi, radius)
    """
    radius = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arccos(z/radius)
    theta = np.arctan2(y, x)
    return np.array([theta, phi, radius])


  @staticmethod
  def rodrigues_rotation(vector, axis, angle) -> np.array:
    """
    Rodrigues rotation formula
    """
    ## equivalent method:
    # axis = axis / np.linalg.norm(axis)
    # cos_theta = np.cos(angle)
    # sin_theta = np.sin(angle)
    # rotated_vector1 = (vector +
    #                   sin_theta * np.cross(axis, vector) +
    #                   (1 - cos_theta) * np.cross(axis, np.cross(axis, vector)))

    # axis should be normalised
    axis = axis / np.linalg.norm(axis)

    cp_matrix = np.array([[0, -axis[2], axis[1]],\
                          [axis[2], 0, -axis[0]],\
                          [-axis[1], axis[0], 0]])
    I= np.eye(3)
    R = I + np.sin(angle)*cp_matrix + (1 - np.cos(angle))*(cp_matrix @ cp_matrix)
    rotated_vector = R @ vector
    return rotated_vector 

  @staticmethod
  def compute_increments(cartesian_path:np.array)->np.array:
    """
    cartesian_path: n_steps+1,d
    return increments:n_steps,d
    """
    increments = cartesian_path[1:,]-cartesian_path[:-1,]
    return increments

  @staticmethod
  def compute_approx(depth:int, mean_tensor_norm:float)->float:
    """
    compute the length conjecture
    """
    return np.power(math.factorial(depth)*(mean_tensor_norm), 1/depth)


  def generate_random_pairs_within_half_injectivity(self, n_samples:int=5, rng=np.random.default_rng(1635134))-> tuple:
    start_positions = []
    target_positions = []
    for _ in range(n_samples):
      theta1 = rng.uniform(0, 2 * np.pi) 
      phi1 = rng.uniform(0, np.pi)
      point1 = np.array([theta1, phi1, self.radius_sphere])
      start_positions.append(point1)

      theta2 = rng.uniform(0, 2 * np.pi)
      phi2 = rng.uniform(0, np.pi)
      point2 = np.array([theta2, phi2, self.radius_sphere])

      while Riemannian_distance_on_2sphere(point1, point2) > self.half_injective_sphere:
        theta2 = rng.uniform(0, 2 * np.pi)
        phi2 = rng.uniform(0, np.pi)
        point2 = np.array([theta2, phi2, self.radius_sphere])
      target_positions.append(point2)

    return start_positions, target_positions


  @staticmethod
  def plot_BB_last_step_hist(distances:list, T:float):
    plt.figure(figsize=(6,4))
    sns.histplot(distances, kde=True, bins=30)
    plt.xlabel('last step discrepancies')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of last step discrepancies for T={T}')
    plt.show()


  def initial_orthonormal_basis(self,theta, phi)->tuple:
    """
    generate the initial orthornormal basis H_1, H_2 at the starting point (theta, phi, radius)
    which are orthonormal vectors long the longitude (theta) and polar (phi) directions
    set it as [1,0,0] and [0,1,0] if starting point is the pole.
    """
    if phi==0:
      H1 = np.array([1,0,0])
      H2 = np.array([0,1,0])
    else:
      # along the longitude direction
      H1 = self.radius_sphere*np.array([-np.sin(theta)*np.sin(phi),
                                        np.cos(theta)*np.sin(phi),
                                        0])
      H1 = H1/np.linalg.norm(H1)

      # along the latitude direction
      H2 = self.radius_sphere*np.array([np.cos(theta)*np.cos(phi),
                                        np.sin(theta)*np.cos(phi),
                                        -np.sin(phi)])
      H2 = H2/np.linalg.norm(H2)
    return H1, H2


  # Brownian motion simulation on a 2-sphere using "rolling without slipping"
  def generate_BM_on_2sphere(self, T:float=1.0, n_steps:int=100, 
                             start_position=np.array([0, 0, 1.0]),
                             rng=np.random.default_rng(1635134))->tuple:

    """
    # start_position: spherical coordinates (theta,phi,self.radius_sphere)
    where theta is azimuthal angle (longitude) and phi is polar angle (latitude)
    """
    if not self.is_on_sphere(start_position):
      raise ValueError('Given start position ({0}) is not on this sphere(r={1})'.format(start_position,self.radius_sphere))

    dt = T / n_steps

    spherical_path = [start_position]
    cartesian_path = [spherical_to_cartesian(*start_position)]

    # initial orthonormal basis in the tangent plane attached to the starting point
    H1, H2 = self.initial_orthonormal_basis(*start_position[:-1])

    dW = rng.normal(scale=np.sqrt(dt), size=(n_steps, 2))

    # generate Brownian motion path
    for k in range(n_steps):
      current_spherical = spherical_path[-1]
      current_cartesian = cartesian_path[-1]

      # current tangent vector
      tangent_vector = dW[k,0] * H1 + dW[k,1] * H2

      # axis of rotation (orthogonal to current position and tangent vector)
      axis = np.cross(current_cartesian, tangent_vector)
      # rotation angle (step size normalized by the radius)
      angle = np.linalg.norm(tangent_vector) / self.radius_sphere
      # Rodrigues' rotation formula to get the new position on the sphere
      new_cartesian = rodrigues_rotation(current_cartesian, axis, angle)

      # update the current position and store it in the path
      cartesian_path.append(new_cartesian)
      spherical_path.append(cartesian_to_spherical(*new_cartesian))

      # current_cartesian = new_cartesian

      H1 = rodrigues_rotation(H1, axis, angle)
      H2 = rodrigues_rotation(H2, axis, angle)

    return np.array(cartesian_path), np.array(spherical_path)


  def generate_multiple_BM_on_2sphere(self, T:float=1.0, n_steps:int=100, 
                                      start_position=np.array([0, 0, 1.0]), n_samples:int=5,
                                      rng=np.random.default_rng(1635134))->np.ndarray:
    cart_trajs_all = []
    for _ in range(n_samples):
        cart_traj, _ = self.generate_BM_on_2sphere(T=T, n_steps=n_steps, start_position=start_position, rng=rng)
        cart_trajs_all.append(cart_traj)
    return np.array(cart_trajs_all)


  def plot_BM_path(self, trajectories:np.array, T:float=1.0, label=False):
    start_position = np.array(spherical_to_cartesian(*trajectories[0,0]))
    start_position_str = np.array2string(start_position, precision=2, separator=',', suppress_small=True)[1:-1]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

    # plot the sphere
    ax.plot_surface(self.sphere_surface[0], self.sphere_surface[1], self.sphere_surface[2], 
                    rstride=1, cstride=1, linewidth=0, color='b', alpha=0.05, antialiased=True)

    # plot each trajectory
    for i in range(trajectories.shape[0]):
        trajectory = trajectories[i]
        x_coords = trajectory[:, 0]
        y_coords = trajectory[:, 1]
        z_coords = trajectory[:, 2]
        if label is True:
          ax.plot(x_coords, y_coords, z_coords, label=f'T={T:.3f}, Path {i+1}', linewidth=0.7)
        else:
          ax.plot(x_coords, y_coords, z_coords, linewidth=0.7)

    # highlight start position
    ax.scatter(start_position[0], start_position[1], start_position[2], color='r', s=100, label='Start Position')
    ax.set_aspect('auto')
    ax.view_init(elev=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Brownian Motion Paths on 2-Sphere(r={self.radius_sphere}) Starting at: ({start_position_str})')
    ax.legend(loc='best')
    plt.show()


  def generate_BB_on_2sphere(self, T:float=0.05, n_steps=10,
                             start_position=np.array([0, 0, 1.0]), target_position=np.array([0, np.pi/2, 1.0]),
                             rng=np.random.default_rng(1635134))->tuple:
    if not self.is_on_sphere(start_position):
      raise ValueError('Given start position ({0}) is not on this sphere(r={1})'.format(start_position,self.radius_sphere))

    if not self.is_on_sphere(target_position):
      raise ValueError('Given target position ({0}) is not on this sphere(r={1})'.format(target_position,self.radius_sphere))

    start_targ_distance = Riemannian_distance_on_2sphere(start_position, target_position)
    if start_targ_distance > self.half_injective_sphere :
      raise ValueError(f"The Riemannian distance between the start and target positions\n\
                        must be less than or equal to {self.half_injective_sphere:.3f}.\n\
                        Given distance: {start_targ_distance:.3f}")
    # step_size
    dt = T / n_steps

    spherical_path = [start_position]
    cartesian_path = [spherical_to_cartesian(*start_position)]

    target_spherical = target_position
    target_cartesian = spherical_to_cartesian(*target_position)

    # generate R^2-Brownian motion increments
    dW = rng.normal(scale=np.sqrt(dt), size=(n_steps, 2))

    # calculate the initial orthornormal basis
    H1, H2 = self.initial_orthonormal_basis(*start_position[:-1])

    for k in range(n_steps):
      # current position
      current_spherical = spherical_path[-1]
      theta_k, phi_k, _ = current_spherical
      current_cartesian = cartesian_path[-1]

      # initial orthonormal basis in the tangent plane attached to the starting point
      H_k = dW[k,0]*H1 + dW[k,1]*H2

      # calculate Riemannian distance between current position to the target position
      r_v = Riemannian_distance_on_2sphere(current_spherical, target_spherical)

      # compute gradient of r_v at current point which should be a vector in the tangent space
      r_v_gradient = - (target_cartesian - current_cartesian * np.dot(current_cartesian, target_cartesian) / radius**2) \
                        / np.linalg.norm(target_cartesian - current_cartesian * np.dot(current_cartesian, target_cartesian) / radius**2)

      # calculate the guiding drift term
      guiding_drift_k = - (2*r_v*r_v_gradient*dt)/(2*(T-k*dt))

      # current tangent vector
      tangent_vector = H_k + guiding_drift_k

      # if np.isclose(np.dot(tangent_vector,current_cartesian), 0, atol=1e-5):
      #   raise ValueError(f" Tangent vector not in the tangential plane, np.dot(tangent_vector,current_cartesian)={np.dot(tangent_vector,current_cartesian)}")
      # else:
      # axis of rotation (orthogonal to current position and tangent vector)
      axis = np.cross(current_cartesian, tangent_vector)
      angle = np.linalg.norm(tangent_vector) / radius
      # Rodrigues' rotation formula to get the new position on the sphere
      new_cartesian = rodrigues_rotation(current_cartesian, axis, angle)

      # update the current position and store it in the path
      cartesian_path.append(new_cartesian)
      spherical_path.append(cartesian_to_spherical(*new_cartesian))

      # update the tangent vectors H1 and H2 to align with the new tangent plane
      H1 = rodrigues_rotation(H1, axis, angle)
      H2 = rodrigues_rotation(H2, axis, angle)
    return np.array(cartesian_path), np.array(spherical_path)


  def generate_multiple_BB_on_2sphere(self, T:float=0.05, n_steps=10,
                                      start_position=np.array([0, 0, 1.0]), target_position=np.array([0, np.pi/2, 1.0]),
                                      n_samples:int=5, rng=np.random.default_rng(1635134))->np.ndarray:
    cart_trajs_all = []
    for _ in range(n_samples):
        cart_traj, _ = self.generate_BB_on_2sphere(T=T, n_steps=n_steps, start_position=start_position, target_position=target_position, rng=rng)
        cart_trajs_all.append(cart_traj)
    return np.array(cart_trajs_all)


  def plot_BB_path(self, trajectories:np.array, T:float=0.05, 
                   target_position=np.array([0, np.pi/2, 1.0]), label=False):
    start_position = np.array(spherical_to_cartesian(*trajectories[0,0]))
    target_position = np.array(spherical_to_cartesian(*target_position))

    start_position_str = np.array2string(start_position, precision=2, separator=',', suppress_small=True)[1:-1]
    target_position_str = np.array2string(target_position, precision=2, separator=',', suppress_small=True)[1:-1]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

    # Plot the sphere
    ax.plot_surface(self.sphere_surface[0], self.sphere_surface[1], self.sphere_surface[2], 
                  rstride=1, cstride=1, linewidth=0, color='b', alpha=0.05, antialiased=True)

    # Plot each trajectory
    for i in range(trajectories.shape[0]):
        trajectory = trajectories[i]
        x_coords = trajectory[:,0]
        y_coords = trajectory[:,1]
        z_coords = trajectory[:,2]
        if label is True:
          ax.plot(x_coords, y_coords, z_coords, label=f'T={T:.3f}, Path {i+1}', linewidth=0.7)
        else:
          ax.plot(x_coords, y_coords, z_coords, linewidth=0.7)
  
    # highlight start and target positions
    ax.scatter(start_position[0], start_position[1], start_position[2], color='r', s=50, label='Start Position')
    ax.scatter(target_position[0], target_position[1], target_position[2], color='black', s=50, label='Target Position')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Brownian Bridge Paths on 2-Sphere(r={self.radius_sphere}) with Start: ({start_position_str}) and Target: ({target_position_str})')
    ax.legend(loc='best')
    plt.show()


  @staticmethod
  def get_hyperbolic_pathdev(trajectories:np.ndarray)-> np.ndarray:
    hyperbolic_pathdev = compute_hyperbolic_pathdev(cart_trajs_all) # N,n_steps,m,m where m=d+1
    # print(hyperbolic_pathdev.shape)


  def explore_sensitivity(self, depth_range:list, start_positions:list, target_positions:list,
                          n_steps=10, n_samples_per_depth=10, n_samples_per_depth_stored=1,
                          rng=np.random.default_rng(1635134), package='signatory')->dict:
    """
    explore sensitivity to parameter variations: start-target positions pairs and depths (T)
    """
    results = {}

    for start_position, target_position in zip(start_positions, target_positions):
      if not self.is_on_sphere(start_position):
        raise ValueError('Given start position ({0}) is not on this sphere(r={1})'.format(start_position,self.radius_sphere))

      if not self.is_on_sphere(target_position):
        raise ValueError('Given target position ({0}) is not on this sphere(r={1})'.format(target_position,self.radius_sphere))

      start_targ_distance = Riemannian_distance_on_2sphere(start_position, target_position)
      if start_targ_distance  > self.half_injective_sphere:
        raise ValueError(f"The Riemannian distance between the start and target positions\n\
                          must be less than or equal to {self.half_injective_sphere:.3f}.\n\
                          Given distance: {start_targ_distance :.2f} bewteen Start ({start_position}) and Target ({target_position})")

      position_key = (tuple(start_position), tuple(target_position))
      results[position_key] = {}

      for depth in depth_range:
        T = np.power(float(depth),-7)
        times = np.linspace(0, T, n_steps+1)[1:]
        interval = rp.RealInterval(0, T+1)
        context = rp.get_context(width=3, depth=depth, coeffs=rp.DPReal)

        # run multiple simulations for each parameter combination
        sample_trajectories = []
        all_trajectories = []
        all_trunctensor = []
        slice_len = 3**depth

        for sample in range(n_samples_per_depth):
          cartesian_path, _ = generate_BB_on_2sphere(radius, T=T, n_steps=n_steps,
                                                                  start_position=start_position, target_position=target_position)
          all_trajectories.append(cartesian_path)
          if sample % int(np.ceil(n_samples_per_depth/n_samples_per_depth_stored)) == 0:
              sample_trajectories.append(cartesian_path)

          if package == 'signatory':
            cartesian_path_tensor = torch.tensor(np.array(cartesian_path))
            sig = signatory.signature(cartesian_path_tensor.unsqueeze(0), depth).squeeze(0)
          else:
            cartesian_path_incre = compute_increments(cartesian_path)
            stream = rp.LieIncrementStream.from_increments(cartesian_path_incre, indices=times, ctx=context)
            sig = stream.signature(interval)
          
          sig = np.array(sig)
          all_trunctensor.append(sig[-slice_len:])

        all_trunctensor_array = np.array(all_trunctensor)
        mean_trunctensor = np.mean(all_trunctensor_array, axis=0)
        mean_trunctensor_norm = np.linalg.norm(mean_trunctensor)
        approx_dist = compute_approx(depth, mean_trunctensor_norm)

        # Store the results for analysis
        results[position_key][(depth, T)] = {
          "all_trajectories": np.array(all_trajectories),
          "sample_trajectories": np.array(sample_trajectories),
          "approx_distance": approx_dist,
          "true_distance": start_targ_distance
        }
    return results


  def plot_sensitivity_results(self, results:dict, plot_sample_paths=True, plot_last_step_hist=True):
    # iterate over the start-target position combinations in the results dictionary
    for (start_position, target_position), depth_T_dict in results.items():

      if plot_sample_paths:
        fig1 = plt.figure(figsize=(12, 6))
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.plot_surface(self.sphere_surface[0], self.sphere_surface[1], self.sphere_surface[2], 
                         rstride=1, cstride=1, linewidth=0, color='b', alpha=0.05, antialiased=True)
        all_lines = []
        all_labels = []

      approx_values = []

      start_position_cart = np.array(spherical_to_cartesian(*start_position))
      target_position_cart = np.array(spherical_to_cartesian(*target_position))

      start_position_str = np.array2string(start_position_cart, precision=2, separator=',', suppress_small=True)[1:-1]
      target_position_str = np.array2string(target_position_cart, precision=2, separator=',', suppress_small=True)[1:-1]
      
      if plot_last_step_hist:
        length = len(depth_T_dict)
        nrows = int(np.ceil(length/3))
        fig2, ax2 = plt.subplots(nrows,3,figsize=(15, 5*nrows))
        ax2 = ax2.flatten()

      # iterate over depths and Ts in the results for the current start-target pair
      for idx, ((depth, T), data) in enumerate(depth_T_dict.items()):
        sample_trajectories = data["sample_trajectories"]
        approx_dist = data["approx_distance"]
        true_dist = data["true_distance"]

        # store the approx values for this depth and T for later plotting
        approx_values.append((depth, T, approx_dist))

        if plot_sample_paths:
          for i in range(sample_trajectories.shape[0]):
            trajectory = sample_trajectories[i]
            x_coords = trajectory[:,0]
            y_coords = trajectory[:,1]
            z_coords = trajectory[:,2]
            line1 = ax1.plot(x_coords, y_coords, z_coords, label=fr'depth={depth}, T=$depth^{{-7}}$, Sample {i+1}', linewidth=0.7)
            all_lines.append(line1[0])
            all_labels.append(f'depth={depth}, T={T:.2e}, Sample {i+1}')

        if plot_last_step_hist:
          all_trajectories = data["all_trajectories"]
          distances = self.compute_last_step_distances(all_trajectories,target_position)
          sns.histplot(distances, kde=True, bins=30,ax=ax2[idx])
          ax2[idx].set_xlabel(f'depth={depth}, T={T:.2e}')
          ax2[idx].set_ylabel('Frequency')

      if plot_sample_paths:
        start_marker = ax1.scatter(start_position_cart[0], start_position_cart[1], start_position_cart[2], color='r', s=50, label='Start Position')
        target_marker = ax1.scatter(target_position_cart[0], target_position_cart[1], target_position_cart[2], color='black', s=50, label='Target Position')

        all_lines.extend([start_marker, target_marker])
        all_labels.extend(['Start Position', 'Target Position'])

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        fig1.suptitle(f'Start: ({start_position_str}), Target: ({target_position_str})')
        fig1.legend(all_lines, all_labels, loc='upper right', fontsize='small')

      if plot_last_step_hist:
        fig2.suptitle(f'Start: ({start_position_str}), Target: ({target_position_str})')
        plt.tight_layout()

      # plot the approx_values vs. depth or T
      fig3, ax3 = plt.subplots(figsize=(6, 4))

      # sort approx_values by depth or T for smooth plotting
      approx_values_sorted = sorted(approx_values, key=lambda x: x[0])  # Sort by depth or T
      depths = [val[0] for val in approx_values_sorted]
      approx_vals = [val[2] for val in approx_values_sorted]
      T_s = [val[1] for val in approx_values_sorted]

      # plot approximate distances against depth
      ax3.plot(depths, approx_vals, label='Approx Value vs Depth', marker='o', linestyle='-')
      ax3.axhline(y=true_dist, color='r', linestyle='--', label=f'True Distance = {true_dist:.5f}')

      # annotation
      for i, (depth, approx_val) in enumerate(zip(depths, approx_vals)):
        ax3.annotate(f'{approx_val:.4f}',
                      (depths[i], approx_vals[i]),
                      textcoords="offset points",
                      xytext=(0, 5),
                      ha='center',
                      fontsize=8)

      ax3.set_xlabel('Depth')
      ax3.set_ylabel('Approx Distance')
      ax3.set_title(f'Approx Values vs Depth for Start: ({start_position_str}), Target: ({target_position_str})')
      ax3.legend()

      plt.tight_layout()
      plt.show()
    
