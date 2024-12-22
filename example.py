from utils import *
from manifold import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


test_H2_pathdev()

Sphere = manifold('sphere',radius_sphere=1.0)
# BM_trajectories = Sphere.generate_multiple_BM_on_2sphere()
# Sphere.plot_BM_path(BM_trajectories,label=True)

rng = np.random.default_rng(1635134)
radius = 1.0
n_steps = 2000
T_list = [1.0, 5.0]
start_position_list = [np.array([0,0,radius]), np.array([0,np.pi,radius])]
n_samples = 5
spheresurface = get_sphere_surface()
for T, start_position in zip(T_list,start_position_list):
  cart_trajs_all = Sphere.generate_multiple_BM_on_2sphere(T, n_steps, start_position, n_samples, rng=rng)
  Sphere.plot_BM_path(cart_trajs_all, T, label=True)


n_start_target_pairs = 5
start_positions, target_positions = Sphere.generate_random_pairs_within_half_injectivity(n_start_target_pairs)
depth_range = [5,6,7,8,9]
T_range = [np.power(float(depth), -7) for depth in depth_range]
n_steps = 20
n_samples_per_depth = 100
n_samples_per_depth_stored= 1
results = Sphere.explore_sensitivity(depth_range, start_positions, target_positions, n_steps, n_samples_per_depth, n_samples_per_depth_stored,rng=rng)
Sphere.plot_sensitivity_results(results)


n_steps = 100
n_steps_arr = np.linspace(0,n_steps, n_steps+1)
# T_list = [1.0, 0.005, 0.001]
T_list = T_range
start_position = np.array([0, 0, radius])
target_position = np.array([0, np.pi/2, radius])
n_samples = 1000
for T in T_list:
  cart_trajs_all = Sphere.generate_multiple_BB_on_2sphere(T, n_steps, start_position, target_position, n_samples, rng=rng)
  distances = Sphere.compute_last_step_distances(cart_trajs_all, target_position)  
  Sphere.plot_BB_path(cart_trajs_all, T=T, target_position=target_position, label=False) 
  Sphere.plot_BB_last_step_hist(distances,T)

