import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import math

# Physical constants (keeping metric for calculations)
g = 9.81  # gravity (m/s²)
rho = 1.225  # air density (kg/m³) at sea level
C_d = 0.47  # drag coefficient for sphere (tennis ball)
radius = 0.0335  # tennis ball radius (m)
A = np.pi * radius**2  # cross-sectional area
mass = 0.057  # tennis ball mass (kg)

# Court dimensions (metric for calculations)
court_length = 23.77  # meters
net_height = 0.914  # meters (at center)

# Conversion factors
M_TO_FT = 3.28084
MS_TO_MPH = 2.23694

# Drag coefficient
k = 0.5 * rho * C_d * A / mass

def projectile_motion_with_drag(state, t):
    """
    Differential equation for projectile motion with quadratic drag
    state = [x, y, vx, vy]
    """
    x, y, vx, vy = state
    
    # Speed for drag calculation
    speed = np.sqrt(vx**2 + vy**2)
    
    # Accelerations due to drag (opposite to velocity direction)
    ax = -k * speed * vx
    ay = -g - k * speed * vy
    
    return [vx, vy, ax, ay]

def find_required_speed_with_drag(height, max_speed=200):
    """
    Find the minimum speed needed to hit the ball out (with drag)
    """
    speeds = np.linspace(1, max_speed, 1000)
    
    for speed in speeds:
        # Initial conditions
        initial_state = [0, height, speed, 0]  # x, y, vx, vy
        
        # Time array (generous upper bound)
        t = np.linspace(0, 5, 1000)
        
        # Solve differential equation
        solution = odeint(projectile_motion_with_drag, initial_state, t)
        
        x_vals = solution[:, 0]
        y_vals = solution[:, 1]
        
        # Find when ball hits ground (y = 0)
        ground_indices = np.where(y_vals <= 0)[0]
        
        if len(ground_indices) > 0:
            # Interpolate to find exact landing position
            idx = ground_indices[0]
            if idx > 0:
                # Linear interpolation
                t1, t2 = t[idx-1], t[idx]
                y1, y2 = y_vals[idx-1], y_vals[idx]
                x1, x2 = x_vals[idx-1], x_vals[idx]
                
                # Time when y = 0
                t_ground = t1 - y1 * (t2 - t1) / (y2 - y1)
                x_ground = x1 + (x2 - x1) * (t_ground - t1) / (t2 - t1)
                
                if x_ground >= court_length:
                    return speed
    
    return None  # No solution found

# Calculate for different heights (convert feet to meters for calculations)
heights_ft = np.linspace(2, 7, 20)  # Heights in feet
heights_m = heights_ft / M_TO_FT  # Convert to meters for calculations

required_speeds_drag = []

print("Calculating required speeds...")
for h_m in heights_m:
    # With drag only
    speed_drag = find_required_speed_with_drag(h_m)
    required_speeds_drag.append(speed_drag)

# Convert to mph and filter out None values
valid_indices = [i for i, speed in enumerate(required_speeds_drag) if speed is not None]
heights_valid_ft = [heights_ft[i] for i in valid_indices]
speeds_drag_mph = [required_speeds_drag[i] * MS_TO_MPH for i in valid_indices]

# # Plotting
# # First Figure - Speed Requirements
# plt.figure(figsize=(10, 5))
# plt.plot(heights_valid_ft, speeds_drag_mph, 'r-', linewidth=2)
# plt.xlabel('Height from which Ball is Struck (feet)')
# plt.ylabel('Required Speed to Travel Long (mph)')
# plt.title('Speed Required to Hit a Tennis Ball Out')
# plt.grid(False)
# plt.show()

# Second Figure - Trajectory
plt.figure(figsize=(10, 5))
height_demo_ft = 3.28084  # This is where to change ball struck from height for second plot
height_demo_m = height_demo_ft / M_TO_FT  # Convert to meters
speed_demo = find_required_speed_with_drag(height_demo_m)

if speed_demo:
    # With drag
    initial_state = [0, height_demo_m, speed_demo, 0]
    t = np.linspace(0, 2, 500)
    solution = odeint(projectile_motion_with_drag, initial_state, t)
    
    # Find trajectory until ground
    ground_idx = np.where(solution[:, 1] <= 0)[0]
    if len(ground_idx) > 0:
        end_idx = ground_idx[0]
        # Convert trajectory to feet
        x_traj_ft = solution[:end_idx, 0] * M_TO_FT
        y_traj_ft = solution[:end_idx, 1] * M_TO_FT
        speed_demo_mph = speed_demo * MS_TO_MPH
        
        plt.plot(x_traj_ft, y_traj_ft, 'r-', linewidth=2, 
            label=f'Ball trajectory (Initial Speed={speed_demo_mph:.1f} mph)')

print(speed_demo_mph)
# Court baseline in feet
court_length_ft = court_length * M_TO_FT
net_height_ft = net_height * M_TO_FT
plt.axvline(x=court_length_ft, color='k', linestyle=':', label='Baseline')
plt.vlines(x=court_length_ft/2, ymin=0, ymax=3, colors='blue', 
           linestyles='--', alpha=0.7, label='Net Position')
plt.axhline(y=net_height_ft, color='y', linestyle=':', label="Net Height")
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('Distance (feet)')
plt.ylabel('Height (feet)')
plt.title(f'Ball Trajectory (Initial Height: {height_demo_m} meters)')
plt.legend()
#plt.grid(True, alpha=0.3)
plt.savefig('TrajectoryPlot.png')
plt.show()
plt.close()