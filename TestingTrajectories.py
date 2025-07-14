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
net_position = court_length / 2  # meters from baseline

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

def check_net_clearance(solution):
    """
    More robust net clearance check with better interpolation
    """
    x_vals = solution[:, 0]
    y_vals = solution[:, 1]
    
    # Find the indices that bracket the net position
    before_net_idx = None
    after_net_idx = None
    
    for i in range(len(x_vals)):
        if x_vals[i] < net_position:
            before_net_idx = i
        elif x_vals[i] >= net_position and after_net_idx is None:
            after_net_idx = i
            break
    
    # Handle edge cases
    if before_net_idx is None:
        # Ball starts past the net
        return y_vals[0] > net_height
    
    if after_net_idx is None:
        # Ball doesn't reach the net
        return False
    
    # Interpolate height at exact net position
    x1, x2 = x_vals[before_net_idx], x_vals[after_net_idx]
    y1, y2 = y_vals[before_net_idx], y_vals[after_net_idx]
    
    # Linear interpolation
    if abs(x2 - x1) > 1e-10:  # Avoid division by zero
        t_interp = (net_position - x1) / (x2 - x1)
        height_at_net = y1 + t_interp * (y2 - y1)
    else:
        height_at_net = y1
    
    # Check if ball clears net - reduced safety margin
    clearance = height_at_net > net_height + 0.005  # 5mm safety margin
    
    return clearance

def find_maximum_speed_with_angle(height, angle_deg, max_speed=200):
    """
    Find the maximum speed that keeps the ball in bounds (binary search)
    """
    angle_rad = np.radians(angle_deg)
    
    # Find a reasonable starting low bound by testing a valid speed first
    low_speed = 10.0  # Start with a reasonable minimum
    high_speed = max_speed
    tolerance = 0.1
    
    best_speed = None
    debug_10_deg = (abs(angle_deg - 10.0) < 0.1)  # Debug flag for 10 degrees
    
    # First, find a valid lower bound by testing incrementally
    found_valid_low = False
    test_speed = 10.0
    while test_speed < max_speed and not found_valid_low:
        vx_initial = test_speed * np.cos(angle_rad)
        vy_initial = test_speed * np.sin(angle_rad)
        initial_state = [0, height, vx_initial, vy_initial]
        
        t = np.linspace(0, 10, 2000)
        
        try:
            solution = odeint(projectile_motion_with_drag, initial_state, t)
            
            if check_net_clearance(solution):
                x_vals = solution[:, 0]
                y_vals = solution[:, 1]
                ground_indices = np.where(y_vals <= 0)[0]
                
                if len(ground_indices) > 0:
                    idx = ground_indices[0]
                    if idx > 0:
                        t1, t2 = t[idx-1], t[idx]
                        y1, y2 = y_vals[idx-1], y_vals[idx]
                        x1, x2 = x_vals[idx-1], x_vals[idx]
                        
                        if abs(y2 - y1) > 1e-10:
                            t_ground = t1 - y1 * (t2 - t1) / (y2 - y1)
                            x_ground = x1 + (x2 - x1) * (t_ground - t1) / (t2 - t1)
                        else:
                            x_ground = x1
                        
                        if x_ground <= court_length:
                            low_speed = test_speed
                            found_valid_low = True
                            break
        except:
            pass
        
        test_speed += 5.0
    
    if not found_valid_low:
        return None
    
    while high_speed - low_speed > tolerance:
        speed = (low_speed + high_speed) / 2
        
        # Initial velocity components
        vx_initial = speed * np.cos(angle_rad)
        vy_initial = speed * np.sin(angle_rad)
        initial_state = [0, height, vx_initial, vy_initial]
        
        # Use consistent time array for all angles
        t = np.linspace(0, 10, 2000)  # Longer time, more points for accuracy
        
        # Solve differential equation
        try:
            solution = odeint(projectile_motion_with_drag, initial_state, t)
        except:
            high_speed = speed
            continue
        
        x_vals = solution[:, 0]
        y_vals = solution[:, 1]
        
        # First check if ball clears net
        net_cleared = check_net_clearance(solution)
        
        if debug_10_deg and speed > 40:  # Only debug higher speeds
            # Find height at net for debugging
            net_idx = np.argmin(np.abs(x_vals - net_position))
            height_at_net_approx = y_vals[net_idx]
            print(f"    Speed {speed:.1f}: Net height = {height_at_net_approx:.3f}m, Net cleared = {net_cleared}")
        
        if not net_cleared:
            high_speed = speed
            continue
        
        # Find when ball hits ground (y <= 0)
        ground_indices = np.where(y_vals <= 0)[0]
        
        if len(ground_indices) > 0:
            # Find exact landing position using interpolation
            idx = ground_indices[0]
            if idx > 0:
                # Linear interpolation to find exact landing spot
                t1, t2 = t[idx-1], t[idx]
                y1, y2 = y_vals[idx-1], y_vals[idx]
                x1, x2 = x_vals[idx-1], x_vals[idx]
                
                # Calculate time when y = 0
                if abs(y2 - y1) > 1e-10:  # Avoid division by zero
                    t_ground = t1 - y1 * (t2 - t1) / (y2 - y1)
                    x_ground = x1 + (x2 - x1) * (t_ground - t1) / (t2 - t1)
                else:
                    x_ground = x1
                
                if debug_10_deg and speed > 40:
                    print(f"    Landing at x = {x_ground:.2f}m (court length = {court_length:.2f}m)")
                
                if x_ground > court_length:
                    # Ball goes out of bounds, reduce speed
                    high_speed = speed
                else:
                    # Ball lands in bounds, try higher speed
                    low_speed = speed
                    best_speed = speed
            else:
                # Ball hits ground immediately
                high_speed = speed
        else:
            # Ball doesn't hit ground within time limit
            high_speed = speed
    
    # Return the best speed found (this was the key missing piece)
    return best_speed

# Calculate for different angles and heights
angles = np.linspace(5, 15, 11)  # 5 to 15 degrees
heights_ft = [3, 4, 5, 6]  # Different hitting heights in feet
heights_m = [h / M_TO_FT for h in heights_ft]  # Convert to meters

# Store results
results = {}
for height_ft, height_m in zip(heights_ft, heights_m):
    results[height_ft] = {'angles': [], 'max_speeds': []}
    
    print(f"Calculating for height: {height_ft} ft")
    for angle in angles:
        print(f"  Angle: {angle}°", end=" ")
        max_speed = find_maximum_speed_with_angle(height_m, angle)
        if max_speed and max_speed > 5:
            results[height_ft]['angles'].append(angle)
            results[height_ft]['max_speeds'].append(max_speed * MS_TO_MPH)
            print(f"→ {max_speed * MS_TO_MPH:.1f} mph")
        else:
            print("→ No solution")

# Plotting
plt.figure(figsize=(12, 8))

# Trajectory comparison at different angles (using 3 ft height)
demo_height_ft = 3.28084
demo_height_m = demo_height_ft / M_TO_FT
demo_angles = [5, 10, 15]
colors_demo = ['red', 'blue', 'green']

print(f"\nPlotting trajectories for {demo_height_m} m height:")
for angle, color in zip(demo_angles, colors_demo):
    max_speed = find_maximum_speed_with_angle(demo_height_m, angle)
    print(f"Angle {angle}°: Max speed = {max_speed}")
    
    if max_speed:
        # Calculate trajectory
        angle_rad = np.radians(angle)
        vx_initial = max_speed * np.cos(angle_rad)
        vy_initial = max_speed * np.sin(angle_rad)
        initial_state = [0, demo_height_m, vx_initial, vy_initial]
        
        t = np.linspace(0, 10, 2000)
        solution = odeint(projectile_motion_with_drag, initial_state, t)
        
        # Find trajectory until ground
        ground_idx = np.where(solution[:, 1] <= 0)[0]
        if len(ground_idx) > 0:
            end_idx = ground_idx[0]
            x_traj_ft = solution[:end_idx, 0] * M_TO_FT
            y_traj_ft = solution[:end_idx, 1] * M_TO_FT
            
            plt.plot(x_traj_ft, y_traj_ft, color=color, linewidth=2, 
                    label=f'{angle}° ({max_speed * MS_TO_MPH:.1f} mph)')

# Add court features
court_length_ft = court_length * M_TO_FT
net_height_ft = net_height * M_TO_FT
net_position_ft = net_position * M_TO_FT

plt.axvline(x=court_length_ft, color='black', linestyle='--', alpha=0.7, label='Baseline')
plt.axhline(y=net_height_ft, color='orange', linestyle='--', alpha=0.7, label='Net Height')

plt.vlines(x=net_position_ft, ymin=0, ymax=3, colors='blue', 
           linestyles='--', alpha=0.7, label='Net Position')

plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.xlabel('Distance (feet)')
plt.ylabel('Height (feet)')
plt.title(f'Tennis Ball Trajectory Comparison at {demo_height_m} m Contact Height')
plt.legend()
#plt.grid(True, alpha=0.3)
plt.xlim(0, 85)
plt.ylim(0, 9)
plt.savefig('TrajectoryMaxSpeeds.png')
plt.tight_layout()
plt.show()

# Print numerical results
print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)
for height_ft in heights_ft:
    if results[height_ft]['max_speeds']:
        print(f"\nHeight: {height_ft} feet")
        for angle, speed in zip(results[height_ft]['angles'], results[height_ft]['max_speeds']):
            print(f"  {angle:4.1f}° → {speed:5.1f} mph")
        
        if results[height_ft]['max_speeds']:
            max_speed_idx = np.argmax(results[height_ft]['max_speeds'])
            optimal_angle = results[height_ft]['angles'][max_speed_idx]
            optimal_speed = results[height_ft]['max_speeds'][max_speed_idx]
            print(f"  OPTIMAL: {optimal_angle:.1f}° = {optimal_speed:.1f} mph")
    else:
        print(f"\nHeight: {height_ft} feet - No valid solutions found")


#Currently working for calculating angles and trajectories for max speeds, does not account for spin