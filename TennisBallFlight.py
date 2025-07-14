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
RPM_TO_RAD_S = 2 * np.pi / 60  # Convert RPM to rad/s

# Drag coefficient
k = 0.5 * rho * C_d * A / mass

def calculate_magnus_coefficient(spin_rpm, velocity):
    """
    Calculate Magnus coefficient based on spin rate and velocity
    Uses empirical relationship for tennis balls
    """
    if spin_rpm == 0:
        return 0
    
    # Spin parameter (dimensionless)
    spin_rad_s = spin_rpm * RPM_TO_RAD_S
    spin_parameter = (spin_rad_s * radius) / velocity if velocity > 0 else 0
    
    # Empirical Magnus coefficient for tennis balls
    # Based on research showing C_M increases with spin parameter
    C_M = 0.1 + 0.4 * (1 - np.exp(-spin_parameter * 2))
    
    return min(C_M, 0.5)  # Cap at reasonable maximum

def projectile_motion_with_drag_and_magnus(state, t, spin_rpm):
    """
    Differential equation for projectile motion with quadratic drag and Magnus effect
    state = [x, y, vx, vy]
    spin_rpm = topspin rate in RPM (positive = topspin)
    """
    x, y, vx, vy = state
    
    # Speed for drag calculation
    speed = np.sqrt(vx**2 + vy**2)
    
    if speed < 0.1:  # Avoid division by zero
        return [vx, vy, 0, -g]
    
    # Drag force (opposite to velocity direction)
    drag_x = -k * speed * vx
    drag_y = -k * speed * vy
    
    # Magnus force calculation
    if spin_rpm != 0:
        C_M = calculate_magnus_coefficient(abs(spin_rpm), speed)
        magnus_force_magnitude = 0.5 * rho * speed**2 * A * C_M
        magnus_acceleration = magnus_force_magnitude / mass
        
        # For topspin (positive RPM), Magnus force is downward
        # Magnus force is perpendicular to velocity
        velocity_unit_x = vx / speed
        velocity_unit_y = vy / speed
        
        # Perpendicular vector (rotated 90° clockwise for topspin)
        if spin_rpm > 0:  # Topspin
            magnus_x = velocity_unit_y * magnus_acceleration
            magnus_y = -velocity_unit_x * magnus_acceleration
        else:  # Backspin
            magnus_x = -velocity_unit_y * magnus_acceleration
            magnus_y = velocity_unit_x * magnus_acceleration
    else:
        magnus_x = 0
        magnus_y = 0
    
    # Total accelerations
    ax = drag_x + magnus_x
    ay = -g + drag_y + magnus_y
    
    return [vx, vy, ax, ay]

def check_net_clearance(solution):
    """
    Check if ball clears the net
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
    
    if before_net_idx is None:
        return y_vals[0] > net_height
    
    if after_net_idx is None:
        return False
    
    # Interpolate height at exact net position
    x1, x2 = x_vals[before_net_idx], x_vals[after_net_idx]
    y1, y2 = y_vals[before_net_idx], y_vals[after_net_idx]
    
    if abs(x2 - x1) > 1e-10:
        t_interp = (net_position - x1) / (x2 - x1)
        height_at_net = y1 + t_interp * (y2 - y1)
    else:
        height_at_net = y1
    
    return height_at_net > net_height + 0.005  # 5mm safety margin

def find_maximum_speed_with_spin(height, angle_deg, spin_rpm, max_speed=200):
    """
    Find the maximum speed that keeps the ball in bounds with given spin
    """
    angle_rad = np.radians(angle_deg)
    
    low_speed = 10.0
    high_speed = max_speed
    tolerance = 0.1
    
    # Find valid lower bound
    found_valid_low = False
    test_speed = 10.0
    while test_speed < max_speed and not found_valid_low:
        vx_initial = test_speed * np.cos(angle_rad)
        vy_initial = test_speed * np.sin(angle_rad)
        initial_state = [0, height, vx_initial, vy_initial]
        
        t = np.linspace(0, 10, 2000)
        
        try:
            solution = odeint(projectile_motion_with_drag_and_magnus, initial_state, t, args=(spin_rpm,))
            
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
    
    # Binary search for maximum speed
    best_speed = None
    while high_speed - low_speed > tolerance:
        speed = (low_speed + high_speed) / 2
        
        vx_initial = speed * np.cos(angle_rad)
        vy_initial = speed * np.sin(angle_rad)
        initial_state = [0, height, vx_initial, vy_initial]
        
        t = np.linspace(0, 10, 2000)
        
        try:
            solution = odeint(projectile_motion_with_drag_and_magnus, initial_state, t, args=(spin_rpm,))
        except:
            high_speed = speed
            continue
        
        x_vals = solution[:, 0]
        y_vals = solution[:, 1]
        
        if not check_net_clearance(solution):
            high_speed = speed
            continue
        
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
                
                if x_ground > court_length:
                    high_speed = speed
                else:
                    low_speed = speed
                    best_speed = speed
            else:
                high_speed = speed
        else:
            high_speed = speed
    
    return best_speed

# Analysis setup - focus on 5-degree angle only
angle = 10.0  # degrees
height_ft = 3.28084  # feet
height_m = height_ft / M_TO_FT

# Test different spin rates
spin_rates = [0, 1000, 2000, 2750, 3000, 3500]  # RPM
colors = ['black', 'blue', 'green', 'orange', 'red', 'purple']

print(f"Analyzing 5-degree shots at {height_ft} ft height with varying topspin...")
print("="*60)

results = {}
for spin_rpm in spin_rates:
    print(f"Testing {spin_rpm} RPM spin...", end=" ")
    max_speed = find_maximum_speed_with_spin(height_m, angle, spin_rpm)
    
    if max_speed:
        results[spin_rpm] = max_speed
        print(f"Max speed: {max_speed * MS_TO_MPH:.1f} mph")
    else:
        print("No valid solution")

# Create single plot
plt.figure(figsize=(12, 8))

# Plot trajectories for different spin rates
print("\nPlotting trajectories...")
for spin_rpm in spin_rates:
    if spin_rpm in results:
        max_speed = results[spin_rpm]
        color = colors[spin_rates.index(spin_rpm)]
        
        # Calculate trajectory
        angle_rad = np.radians(angle)
        vx_initial = max_speed * np.cos(angle_rad)
        vy_initial = max_speed * np.sin(angle_rad)
        initial_state = [0, height_m, vx_initial, vy_initial]
        
        t = np.linspace(0, 10, 2000)
        solution = odeint(projectile_motion_with_drag_and_magnus, initial_state, t, args=(spin_rpm,))
        
        # Find trajectory until ground
        ground_idx = np.where(solution[:, 1] <= 0)[0]
        if len(ground_idx) > 0:
            end_idx = ground_idx[0]
            x_traj_ft = solution[:end_idx, 0] * M_TO_FT
            y_traj_ft = solution[:end_idx, 1] * M_TO_FT
            
            plt.plot(x_traj_ft, y_traj_ft, color=color, linewidth=2, 
                    label=f'{spin_rpm} RPM ({max_speed * MS_TO_MPH:.1f} mph)')

# Add court features to trajectory plot
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
plt.title(f'Tennis Ball Trajectories at Maximum Speed ({angle}° angle, {height_m} m height)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 85)
plt.ylim(0, 9)

plt.tight_layout()
plt.savefig('Magnus_Effect_Analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed results
print("\n" + "="*60)
print("DETAILED RESULTS - 5° ANGLE SHOTS")
print("="*60)
print(f"Contact Height: {height_ft} feet ({height_m:.2f} m)")
print(f"Launch Angle: {angle}°")
print("\nSpin Rate vs Maximum Speed:")
print("-" * 30)

valid_spins = [spin for spin in spin_rates if spin in results]
valid_speeds = [results[spin] * MS_TO_MPH for spin in valid_spins]

for spin_rpm in spin_rates:
    if spin_rpm in results:
        max_speed_mph = results[spin_rpm] * MS_TO_MPH
        print(f"{spin_rpm:4d} RPM → {max_speed_mph:5.1f} mph")

print("\nKey Findings:")
print("- Higher topspin allows significantly higher maximum speeds")
print("- Topspin creates downward Magnus force, pulling ball into court")
print("- Effect is most pronounced at higher spin rates (3000+ RPM)")
print("- Professional players typically use 2000-4000 RPM for groundstrokes")