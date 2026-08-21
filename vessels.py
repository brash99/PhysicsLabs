import math

# this is a program to calculate the time to empty a slanted cylindrical vessel
# given the dimensions of the vessel and the initial height of the liquid

def time_to_empty_slanted_cylinder(small_radius, large_radius,initial_height,inverted=False):
    time_array = []
    height_array = []
    volume_array = []

    initial_volume = 1./3.*math.pi*initial_height*(large_radius**2 + large_radius*small_radius + small_radius**2)
    print(f"Initial Volume: {initial_volume:.3f} m^3")

    dt = 0.5  # time step in seconds
    g = 9.81  # acceleration due to gravity in m/s^2
    time = 0.0
    area_nozzle = math.pi*0.01**2  # assuming outflow through a nozzle of radius 0.01 m

    current_height = initial_height
    current_volume = initial_volume

    while current_height > 0:
        time_array.append(time)
        height_array.append(current_height)
        volume_array.append(current_volume)
        if inverted:
            radius_at_height = large_radius + (small_radius - large_radius) * (current_height / initial_height)
        else:
            radius_at_height = small_radius + (large_radius - small_radius) * (current_height / initial_height)
        area_at_height = math.pi * radius_at_height**2
        outflow_velocity = math.sqrt(2 * g * current_height)
        outflow_volume = outflow_velocity * dt * area_nozzle

        current_volume -= outflow_volume
        if current_volume < 0:
            current_volume = 0

        current_height = current_height - (outflow_volume / area_at_height)

        time += dt
        print(f"Time: {time:.3f} s, Height: {current_height:.3f} m, Volume: {current_volume:.3f} m^3, Radius at Height: {radius_at_height:.3f} m")

    return time, time_array, height_array, volume_array

# Example usage:
small_radius = 0.4625  # in meters
large_radius = 0.66 # in meters
initial_height = 1.0  # in meters

time_needed_y, time_y, h_y, volume_y = time_to_empty_slanted_cylinder(small_radius, large_radius, initial_height, True)
print(f"Time to empty the vessel Y: {time_needed_y:.3f} seconds")
print("-----------------------------------")

time_needed_x, time_x, h_x, volume_x = time_to_empty_slanted_cylinder(small_radius, large_radius, initial_height, False)
print(f"Time to empty the vessel X: {time_needed_x:.3f} seconds")

# make plots
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time_y, h_y, label='Height Y', color='blue')
plt.plot(time_x, h_x, label='Height X', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.title('Height vs Time')
plt.legend()
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(time_y, volume_y, label='Volume Y', color='blue')
plt.plot(time_x, volume_x, label='Volume X', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Volume (m^3)')
plt.title('Volume vs Time')
plt.legend()
plt.grid()
# also plot output flow rate vs time
plt.subplot(3, 1, 3)
outflow_rate_y = [ (volume_y[i-1] - volume_y[i]) / 50.0 for i in range(1, len(volume_y))]
outflow_rate_x = [ (volume_x[i-1] - volume_x[i]) / 50.0 for i in range(1, len(volume_x))]
plt.plot(time_y[1:], outflow_rate_y, label='Outflow Rate Y', color='blue')
plt.plot(time_x[1:], outflow_rate_x, label='Outflow Rate X', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Outflow Rate (m^3/s)')
plt.title('Outflow Rate vs Time')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()




