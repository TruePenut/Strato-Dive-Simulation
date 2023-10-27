import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
M = 5.972e24  # Mass of the Earth (kg)
R = 6371000  # Radius of the Earth (m)
h_0 = 38969.4  # Initial altitude above the Earth's surface (m)
A_person = 0.7  # Cross-sectional area of the falling person (m^2)
A_parachute = 18.0  # Cross-sectional area of the deployed parachute (m^2)
C_d_person = 0.62  # Drag coefficient for the person (for a sphere)
C_d_parachute = 3  # Drag coefficient for the deployed parachute
m = 84.0  # Mass of the person (kg)

# Define the function for gravity at altitude h
def gravity(h):
    if h < 0 :
        return 0
    return G * (M / (R + h)**2)

# Define the air density function
def air_density(h):
    rho = 0
    # Calculate air density based on altitude using the provided equations for different altitude ranges
    if 0 <= h <= 11000:
        # Troposphere (linear temperature decrease, exponential pressure decrease)
        T = 15.04 - 0.00649 * h
        p = 101.29 * ((T + 273.1) / 288.08)**5.256
        rho = p / (0.2869 * (T + 273.1))
    elif 11000 < h <= 25000:
        # Lower Stratosphere (constant temperature, exponential pressure decrease)
        T = -56.46
        p = 22.65 * np.exp(1.73 - 0.000157 * h)
        rho = p / (0.2869 * (T + 273.1))
    elif h > 25000:
        # Upper Stratosphere (temperature increase, exponential pressure decrease)
        T = -131.21 + 0.00299 * h
        p = 2.488 * ((T + 273.1) / 216.6)**-11.388
        rho = p / (0.2869 * (T + 273.1))
    else:
        p = 0  # Set density to 0 for altitudes below 0

    # Calculate air density from pressure and temperature

    return rho

# Define the total acceleration function including air drag
def total_acceleration(y, t, use_parachute, use_airdrag):
    h, v = y

    # Calculate air density at the current altitude
    rho = air_density(h)

    # Update the cross-sectional area based on parachute deployment
    if use_parachute:
        if t > 259:
            A = A_parachute
            C_d = C_d_parachute
        else:
            A = A_person
            C_d = C_d_person
    else:
        A = A_person
        C_d = C_d_person

    # Calculate acceleration due to drag and gravity
    a_drag = (-0.5 * rho * A * C_d * v**2) / m 
    a_gravity = gravity(h) # m/s^2

    # Total acceleration is the sum of acceleration due to gravity and drag
    if use_airdrag:
        a_total = -a_gravity - a_drag
    else:
        a_total = -a_gravity

    return [v, a_total]

# Time points
t = np.linspace(0, 600, 10000)  # Adjust time points as needed

# Initial conditions
y0 = [h_0, 0]  # Initial altitude and velocity

# Solve the differential equations using odeint for each scenario with air drag
y_with_parachute_airdrag = odeint(total_acceleration, y0, t, args=(True, True))
y_without_parachute_with_airdrag = odeint(total_acceleration, y0, t, args=(False, True))
y_without_parachute_and_airdrag = odeint(total_acceleration, y0, t, args=(False, False))

# Extract height, velocity, and acceleration data for scenarios with air drag
h_with_parachute_airdrag = y_with_parachute_airdrag[:, 0]
v_with_parachute_airdrag = y_with_parachute_airdrag[:, 1]
h_without_parachute_airdrag = y_without_parachute_with_airdrag[:, 0]
v_without_parachute_airdrag = y_without_parachute_with_airdrag[:, 1]
h_without_parachute_and_airdrag = y_without_parachute_and_airdrag[:, 0]
v_without_parachute_and_airdrag = y_without_parachute_and_airdrag[:, 1]

# Calculate acceleration for all scenarios
a_with_parachute_airdrag = np.gradient(v_with_parachute_airdrag, t)
a_without_parachute_airdrag = np.gradient(v_without_parachute_airdrag, t)
a_without_parachute_and_airdrag = np.gradient(v_without_parachute_and_airdrag, t)

# Create plots for height, velocity, and acceleration for scenarios with air drag
plt.figure(figsize=(15, 5))

plt.subplot(2, 1, 2)
plt.plot(t, h_with_parachute_airdrag, label='With Parachute and Air Drag', linestyle='-')
plt.plot(t, h_without_parachute_airdrag, label='Without Parachute With Air Drag', linestyle='--')
plt.plot(t, h_without_parachute_and_airdrag, label='Without Parachute and Air Drag', linestyle='--')
plt.axvline(x=260, color='red', linestyle='--', label='Parachute Deployment')  # Vertical line at 260 seconds
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.ylim(0, max(h_with_parachute_airdrag) + 500)  # Adjust the maximum value as needed
plt.legend()
plt.title('Descent of a Person - Height vs. Time with Air Drag')

plt.subplot(2, 1, 1)
plt.plot(t, abs(v_with_parachute_airdrag), label='With Parachute and Air Drag', linestyle='-')
plt.plot(t, abs(v_without_parachute_airdrag), label='Without Parachute With Air Drag', linestyle='--')
plt.plot(t, abs(v_without_parachute_and_airdrag), label='Without Parachute and Air Drag', linestyle='--')
plt.axvline(x=260, color='red', linestyle='--', label='Parachute Deployment')  # Vertical line at 260 seconds
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.ylim(0, (max(abs(v_with_parachute_airdrag))+50))  # Adjust the maximum value as needed
plt.legend()
plt.title('Descent of a Person - Absolute Velocity vs. Time with Air Drag')


plt.tight_layout()
plt.show()
