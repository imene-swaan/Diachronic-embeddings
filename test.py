# Redo the black dashed line as an exponential curve on the same plot settings as before.

import numpy as np
# import matplotlib.pyplot as plt

# c3_noise = np.random.rand(300)
# c3_colors = np.random.rand(300,4)

# plt.figure(figsize=(6, 6))  # Making the plot square
# x = np.linspace(0, 60, 300)


# # Plotting each point as a vertical line to simulate a histogram
# for i in range(len(x)):
#     plt.vlines(x[i], 0, c3_noise[i], colors=c3_colors[i])

# # Create an exponential curve for the dashed line
# c3_exponential_line = np.exp(0.1 * x) - 1  # This creates an exponential growth starting from 0
# c3_exponential_line /= np.max(c3_exponential_line)  # Normalize to max at 1

# plt.plot(x, c3_exponential_line, 'k--', linewidth= 4)  # Exponential dashed line

# # Remove ticks and labels
# plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

# # Set the limits to ensure the dashed line reaches the top left edge
# plt.xlim(0, 60)
# plt.ylim(0, 1)

# plt.xlabel('Time', fontsize=24)
# plt.ylabel('Frequency', fontsize=24)

# plt.title('C3', fontsize=28)

# # Add Arrows to the end of the axes
# # plt.annotate('', xy=(60, 0), xytext=(0, 0), arrowprops=dict(facecolor='black', arrowstyle='->'))
# # plt.annotate('', xy=(0, 1.1), xytext=(0, 0), arrowprops=dict(facecolor='black', arrowstyle='->'))

# plt.grid(True)
# plt.tight_layout()

# # save
# plt.savefig('test.png', dpi=300)


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm

# # Generate data for a narrow normal distribution skewed to the left
# mu = 1  # mean
# sigma = 0.005  # standard deviation
# # skewness = -5  # skewness parameter
# size = 1000  # number of data points

# data = np.random.normal(loc=mu, scale=sigma, size=size)
# # data = np.power(data, skewness)

# # Calculate the probability density function
# x = np.linspace(0, 5, 100)
# pdf = (1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))) # * np.abs(skewness)

# # Plot the probability density function
# plt.plot(x, pdf, color='g', linewidth=2)
# plt.title('Narrow Left Skewed Normal Distribution')
# plt.xlabel('X')
# plt.ylabel('Density')
# plt.grid(True)

# plt.savefig('test.png', dpi=300)

# import matplotlib.pyplot as plt
# import numpy as np

# # Create data for the normal distribution
# x = np.linspace(-5, 5, 1000)
# y = 1/(np.sqrt(2*np.pi*0.5**2)) * np.exp(-(x-2)**2/(2*0.5**2))  # Narrow normal distribution with mean close to 0

# # Create the plot
# plt.figure(figsize=(6, 6))

# # Plot the orange straight horizontal line
# plt.plot([-5, 60], [0.6, 0.6], color='orange', linewidth=4, label='Horizontal Line from 0 to 60')

# # Plot the narrow normal distribution
# plt.plot(x, y, 'k--', linewidth=4)

# # Add labels and legend
# plt.xlabel('Time', fontsize=24)
# plt.ylabel('Frequency', fontsize=24)

# plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
# plt.title('D2', fontsize=28)
# # Show the plot
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('test.png', dpi=300)





import matplotlib.pyplot as plt
import numpy as np

# Define the parameters for the first normal distribution
x1 = np.linspace(-5, 5, 1000)
y1 = 1/(np.sqrt(2*np.pi*0.5**2)) * np.exp(-(x1+3)**2/(2*0.5**2))  # First normal distribution with mean 2

# Define the parameters for the second normal distribution
x2 = np.linspace(-5, 5, 1000)
y2 = 1/(np.sqrt(2*np.pi*0.5**2)) * np.exp(-(x2+1)**2/(2*0.5**2))  # Second normal distribution with mean -2

# Define the parameters for the third normal distribution
x3 = np.linspace(-5, 5, 1000)
y3 = 1/(np.sqrt(2*np.pi*0.5**2)) * np.exp(-(x3-1)**2/(2*0.5**2))  # Third normal distribution with mean 0

# Define the parameters for the fourth normal distribution
x4 = np.linspace(-5, 5, 1000)
y4 = 1/(np.sqrt(2*np.pi*0.5**2)) * np.exp(-(x4-3)**2/(2*0.5**2))  # Fourth normal distribution with mean 0

# Create the plot
plt.figure(figsize=(6, 6))

plt.plot([min(x1), 5], [0.6, 0.6], color='orange', linewidth=4)

# Plot the first normal distribution
plt.plot(x1, y1, 'k--', linewidth=3)

# Plot the second normal distribution
plt.plot(x2, y2, 'k--', linewidth=3)

# Plot the third normal distribution
plt.plot(x3, y3, 'k--', linewidth=3)

# Plot the fourth normal distribution
plt.plot(x4, y4, 'k--', linewidth=3)

plt.xlim(-5, 5)
# plt.ylim(0, 1)
# Add labels and legend
plt.xlabel('Time', fontsize=24)
plt.ylabel('Frequency', fontsize=24)
plt.title('D3', fontsize=28)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

# Show the plot
plt.grid(True)
plt.savefig('test.png', dpi=300)
