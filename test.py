import matplotlib.pyplot as plt
import numpy as np

# Generate a random image (you can replace this with your image data)
image_data = np.random.random((100, 100))

# Plot the image
plt.imshow(image_data, cmap='gray')  # You can specify the colormap if needed
plt.axis('off')  # Turn off axis
plt.savefig("tset.png")
