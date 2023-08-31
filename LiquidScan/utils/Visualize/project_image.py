import numpy as np
import plotly.graph_objects as go
import skimage.io as sio
import cv2

x = np.linspace(-10, 10, 600)
x, z = np.meshgrid(x,x)
#print(x)
y = np.sin(x**2*z)
Y = 0.5 * np.ones(y.shape)


image = cv2.imread("../dataset\interior_mode\pipes\images\DSC_0647.JPG", cv2.IMREAD_GRAYSCALE)
print(image.shape)
image = cv2.resize(image, (600, 600))
#img = image[:,:, 1] 

fig = go.Figure(go.Surface(x=x, y=Y, z=z,
                           surfacecolor = np.flipud(image),
                           colorscale='gray', 
                           showscale=False))
"""
fig.add_surface(x=x, y=Y, z=z, 
                surfacecolor=np.flipud(img), 
                colorscale='gray', 
                showscale=False)
"""
fig.update_layout(width=800, height=800, 
                  scene_camera_eye_z=0.6, 
                  scene_aspectratio=dict(x=0.9, y=1, z=1))
fig.show()