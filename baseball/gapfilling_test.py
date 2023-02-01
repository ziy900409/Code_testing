import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt 

X = motion_data.iloc[2, :]
Y = motion_data.iloc[3, :]
X_vals = np.linspace(len(motion_data.iloc[2, :])
frequency = interpolate.interp1d(X, Y,'cubic')
Y_vals = frequency(X_vals)
plt.plot(X_vals, Y_vals, '-X')
 

import numpy as np
import matplotlib.pyplot as plt
x_pts = np.linspace(0, 2*np.pi, 100)
y_pts = np.sin(x_pts)
x_vals = np.linspace(0, 2*np.pi, 50)
y_vals = np.interp(x_vals, x_pts, y_pts)
plt.plot(x_vals, y_vals, '-x')
plt.plot(x_pts, y_pts, 'o')
# =============================================================================
# A = np.linspace(0, 2*np.pi, 10)
# B = np.sin(A)
# X_vals = np.linspace(0, 2*np.pi, 50)
# f = interpolate.interp1d(A, B,'cubic')
# Y_vals = f(X_vals)
# plt.plot(X_vals, Y_vals, '-X')
# =============================================================================
