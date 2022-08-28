import matplotlib.pyplot as plt
import numpy as np

vec_noisy = np.array(noisy_synth_data)
plt.figure(figsize=(7.5, 5))
mean_line = np.mean(vec_noisy,axis = 0 )
lower_line = mean_line - 2*np.std(vec_noisy,axis = 0 )
higher_line = mean_line + 2*np.std(vec_noisy,axis = 0 )
plt.plot(mtu_times, vec_noisy[:,4],'gx', label="Y^2_bar")
plt.plot(mtu_times, mean_line[4]*np.ones(30),'r', linewidth = 2)
plt.plot(mtu_times, lower_line[4]*np.ones(30), 'b-.')
plt.plot(mtu_times, higher_line[4]*np.ones(30), 'b-.')
plt.legend(loc=0)
plt.xlabel("i-th Intervals")
plt.ylabel("Y^2_bar Values")
plt.show()
