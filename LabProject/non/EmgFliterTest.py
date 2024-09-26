import numpy as np
import matplotlib.pyplot as plt

# Comb filter function based on the formula y(k) = x(k) - x(k-N)
def comb_filter(signal, delay_samples):
    # Create a delayed version of the signal
    delayed_signal = np.zeros(len(signal))
    delayed_signal[delay_samples:] = signal[:-delay_samples]  # Apply delay
    
    # Comb filter output is the original signal minus the delayed signal
    filtered_signal = signal - delayed_signal
    
    return filtered_signal

# Create a test signal: a sinusoidal wave
fs = 1000  # Sampling frequency (samples per second)
t = np.linspace(0, 1, fs, False)  # 1 second time vector
freq = 5  # Frequency of the sinusoidal signal (Hz)
signal = np.sin(2 * np.pi * freq * t)

# Apply comb filter
delay_samples = int(0.01 * fs)  # Delay of 10 milliseconds
filtered_signal = comb_filter(signal, delay_samples)


raw_data = pd.read_csv(r"D:\BenQ_Project\01_UR_lab\2024_07 non-symmetry\2.EMG\S11\S11_LargeTrack_Rep_15.69.csv",
                         encoding='UTF-8')




# Plot the original and filtered signals
plt.figure(figsize=(10, 6))
plt.plot(t, signal, label="Original Signal")
plt.plot(t, filtered_signal, label="Filtered Signal (Comb Filter)")
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.title("Feed-forward Comb Filter (y(k) = x(k) - x(k-N))")
plt.show()
