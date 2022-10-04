data = pd.read_csv(cvs_file_list,encoding='UTF-8')
# to define data time
data_time = data.iloc[:,0]
Fs = 1/(data_time[2]-data_time[1]) # sampling frequency
data = pd.DataFrame(data)
# need to change column name or using column number
EMG_data = data.iloc[:, [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]]
    # exchange data type to float64
EMG_data = [pd.to_numeric(EMG_data.iloc[:, i], errors = 'coerce') 
            for i in range(np.shape(EMG_data)[1])]
EMG_data = pd.DataFrame(np.transpose(EMG_data),
            columns=data.iloc[:, [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]].columns)
    # bandpass filter use in signal
bandpass_sos = signal.butter(2, [20, 500],  btype='bandpass', fs=Fs, output='sos')
    
bandpass_filtered_data = np.zeros(np.shape(EMG_data))
for i in range(np.shape(EMG_data)[1]):
        # print(i)
        # using dual filter to processing data to avoid time delay
    bandpass_filtered = signal.sosfiltfilt(bandpass_sos, EMG_data.iloc[:,i])
    bandpass_filtered_data[:, i] = bandpass_filtered 
    
    # caculate absolute value to rectifiy EMG signal
bandpass_filtered_data = abs(bandpass_filtered_data)     
    # -------Data smoothing. Compute RMS
    # The user should change window length and overlap length that suit for your experiment design
    # window width = window length(second)//time period(second)
window_width = int(0.1/(1/np.floor(Fs)))
moving_data = np.zeros([int(np.shape(bandpass_filtered_data)[0] / window_width),
                        np.shape(bandpass_filtered_data)[1]])
for i in range(np.shape(moving_data)[1]):
    for ii in range(np.shape(moving_data)[0]):
        data_location = ii
        moving_data[int(data_location), i] = (np.sum(bandpass_filtered_data[ii*(ii+1):(ii+window_width)*(ii+1), i]) 
                                              /window_width)
    # ------linear envelop analysis-----------                          
    # ------lowpass filter parameter that the user must modify for your experiment        
lowpass_sos = signal.butter(2, 6, btype='low', fs=Fs, output='sos')        
lowpass_filtered_data = np.zeros(np.shape(bandpass_filtered_data))
for i in range(np.shape(moving_data)[1]):
    lowpass_filtered = signal.sosfiltfilt(lowpass_sos, bandpass_filtered_data[:,i])
    lowpass_filtered_data[:, i] = lowpass_filtered
    # add columns name to data frame
bandpass_filtered_data = pd.DataFrame(bandpass_filtered_data, columns=EMG_data.columns)
moving_data = pd.DataFrame(moving_data, columns=EMG_data.columns)
lowpass_filtered_data = pd.DataFrame(lowpass_filtered_data, columns=EMG_data.columns)
    # insert time data in the DataFrame
lowpass_filtered_data.insert(0, 'time', data_time)
moving_time_index = np.linspace(0, np.shape(data_time)[0]-1, np.shape(moving_data)[0])
moving_time_index = moving_time_index.astype(int)
moving_data.insert(0, 'time', data_time.iloc[moving_time_index])
bandpass_filtered_data.insert(0, 'time', data_time)  
