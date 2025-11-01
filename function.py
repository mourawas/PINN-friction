import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadData(folder_path, adjust, window_size, window_size_pos, experiment_number,start_time, end_time, Plot_prepare, select_time):


    # Read the first CSV file (displacement of the stage)
    stage_df = pd.read_csv(folder_path+'F2Sweep Up and Down 100kHz to  1 MHZ In 1 ms'+experiment_number+'.csv', skiprows=5, header=None, names=['Time', 'Displacement'])
    # Read the second CSV file (speed of the mobile)
    mobile_df = pd.read_csv(folder_path+'F1Sweep Up and Down 100kHz to  1 MHZ In 1 ms'+experiment_number+'.csv', skiprows=5, header=None, names=['Time', 'Speed_raw'])
    stage_df['Displacement'] = -stage_df['Displacement']
    # The mobile speed is set to 0, we remove any constant offset that can be present
    mean_first_1000 = mobile_df['Speed_raw'][:1000].mean()
    mobile_df['Speed_raw'] -= mean_first_1000
    mobile_df['Speed_raw'] *= 0.125 # conversion to m/s
    if Plot_prepare:
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Increase spacing between subplots
        axs[2, 0].plot(mobile_df['Time'], stage_df['Displacement'], label='stage disp', linewidth=2, color='r')
        axs[2, 0].plot(mobile_df['Time'], mobile_df['Speed_raw'], label='mobile speed', linestyle=':', color='b')
        axs[2, 0].set_title('Raw data')
        axs[2, 0].set_xlabel('time [us]')
        axs[2, 0].set_ylabel('values')
        axs[2, 0].legend()
        axs[2, 0].grid(True)

    stage_df['Displacement'] = np.convolve(stage_df['Displacement'], np.ones(window_size_pos)/window_size_pos, mode='same')
    # Time window zooming
    if select_time:
        mobile_df = mobile_df[(mobile_df['Time'] >= start_time) & (mobile_df['Time'] <= end_time)]
        stage_df = stage_df[(stage_df['Time'] >= start_time) & (stage_df['Time'] <= end_time)]
        mobile_df['Time'] = mobile_df['Time'] + start_time
        stage_df['Time'] = stage_df['Time'] + start_time

    #smooth the accelration which comes fronm the speed derivative
    mobile_df['Acceleration_raw'] = np.gradient(mobile_df['Speed_raw'], mobile_df['Time'])


    # Compute the stage speed
    stage_df['Speed_raw'] = np.gradient(stage_df['Displacement'], stage_df['Time'])/2000 # m/s (V/mm->mm/s) timestep is 2ms
    # shift in tame the stage speed
    stage_df['Speed_raw'] = stage_df['Speed_raw'].shift(adjust)
    # Drop the last rows to remove NaN values introduced by the shift
    stage_df = stage_df.dropna(subset=['Speed_raw'])
    # Trim the mobile_df to match the size of the stage_df
    mobile_df = mobile_df.iloc[:len(stage_df)]

    # Apply a moving average filter to smooth
    stage_df['Speed_smoothed'] = np.convolve(stage_df['Speed_raw'], np.ones(window_size)/window_size, mode='same')
    mobile_df['Speed_smoothed'] = np.convolve(mobile_df['Speed_raw'], np.ones(window_size)/window_size, mode='same')
    mobile_df['Acceleration_smoothed'] = np.convolve(mobile_df['Acceleration_raw'], np.ones(window_size)/window_size, mode='same')/50


    time = mobile_df['Time']
    rel_speed = -mobile_df['Speed_smoothed'] + stage_df['Speed_smoothed']

    if Plot_prepare:
    # Plot the internal state variables and friction force vs relative velocity in a 3x2 grid
        

        # Create twin axis for the first subplot
        ax1 = axs[0, 0]
        ax2 = ax1.twinx()

        # Plot displacement on the first y-axis (left)
        line1 = ax1.plot(time, stage_df['Displacement_raw'], label='Position', color='r')
        ax1.set_ylabel('Displacement [mm]')

        # Plot speeds on the second y-axis (right)
        line2 = ax2.plot(time, stage_df['Speed_raw'], label='Velocity', color='c')
        line3 = ax2.plot(time, stage_df['Speed_smoothed'], label='Velocity smoothed', color='b')
        ax2.set_ylabel('Velocity [m.s⁻¹]')

        # Title and x-axis label
        ax1.set_title('Stage motion')
        ax1.set_xlabel('time [us]')

        # Combine legends from both axes
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels)

        # Add grid
        ax1.grid(True)

        # Plot Cube Motion under LuGre Friction
        axs[0, 1].plot(time, mobile_df['Speed_raw'], label='Velocity (m.s⁻¹)', color='c')
        axs[0, 1].plot(time, mobile_df['Speed_smoothed'], label='Velocity smoothed (m.s⁻¹)',color='darkviolet')
        axs[0, 1].set_title('Mobile motion')
        axs[0, 1].set_xlabel('time [us]')
        axs[0, 1].set_ylabel('Motion')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Plot Relative Velocity v_rel
        axs[1, 0].plot(time, stage_df['Speed_smoothed'], label='stage speed(m.s⁻¹)')
        axs[1, 0].plot(time, mobile_df['Speed_smoothed'], label='mobile speed(m.s⁻¹)')
        axs[1, 0].set_title('comparison stage and mobile')
        axs[1, 0].set_xlabel('time [us]')
        axs[1, 0].set_ylabel('Relative Velocity (m.s⁻¹)')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Create twin axis for the friction measurements plot
        ax1 = axs[1, 1]
        ax2 = ax1.twinx()

        # Plot acceleration on the left y-axis
        line1 = ax1.plot(time, mobile_df['Acceleration_smoothed'], label='Mobile acceleration', color='r')
        ax1.set_ylabel('Acceleration [m.s⁻²]')

        # Plot relative speed on the right y-axis
        line2 = ax2.plot(time, rel_speed, label='Relative speed', color='b')
        ax2.set_ylabel('Velocity [m.s⁻¹]')

        # Title and x-axis label
        ax1.set_title('Friction measurements')
        ax1.set_xlabel('time [µs]')

        # Combine legends from both axes
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels)

        # Add grid
        ax1.grid(True)

        axs[2, 1].plot(rel_speed, mobile_df['Acceleration_smoothed'], label='Mobile acceleration')
        axs[2, 1].set_title('mobile acceleration versus speed')
        axs[2, 1].set_xlabel('relative speed')
        axs[2, 1].set_ylabel('Friction Force (N)')
        axs[2, 1].legend()
        axs[2, 1].grid(True)

        plt.tight_layout()  # Automatically adjust subplot parameters for better fit

    return time, rel_speed, stage_df['Speed_smoothed'], mobile_df



def loadDatamatrix(position_stage_raw, mobile_speed_raw, time, adjust, window_size, window_size_pos, start, end, Plot_prepare, select_time, masse):

    mobile_df = {};
    stage_df = {};
    if select_time:
        time = time
        mobile_df['Speed_raw'] = -0.125*np.array(mobile_speed_raw[start:end]);
        mobile_df['Time'] = time[start:end]
        stage_df['Displacement_raw'] = np.array(position_stage_raw[start:end]);
        stage_df['Time'] = mobile_df['Time']
    else:
        mobile_df['Speed_raw'] = -0.125*np.array(mobile_speed_raw);
        mobile_df['Time'] = time
        stage_df['Displacement_raw'] = np.array(position_stage_raw);
        stage_df['Time'] = mobile_df['Time']

    #print(np.mean(mobile_df['Speed_raw'][1:20]))
    #mobile_df['Speed_raw'] = (mobile_df['Speed_raw'] - np.mean(mobile_df['Speed_raw'][1:20]))*0.05

    if Plot_prepare:
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Increase spacing between subplots
        
        """ ax1 = axs[2, 0]
        ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
        
        # Plot displacement on the first y-axis
        line1 = ax1.plot(mobile_df['Time'], stage_df['Displacement_raw'], label='stage disp', 
                        linewidth=2, color='r')
        ax1.set_ylabel('Displacement [mm]')
        
        # Plot speed on the second y-axis
        line2 = ax2.plot(mobile_df['Time'], mobile_df['Speed_raw'], label='mobile speed', 
                        linestyle=':', color='cyan')
        ax2.set_ylabel('Speed [mm.s⁻¹]')
        
        # Title and x-axis label
        ax1.set_title('Raw data')
        ax1.set_xlabel('time [us]')
        
        # Combine legends from both axes
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels)
        
        # Add grid
        ax1.grid(True)
 """



    # Time window zooming

    # Compute the stage speed
    stage_df['Speed_raw'] = 0.125*7800*np.gradient(stage_df['Displacement_raw'], stage_df['Time'])# m.s⁻¹ (V/mm->mm.s⁻¹) timestep is 2ms



    # Apply a moving average filter to smooth
    stage_df['Speed_smoothed'] = np.convolve(stage_df['Speed_raw'], np.ones(window_size_pos)/window_size_pos, mode='same')
    mobile_df['Speed_smoothed'] = np.convolve(mobile_df['Speed_raw'], np.ones(window_size)/window_size, mode='same')
    mobile_df['Speed_smoothed_cut'] =  mobile_df['Speed_smoothed'][1:-100]
    #smooth the accelration which comes fronm the speed derivative
    mobile_df['Acceleration_raw'] = np.gradient(mobile_df['Speed_smoothed'], mobile_df['Time'])

    mobile_df['Acceleration_smoothed'] = 1000000*np.convolve(mobile_df['Acceleration_raw'], np.ones(25)/25, mode='same')
    mobile_df['Acceleration_smoothed_cut'] = mobile_df['Acceleration_smoothed'][1:-100]
    time_cut = time[1:-100]
    stage_df['Speed_smoothed_cut'] = stage_df['Speed_smoothed'][1:-100]
    rel_speed = -mobile_df['Speed_smoothed'] + stage_df['Speed_smoothed']
    rel_speed_cut = rel_speed[1:-100]
    mobile_df['friction_force_cut']=mobile_df['Acceleration_smoothed_cut']*(masse*0.001)
    if Plot_prepare:
    # Plot the internal state variables and friction force vs relative velocity in a 3x2 grid

        # Create twin axis for the first subplot
        ax1 = axs[0, 0]
        ax2 = ax1.twinx()

        # Plot displacement on the first y-axis (left)
        line1 = ax1.plot(time, stage_df['Displacement_raw'], label='Position', color='skyblue')
        ax1.set_ylabel('Displacement [mm]', color='skyblue')

        # Plot speeds on the second y-axis (right)
        line2 = ax2.plot(time, stage_df['Speed_raw'], label='Velocity', color='royalblue')
        line3 = ax2.plot(time, stage_df['Speed_smoothed'], label='Velocity smoothed', color='b')
        ax2.set_ylabel('Speed [m.s⁻¹]', color='b')

        # Title and x-axis label
        ax1.set_title('Stage motion')
        ax1.set_xlabel('time [us]')

        # Combine legends from both axes
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels)

        # Add grid
        ax1.grid(True)

       # Plot Cube Motion under LuGre Friction
        axs[0, 1].plot(time, mobile_df['Speed_raw'], label='Velocity', color='violet')
        axs[0, 1].plot(time, mobile_df['Speed_smoothed'], label='Velocity smoothed',color='darkviolet')
        axs[0, 1].set_title('Mobile motion')
        axs[0, 1].set_xlabel('time [us]')
        axs[0, 1].set_ylabel('speed [m.s⁻¹]')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        # Plot Relative Velocity v_rel
        axs[1, 0].plot(time_cut, stage_df['Speed_smoothed_cut'], label='stage speed(m.s⁻¹)', color = 'b')
        axs[1, 0].plot(time_cut, mobile_df['Speed_smoothed_cut'], label='mobile speed(m.s⁻¹)', color='darkviolet')
        axs[1, 0].set_title('comparison speed stage and mobile')
        axs[1, 0].set_xlabel('time [us]')
        axs[1, 0].set_ylabel('Relative Speed (m.s⁻¹)')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Create twin axis for the friction measurements plot
        ax1 = axs[1, 1]
        ax2 = ax1.twinx()

        # Plot acceleration on the left y-axis
        line1 = ax1.plot(time_cut, mobile_df['Acceleration_smoothed_cut'], label='Mobile acceleration', color='r')
        ax1.set_ylabel('Acceleration [m.s⁻²]', color='r')

        # Plot relative speed on the right y-axis
        line2 = ax2.plot(time_cut, rel_speed_cut, label='Relative speed', color='black')
        ax2.set_ylabel('Velocity [m.s⁻¹]', color='black')

        # Get the limits for both axes
        y1_min, y1_max = ax1.get_ylim()
        y2_min, y2_max = ax2.get_ylim()

        # Calculate the positions of zero on both axes as a fraction of the total range
        y1_zero = -y1_min / (y1_max - y1_min)
        y2_zero = -y2_min / (y2_max - y2_min)

        # Set the limits to make zeros align
        if y1_zero > y2_zero:
            y2_range = y2_max - y2_min
            y2_min_new = -(y1_zero * y2_range) / (1 - y1_zero)
            ax2.set_ylim(y2_min_new, y2_max)
        else:
            y1_range = y1_max - y1_min
            y1_min_new = -(y2_zero * y1_range) / (1 - y2_zero)
            ax1.set_ylim(y1_min_new, y1_max)

        # Title and x-axis label
        ax1.set_title('Friction measurements')
        ax1.set_xlabel('time [µs]')

        # Combine legends from both axes
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels)

        # Add grid
        ax1.grid(True)

        axs[2, 1].plot(rel_speed_cut, mobile_df['friction_force_cut'], label='Mobile acceleration', marker='.', linestyle='None')
        axs[2, 1].set_title('mobile acceleration versus speed')
        axs[2, 1].set_xlabel('relative speed[m.s⁻¹]')
        axs[2, 1].set_ylabel('Friction Force (N)')
        axs[2, 1].legend()
        axs[2, 1].grid(True)


        # Plot Mobile acceleration versus speed in a separate window
        plt.figure()  # Create a new figure for the separate window
        plt.plot(rel_speed_cut[1:-1:2], mobile_df['Acceleration_smoothed_cut'][1:-1:2], label='Mobile acceleration', marker='.', linestyle='None')
        plt.title('Mobile acceleration versus speed')
        plt.xlabel('Relative speed')
        plt.ylabel('Friction Force (N)')
        plt.legend()
        plt.grid(True)
        for i in range(0,len(rel_speed_cut),16):
            plt.text(rel_speed[i], mobile_df['Acceleration_smoothed'][i], str(i), fontsize=8, ha='right')
        plt.show()  # Show the new figure

        plt.tight_layout()  # Automatically adjust subplot parameters for better fit

    return time_cut,rel_speed_cut, stage_df, mobile_df