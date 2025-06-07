import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from function import loadDatamatrix
import os

#use m4, m5, m6 as datasets

PATH = 'plots pinn'
os.makedirs(PATH, exist_ok=True)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.hidden1 = nn.Linear(3, 64)  # input v, direction, and position
        self.ln1 = nn.LayerNorm(64)
        self.dropout1 = nn.Dropout(0.3)
        self.hidden2 = nn.Linear(64, 128)
        self.ln2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(0.3)
        self.hidden3 = nn.Linear(128, 128)
        self.ln3 = nn.LayerNorm(128)
        self.dropout3 = nn.Dropout(0.3)
        self.output = nn.Linear(128, 1)  # output z

    def forward(self, x):
        x = self.hidden1(x)
        x = self.ln1(x)
        x = torch.tanh(x)
        x = self.dropout1(x)
        
        x = self.hidden2(x)
        x = self.ln2(x)
        x = torch.tanh(x)
        x = self.dropout2(x)
        
        x = self.hidden3(x)
        x = self.ln3(x)
        x = torch.tanh(x)
        x = self.dropout3(x)
        return self.output(x)

class BoundedParameter:
    def __init__(self, init_value, min_val, max_val):
        init_value_clipped = max(min(init_value, max_val), min_val)
        
        # map to [-1, 1]
        normalized = 2 * (init_value_clipped - min_val) / (max_val - min_val) - 1
        normalized = max(min(normalized, 0.999), -0.999)  # avoid infinities
        
        # inverse tanh: arctanh(x) = 0.5 * log((1+x)/(1-x))
        normalized_tensor = torch.tensor(normalized, dtype=torch.float32)
        unbounded = 0.5 * torch.log((1 + normalized_tensor) / (1 - normalized_tensor))
        
        self.unbounded = nn.Parameter(unbounded)
        self.min_val = min_val
        self.max_val = max_val
        
    def get_value(self):
        # tanh maps (-inf,inf) to (-1,1)
        bounded_m1_1 = torch.tanh(self.unbounded)
        # scale to [min_val, max_val]
        return self.min_val + (self.max_val - self.min_val) * (bounded_m1_1 + 1) / 2
    
    def item(self):
        # convenience method to get value as a Python scalar
        return self.get_value().item()

class Friction:
    def __init__(self, time_data, velocity_data, friction_data, learning_rate=0.001):
        
        # dt
        self.t_data = torch.tensor(time_data, dtype=torch.float32).view(-1, 1)
        self.dt = time_data[1] - time_data[0]

        # normalize velocity
        self.v_mean, self.v_std = np.mean(velocity_data), np.std(velocity_data)
        norm_velocity = (velocity_data - self.v_mean) / self.v_std
        self.v_data = torch.tensor(norm_velocity, dtype=torch.float32).view(-1, 1)
        
        # calculate position by integrating velocity
        self.position = np.zeros_like(velocity_data)
        for i in range(1, len(velocity_data)):
            self.position[i] = self.position[i-1] + velocity_data[i-1] * self.dt
        
        # use position=2 for masking
        # find first index where position exceeds 2
        mask_indices = np.where(self.position >= 2.0)[0]
        if len(mask_indices) > 0:
            self.first_peak_idx = mask_indices[0]
            print(f"Position >= 2 found at index {self.first_peak_idx}, value: {self.position[self.first_peak_idx]}")
        else:
            # edge case
            self.first_peak_idx = len(self.position) // 4
            print(f"Position never reaches 2. Using default index {self.first_peak_idx}")
        
        # mask position
        self.masked_position = np.copy(self.position)
        if self.first_peak_idx + 1 < len(self.position):
            self.masked_position[self.first_peak_idx + 1:] = self.position[self.first_peak_idx + 1]
        
        # normalize the masked position
        self.pos_mean, self.pos_std = np.mean(self.masked_position), np.std(self.masked_position)
        norm_position = (self.masked_position - self.pos_mean) / self.pos_std
        self.pos_data = torch.tensor(norm_position, dtype=torch.float32).view(-1, 1)
        
        # calculate direction based on velocity changes
        # find max velocity index
        max_vel_idx = np.argmax(velocity_data)
        
        # find min velocity index (after max velocity)
        # captures the second direction change
        min_vel_after_max_idx = max_vel_idx + np.argmin(velocity_data[max_vel_idx:])
        
        # three state direction:
        # 0: initial go phase (start to max velocity)
        # 1: first return phase (max velocity to min velocity)
        # 2: second return" phase (min velocity to end)
        direction = np.zeros_like(velocity_data)
        direction[max_vel_idx:min_vel_after_max_idx] = 1
        direction[min_vel_after_max_idx:] = 2
        
        # normalize direction to [0,1] for neural network
        direction = direction / 2.0
        
        self.direction_data = torch.tensor(direction, dtype=torch.float32).view(-1, 1)
        self.max_vel_idx = max_vel_idx
        self.min_vel_after_max_idx = min_vel_after_max_idx
        
        # combine velocity, direction, and position into a single tensor for input
        self.v_dir_pos_data = torch.cat((self.v_data, self.direction_data, self.pos_data), dim=1)
        
        self.v_data_orig = torch.tensor(velocity_data, dtype=torch.float32).view(-1, 1)
        self.F_data_orig = torch.tensor(friction_data, dtype=torch.float32).view(-1, 1)
        self.pos_data_orig = torch.tensor(self.position, dtype=torch.float32).view(-1, 1)
        
        self.model = Network()
        
        # physics parameters with bounds
        # set min and max values for each parameter based on physical constraints
        self.sigma0 = BoundedParameter(2100.0, min_val=1000.0, max_val=7000.0)
        self.sigma1 = BoundedParameter(0.2, min_val=0.1, max_val=1.0)
        self.sigma2 = BoundedParameter(1.9, min_val=0.01, max_val=5.0)
        self.Fc = BoundedParameter(0.45, min_val=0.1, max_val=1.0)
        self.Fs = BoundedParameter(0.05, min_val=0.01, max_val=1.0)
        self.vs = BoundedParameter(0.09, min_val=0.05, max_val=1.0)
    
    def g_function(self, v):
        # get bounded parameters for computation
        sigma0 = self.sigma0.get_value()
        Fc = self.Fc.get_value()
        Fs = self.Fs.get_value()
        vs = self.vs.get_value()
        
        return (Fc + (Fs - Fc) * torch.exp(-(v/vs)**2)) / sigma0
    
    def train(self, epochs=10000, lr=0.001):
        # parameter lists for optimizer - both neural network and bounded parameters
        model_params = list(self.model.parameters())
        physics_params = [
            self.sigma0.unbounded, 
            self.sigma1.unbounded, 
            self.sigma2.unbounded, 
            self.Fc.unbounded, 
            self.Fs.unbounded, 
            self.vs.unbounded
        ]
        
        # optimizer with all parameters
        optimizer = torch.optim.Adam(model_params + physics_params, lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

        loss_history = []
        data_loss_history = []
        physics_loss_history = []
        param_history = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            # use combined velocity, direction, and position as input
            z_pred = 0.00001 * self.model(self.v_dir_pos_data)
            
            # numerical derivative
            dzdt_pred = torch.zeros_like(z_pred)
            
            # for derivatives we need to create the next combined input
            v_dir_pos_next = self.v_dir_pos_data[1:].clone()
            dzdt_pred[:-1] = 0.00001 * (self.model(v_dir_pos_next) - self.model(self.v_dir_pos_data[:-1])) / self.dt
            
            # last value
            dzdt_pred[-1] = dzdt_pred[-2]

            # get bounded parameter values for computation
            sigma0 = self.sigma0.get_value()
            sigma1 = self.sigma1.get_value()
            sigma2 = self.sigma2.get_value()

            # using original velocity values
            g_v = self.g_function(self.v_data_orig)
            
            # in original units
            F_pred = sigma0 * z_pred + sigma1 * dzdt_pred + sigma2 * self.v_data_orig
            
            # in original units
            data_loss = torch.mean((F_pred - self.F_data_orig)**2)
            
            # in original units
            residual = dzdt_pred - (self.v_data_orig - ((torch.abs(self.v_data_orig))/(g_v + 1e-6))*z_pred)
            physics_loss = torch.mean(residual**2)

            # loss
            loss = 30*data_loss + physics_loss

            loss.backward()
            optimizer.step()

            if 4000 < epoch < 10000:
                scheduler.step()

            loss_history.append(loss.item())
            data_loss_history.append(data_loss.item())
            physics_loss_history.append(physics_loss.item())
            
            # store parameter values
            if epoch % 100 == 0:
                param_history.append({
                    'sigma0': self.sigma0.item(),
                    'sigma1': self.sigma1.item(),
                    'sigma2': self.sigma2.item(),
                    'Fc': self.Fc.item(),
                    'Fs': self.Fs.item(),
                    'vs': self.vs.item(),
                })

            if epoch % 5 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.6f}, Data Loss: {data_loss.item():.6f}, Physics Loss: {physics_loss.item():.6f}')
                print(f'Learning Rate: {scheduler.get_last_lr()[0]}')
                print(f'Parameters: sigma0={self.sigma0.item():.2f}, sigma1={self.sigma1.item():.2f}, sigma2={self.sigma2.item():.4f}, '
                      f'Fc={self.Fc.item():.4f}, Fs={self.Fs.item():.4f}, vs={self.vs.item():.6f}')
            
        return loss_history, data_loss_history, physics_loss_history, param_history
    
    def predict(self, t, v, f):

        self.model.eval()

        dt = t[1] - t[0]

        # normalize velocity with test set's own mean and std
        test_v_mean, test_v_std = np.mean(v), np.std(v)
        v_norm = (v - test_v_mean) / test_v_std
        v_norm_tensor = torch.tensor(v_norm, dtype=torch.float32).view(-1, 1)
        
        # calculate position by integrating velocity
        position = np.zeros_like(v)
        for i in range(1, len(v)):
            position[i] = position[i-1] + v[i-1] * dt
        
        # use position=2 for masking
        # find the first index where position exceeds 2
        mask_indices = np.where(position >= 2.0)[0]
        if len(mask_indices) > 0:
            first_peak_idx = mask_indices[0]
            print(f"Prediction: Position >= 2 found at index {first_peak_idx}, value: {position[first_peak_idx]}")
        else:
            # edge case
            first_peak_idx = len(position) // 4
            print(f"Prediction: Position never reaches 2. Using default index {first_peak_idx}")
        
        # masked position, preserve values up to threshold, then freeze
        masked_position = np.copy(position)
        if first_peak_idx + 1 < len(position):
            masked_position[first_peak_idx + 1:] = position[first_peak_idx + 1]
        
        # normalize position with test set's own mean and std
        test_pos_mean, test_pos_std = np.mean(masked_position), np.std(masked_position)
        pos_norm = (masked_position - test_pos_mean) / test_pos_std
        pos_norm_tensor = torch.tensor(pos_norm, dtype=torch.float32).view(-1, 1)
        
        # calculate direction based on velocity changes
        # find max velocity index
        max_vel_idx = np.argmax(v)
        
        # find min velocity index (after max velocity)
        # captures the second direction change
        min_vel_after_max_idx = max_vel_idx + np.argmin(v[max_vel_idx:])
        
        # three-state direction:
        # 0: initial "go" phase (start to max velocity)
        # 1: first "return" phase (max velocity to min velocity)
        # 2: second "return" phase (min velocity to end)
        direction = np.zeros_like(v)
        direction[max_vel_idx:min_vel_after_max_idx] = 1
        direction[min_vel_after_max_idx:] = 2
        
        # normalize direction to [0,1] range for neural network
        direction = direction / 2.0
        
        direction_tensor = torch.tensor(direction, dtype=torch.float32).view(-1, 1)
        
        # combine for input
        v_dir_pos_tensor = torch.cat((v_norm_tensor, direction_tensor, pos_norm_tensor), dim=1)
        
        # no norm velocity for other calculations
        v_tensor = torch.tensor(v, dtype=torch.float32).view(-1, 1)

        # predict with combined input
        z_pred = 0.00001 * self.model(v_dir_pos_tensor)

        # Print normalization values for debugging
        print(f"Test set normalization - Velocity: mean={test_v_mean:.6f}, std={test_v_std:.6f}")
        print(f"Test set normalization - Position: mean={test_pos_mean:.6f}, std={test_pos_std:.6f}")
        print(f"Train set normalization - Velocity: mean={self.v_mean:.6f}, std={self.v_std:.6f}")
        print(f"Train set normalization - Position: mean={self.pos_mean:.6f}, std={self.pos_std:.6f}")

        dzdt_pred = torch.zeros_like(z_pred)
        
        # for next prediction, we need to prepare the next input
        v_dir_pos_next = v_dir_pos_tensor[1:].clone()
        z_next = 0.00001 * self.model(v_dir_pos_next)
        dzdt_pred[:-1] = (z_next - z_pred[:-1]) / dt
        dzdt_pred[-1] = dzdt_pred[-2]

        # Get bounded parameter values for prediction
        sigma0 = self.sigma0.get_value()
        sigma1 = self.sigma1.get_value()
        sigma2 = self.sigma2.get_value()

        # using original v
        g_v = self.g_function(v_tensor)
        
        # original units
        F_pred = sigma0 * z_pred + sigma1 * dzdt_pred + sigma2 * v_tensor
        
        # convert predictions to numpy arrays
        z_np = z_pred.detach().numpy()
        F_np = F_pred.detach().numpy()
        dzdt_np = dzdt_pred.detach().numpy()
        g_v_np = g_v.detach().numpy()

        return z_np, F_np, dzdt_np, g_v_np, position, masked_position, first_peak_idx
      
def main():
    # train
    time_train = np.load('DATAM/timeM4.npy')
    stage_pos_train = np.load('DATAM/stage_posM4.npy')
    mobile_speed_train = np.load('DATAM/mobile_speedM4.npy')
    
    # test
    time_test = np.load('DATAM/timeM6.npy')
    stage_pos_test = np.load('DATAM/stage_posM6.npy')
    mobile_speed_test = np.load('DATAM/mobile_speedM6.npy')
    
    M = 9.7
    adjust = 0
    window_size = 120
    window_size_pos = 120
    start = 0
    end = 2400
    select_time = False
    Plot_prepare = False
    
    # process training data
    time_train, rel_speed_train, stage_df_train, mobile_df_train = loadDatamatrix(
        stage_pos_train, 
        mobile_speed_train, 
        time_train, 
        adjust, 
        window_size, 
        window_size_pos,
        start, 
        end, 
        Plot_prepare, 
        select_time,
        M
    )
    
    friction_force_train = mobile_df_train['friction_force_cut']
    friction_force_train = friction_force_train / 10.0

    # process test data
    time_test, rel_speed_test, stage_df_test, mobile_df_test = loadDatamatrix(
        stage_pos_test, 
        mobile_speed_test, 
        time_test, 
        adjust, 
        window_size, 
        window_size_pos,
        start, 
        end, 
        Plot_prepare, 
        select_time,
        M
    )
    
    friction_force_test = mobile_df_test['friction_force_cut']
    friction_force_test = friction_force_test / 10.0

    # for m6
    time_test = time_test[:1200]
    rel_speed_test = rel_speed_test[:1200]
    friction_force_test = friction_force_test[:1200]

    # initialize and train the model with training data
    pinn = Friction(time_data=time_train, velocity_data=rel_speed_train, friction_data=friction_force_train)

    # Total parameters in neural network
    print(f"NN parameters: {sum(p.numel() for p in pinn.model.parameters()):,}")
    # Total size in MB
    print(f"Size: {sum(p.numel() * p.element_size() for p in pinn.model.parameters()) / 1024 / 1024:.2f} MB")
    
    total_epochs = 10000
    loss_history, data_loss_history, physics_loss_history, param_history = pinn.train(epochs=total_epochs, lr=0.01)
    
    # get predictions for training data
    z_train, F_train, dzdt_train, g_v_train, position_train, masked_position_train, train_peak_idx = pinn.predict(time_train, rel_speed_train, friction_force_train)
    
    # get predictions for test data
    z_test, F_test, dzdt_test, g_v_test, position_test, masked_position_test, test_peak_idx = pinn.predict(time_test, rel_speed_test, friction_force_test)

    # calculate test error
    test_error = np.mean((F_test - friction_force_test)**2)
    print(f"Test MSE: {test_error:.6f}")

    #calculate training error
    train_error = np.mean((F_train - friction_force_train)**2)
    print(f"Train MSE: {train_error:.6f}")

    # Generate a plot showing parameter evolution during training
    plt.figure(figsize=(15, 10))
    param_names = ['sigma0', 'sigma1', 'sigma2', 'Fc', 'Fs', 'vs']
    epoch_range = range(0, total_epochs, 100)
    
    for i, param_name in enumerate(param_names):
        plt.subplot(3, 2, i+1)
        values = [ph[param_name] for ph in param_history]
        plt.plot(epoch_range, values, 'b-')
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel(param_name, fontsize=15)
        plt.title(f'{param_name} Evolution', fontsize=16)
        plt.tick_params(axis='both', labelsize=15)
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{PATH}/parameter_evolution.png', dpi=300)
    plt.show()

    # Loss history plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history, 'b-', label='Total Loss')
    plt.semilogy(data_loss_history, 'r-', label='Data Loss')
    plt.semilogy(physics_loss_history, 'g-', label='Physics Loss')
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.title('Training Loss History', fontsize=16)
    plt.tick_params(axis='both', labelsize=15)
    plt.legend(fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PATH}/training_loss_history.png', dpi=300)
    plt.show()

    # Training set: Friction vs Velocity
    plt.figure(figsize=(10, 5))
    plt.scatter(rel_speed_train, friction_force_train, color='tab:blue', alpha=0.5, s=10, label='Actual')
    plt.scatter(rel_speed_train, F_train, color='tab:red', alpha=0.5, s=10, label='Predicted')
    plt.xlabel('Relative Velocity (m/s)', fontsize=15)
    plt.ylabel('Friction Force (N)', fontsize=15)
    plt.title('Training Set: Friction Force vs Velocity', fontsize=16)
    plt.tick_params(axis='both', labelsize=15)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=15, markerscale=3)
    plt.tight_layout()
    plt.savefig(f'{PATH}/training_friction_vs_velocity.png', dpi=300)
    plt.show()

    # Testing set: Friction vs Velocity
    plt.figure(figsize=(10, 5))
    plt.scatter(rel_speed_test, friction_force_test, color='tab:blue', alpha=0.5, s=10, label='Actual')
    plt.scatter(rel_speed_test, F_test, color='tab:red', alpha=0.5, s=10, label='Predicted')
    plt.xlabel('Relative Velocity (m/s)', fontsize=15)
    plt.ylabel('Friction Force (N)', fontsize=15)
    plt.title('Testing Set: Friction Force vs Velocity', fontsize=16)
    plt.tick_params(axis='both', labelsize=15)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=15, markerscale=3)
    plt.tight_layout()
    plt.savefig(f'{PATH}/testing_friction_vs_velocity.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
