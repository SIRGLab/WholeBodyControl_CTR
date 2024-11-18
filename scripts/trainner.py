import torch
import torch.nn as nn
import torch.nn.init as init

from torchdiffeq import odeint  # Ensure you have torchdiffeq installed
import matplotlib.pyplot as plt
import numpy as np
import time

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

class ODE(nn.Module):
    def __init__(self) -> None:
        super(ODE, self).__init__()
        self.l0 = 50e-3  # initial length of robot
        self.d = 7.5e-3  # cables offset
        self.ds = 0.005  # ode step time
        
        r0 = torch.zeros(3, 1).to(device)
        R0 = torch.eye(3).reshape(9, 1).to(device)
        y0 = torch.cat((r0, R0, torch.zeros([2, 1],device=device)), dim=0)
        
        self.y0 = y0.squeeze()

    def updateAction(self, actions):
        # Assuming actions is of shape (batch_size, 3)
        l = self.l0 + actions[:, 0]  # batch_size
        ux = actions[:, 2] / -(l * self.d)  # batch_size
        uy = actions[:, 1] / (l * self.d)  # batch_size
        return l, ux, uy

    def odeFunction(self, s, y):
        batch_size = y.shape[0]
        dydt = torch.zeros((batch_size, 14)).to(device)
        
        e3 = torch.tensor([0.0, 0.0, 1.0],device=device).reshape(1, 3, 1).repeat(batch_size, 1, 1)
        ux = y[:, 12]  # batch_size
        uy = y[:, 13]  # batch_size
        
        # Compute u_hat for each batch element
        u_hat = torch.zeros((batch_size, 3, 3),device=device)
        u_hat[:, 0, 2] = uy
        u_hat[:, 1, 2] = -ux
        u_hat[:, 2, 0] = -uy
        u_hat[:, 2, 1] = ux

        r = y[:, 0:3].reshape(batch_size, 3, 1)
        R = y[:, 3:12].reshape(batch_size, 3, 3)
        
        dR = torch.matmul(R, u_hat)  # batch_size x 3 x 3
        dr = torch.matmul(R, e3).squeeze(-1)  # batch_size x 3

        # Reshape and assign to dydt
        dydt[:, 0:3] = dr
        dydt[:, 3:12] = dR.reshape(batch_size, 9)
        return dydt

    def odeStepFull(self, actions):
        batch_size = actions.size(0)
        
        # Create a batch of initial conditions
        y0_batch = self.y0.unsqueeze(0).repeat(batch_size, 1).to(device)  # (batch_size, 14)
        l, ux, uy = self.updateAction(actions)
        y0_batch[:, 12] = ux
        y0_batch[:, 13] = uy
        
        sol = None
        number_of_segment = 3  
        for n in range(number_of_segment):
            
            # Determine the maximum length in the batch to ensure consistent integration steps
            max_length = torch.max(l).detach().cpu().numpy()
            t_eval = torch.arange(0.0, max_length + self.ds, self.ds).to(device)
        
            # Solve ODE for all batch elements simultaneously
            sol_batch = odeint(self.odeFunction, y0_batch, t_eval)  # (timesteps, batch_size, 14)

            # Mask out solutions for each trajectory after their respective lengths
            lengths = (l / self.ds).long()
            
            sol_masked = sol_batch.to(device)  # (timesteps, batch_size, 14)
        
            for i in range(batch_size):
                sol_masked[lengths[i]:, i ] = sol_masked[lengths[i], i]  # Masking with last one after trajectory ends
        
            if sol is None:
                sol = sol_masked
            else:                
                sol = torch.cat((sol, sol_masked), dim=0)
                    
            y0_batch = sol_masked[-1]  # (batch_size, 14)
            if n < number_of_segment-1:
                l, ux, uy = self.updateAction(actions[:, (n+1)*3:(n+2)*3])
                y0_batch[:, 12] = ux
                y0_batch[:, 13] = uy
                
        return sol  # (timesteps, batch_size, 14)

class NeuralODEController(nn.Module):
    def __init__(self,number_of_segment=3, down_sample=10,taget_pos_size=3):
        super(NeuralODEController, self).__init__()
        self._number_of_segment = number_of_segment
        self._down_sample = down_sample
        self._taget_pos_size = taget_pos_size
        self.net = nn.Sequential(
            nn.Linear(self._number_of_segment*3 + 3*self._down_sample + self._taget_pos_size, 256),  # 3*2 current u, 10*current state,  state variables + 3 target positions
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 9),  # Outputs 3 action variables
            nn.Tanh()
        )
        
        # Define desired output ranges for actions
        self.min_action_values = torch.tensor([-0.03, -0.015, -0.015, -0.03, -0.015, -0.015, -0.03, -0.015, -0.015],device=device)  # replace with your desired mins
        self.max_action_values = torch.tensor([0.03,   0.015, 0.015, 0.03, 0.015, 0.015, 0.03, 0.015, 0.015],device=device)        # replace with your desired maxs
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, action, state, target):
        # Concatenate state and target
        x = torch.cat([action, state, target], dim=1)
        
        # Original output from network, range [-1, 1] due to Tanh
        tanh_output = self.net(x)
        
        # Calculate the center and range for scaling
        action_range = (self.max_action_values - self.min_action_values) / 2
        action_center = (self.max_action_values + self.min_action_values) / 2
        
        # Apply scaling to convert [-1, 1] range to the desired range [min_action_values, max_action_values]
        scaled_output = action_center + action_range * tanh_output
        
        return scaled_output



def downsample_simple(arr, m):
    n = len(arr)
    indices = np.linspace(0, n - 1, m, dtype=int)  # Linearly spaced indices
    return arr[indices]

if __name__ == "__main__":
    
    # Initialize ODE and Neural ODE Controller
    sf = ODE().to(device)
    controller = NeuralODEController().to(device)
    
    batch_size = 500
    num_epochs = 20000
    down_sample = 10
    obs = torch.Tensor([0.0,0.0,0.1]).to(device) 
        
  
    optimizer  = torch.optim.Adam(controller.parameters(), lr=0.001)
    loss_fn    = nn.MSELoss()

    for epoch in range(num_epochs):
        reset_actions = (torch.randint(-150,150,[batch_size,9])/10000.0).to(device)  # Random targets for tip position
        reset_states  = sf.odeStepFull(reset_actions)
        reset_states_down_sample  = downsample_simple(reset_states,down_sample)[:,:,:3].reshape(batch_size,3*down_sample)   

        
        target_tip_position = reset_states[-1,:,:3] + (torch.randint(-500,500,[batch_size,3])/100000.0).to(device)  # Random targets for tip position
        
        # Forward pass through neural network to get actions
        actions = controller(reset_actions, reset_states_down_sample, target_tip_position)
    
    
        states_batch = sf.odeStepFull(actions)
        
        states_down_sample  = downsample_simple(states_batch,down_sample)[:,:,:3]   
        
        s = states_down_sample[:,:,:3]
        dist = torch.linalg.norm(s-obs,dim=2)
        
        obs_loss = torch.mean(torch.lt(dist,0.01)*10.0)
        
        states_down_sample = states_down_sample.reshape(batch_size,3*down_sample)
       
        # Calculate loss using only the tip position (final state position)
        final_tip_position = states_batch[-1, :, :3] 
        loss = 5000*loss_fn(final_tip_position, target_tip_position) +\
                100*loss_fn(reset_actions, actions) +\
                200*loss_fn(states_down_sample, reset_states_down_sample)+\
                obs_loss    

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():3.8f}')
            
        if epoch >0 and epoch % 1000 == 0:
            timestr   = time.strftime("%Y%m%d-%H%M%S")
            modelName = f"trainedModel/model_controller_temp_{epoch}_"+ timestr+".zip"
            torch.save(controller.state_dict(), modelName)
            print (f"model: {modelName} has been saved.")
                


    timestr   = time.strftime("%Y%m%d-%H%M%S")
    modelName = "trainedModel/model_controller_"+ timestr+".zip"
    torch.save(controller.state_dict(), modelName)
    print (f"model: {modelName} has been saved.")
    
    # Plotting the results after training
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Visualizer-1.01')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_xlim([-0.08, 0.08])
    ax.set_ylim([-0.08, 0.08])
    ax.set_zlim([-0.0, 0.15])
    
    for t in range(states_batch.size(1)):
        states = states_batch[:, t, :].detach().cpu().numpy()
        ax.plot(states[:, 0], states[:, 1], states[:, 2])

    plt.show()
