 
import torch
import torch.nn as nn
from torchdiffeq import odeint  # Ensure you have torchdiffeq installed
import matplotlib.pyplot as plt

class ODE(nn.Module):
    def __init__(self) -> None:
        super(ODE, self).__init__()
        self.l0 = 50e-3  # initial length of robot
        self.d = 7.5e-3  # cables offset
        self.ds = 0.005  # ode step time
        
        r0 = torch.zeros(3, 1)
        R0 = torch.eye(3).reshape(9, 1)
        y0 = torch.cat((r0, R0, torch.zeros([2, 1])), dim=0)
        
        self.y0 = y0.squeeze()

    def updateAction(self, actions):
        # Assuming actions is of shape (batch_size, 3)
        l = self.l0 + actions[:, 0]  # batch_size
        ux = actions[:, 2] / -(l * self.d)  # batch_size
        uy = actions[:, 1] / (l * self.d)  # batch_size
        return l, ux, uy

    def odeFunction(self, s, y):
        batch_size = y.shape[0]
        dydt = torch.zeros((batch_size, 14))
        
        e3 = torch.tensor([0.0, 0.0, 1.0]).reshape(1, 3, 1).repeat(batch_size, 1, 1)
        ux = y[:, 12]  # batch_size
        uy = y[:, 13]  # batch_size
        
        # Compute u_hat for each batch element
        u_hat = torch.zeros((batch_size, 3, 3))
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
        y0_batch = self.y0.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, 14)
        l, ux, uy = self.updateAction(actions)
        y0_batch[:, 12] = ux
        y0_batch[:, 13] = uy
        
        sol = None
        number_of_segment = 2  
        for n in range(number_of_segment):
            
            # Determine the maximum length in the batch to ensure consistent integration steps
            max_length = torch.max(l)
            t_eval = torch.arange(0, max_length + self.ds, self.ds)
        
            # Solve ODE for all batch elements simultaneously
            sol_batch = odeint( self.odeFunction, y0_batch, t_eval)  # (timesteps, batch_size, 14)

            # Mask out solutions for each trajectory after their respective lengths
            lengths = (l / self.ds).long()
            
            
            sol_masked = sol_batch  # (batch_size, timesteps, 14)
        
            for i in range(batch_size):
                sol_masked[lengths[i]:, i ] = sol_masked[lengths[i],i]  # Masking with lastone after trajectory ends
        
            # sol_masked = sol_batch.permute(1, 0, 2)  # (batch_size, timesteps, 14)
            
            # for i in range(batch_size):
            #     sol_masked[i, lengths[i]:] = sol_masked[i, lengths[i]] #float('nan')  # Masking with NaNs after trajectory ends
                
            # sol_masked.permute(1, 0, 2)  # (timesteps, batch_size, 14)
            if sol is None:
                sol = sol_masked
            else:                
                sol = torch.cat((sol,sol_masked),dim=0)
                    
            y0_batch = sol_masked[-1]  # (batch_size, 14)
            l, ux, uy = self.updateAction(actions[:,n*3:(n+1)*3])
            y0_batch[:, 12] = ux
            y0_batch[:, 13] = uy
       
            
            

        return sol  # (timesteps, batch_size, 14)

if __name__ == "__main__":
    
    sf = ODE()
    
    batch_size = 3
    actions = torch.randint(-100,100,[batch_size,9])/10000.0
    states_batch = sf.odeStepFull(actions)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Visualizer-1.01')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_xlim([-0.08, 0.08])
    ax.set_ylim([-0.08, 0.08])
    ax.set_zlim([-0.0, 0.15])
    
    for i in range(states_batch.size(1)):
        states = states_batch[:, i, :]
        ax.plot(states[:, 0], states[:, 1], states[:, 2])

    plt.show()
    