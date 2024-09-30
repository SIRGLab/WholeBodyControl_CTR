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
        self.number_of_segment = 1
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
        number_of_segment = self.number_of_segment
        for n in range(number_of_segment):
            
            # Determine the maximum length in the batch to ensure consistent integration steps
            max_length = torch.max(l).detach().cpu().numpy()
            t_eval = torch.arange(0.0, max_length + self.ds, self.ds).to(device)
        
            # Solve ODE for all batch elements simultaneously
            sol_batch = odeint(self.odeFunction, y0_batch, t_eval)  # (timesteps, batch_size, 14)

            # Mask out solutions for each trajectory after their respective lengths
            lengths = (l / self.ds).long()
            
            sol_masked = sol_batch.to(device).clone()  # (timesteps, batch_size, 14)
        
            for i in range(batch_size):
                sol_masked[lengths[i]:, i ] = sol_batch[lengths[i], i]  # Masking with last one after trajectory ends
        
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
    def __init__(self):
        super(NeuralODEController, self).__init__()
 
        self.net = nn.Sequential(
            nn.Linear(3+2, 64),  # 3*2 current u, 10*current state,  state variables + 3 target positions
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 3+2),  # Outputs 3 action variables
            nn.Tanh()
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, t, y):
        
        # batch_size = y.shape[0]
        # dydt = torch.zeros((batch_size, 14)).to(device)
        # # tanh_output = self.net(torch.cat([torch.tensor([t]).to(device),y[:, 12:]]))
        
        # e3 = torch.tensor([0.0, 0.0, 1.0],device=device).reshape(1, 3, 1).repeat(batch_size, 1, 1)
        
        # tanh_output = self.net(y[:, 12:])
        # ux = y[:, 12] + tanh_output[:,0]  # batch_size
        # uy = y[:, 13] + tanh_output[:,1] # batch_size
    
        # # Compute u_hat for each batch element
        # u_hat = torch.zeros((batch_size, 3, 3),device=device)
        # u_hat[:, 0, 2] = uy
        # u_hat[:, 1, 2] = -ux
        # u_hat[:, 2, 0] = -uy
        # u_hat[:, 2, 1] = ux

        # r = y[:, 0:3].reshape(batch_size, 3, 1)
        # R = y[:, 3:12].reshape(batch_size, 3, 3)
        
        # dR = torch.matmul(R, u_hat)  # batch_size x 3 x 3
        # dr = torch.matmul(R, e3).squeeze(-1)  # batch_size x 3

        # # Reshape and assign to dydt
        # dydt[:, 0:3] = dr
        # dydt[:, 3:12] = dR.reshape(batch_size, 9)
        # return dydt
    
        # Original output from network, range [-1, 1] due to Tanh
        tanh_output = self.net(y)
        # tanh_output[:,3] *= 0
        # tanh_output[:,4] *= 0
        
        return tanh_output



def downsample_simple(arr, m):
    n = len(arr)
    indices = np.linspace(0, n - 1, m, dtype=int)  # Linearly spaced indices
    return arr[indices]

if __name__ == "__main__":
    
    # Initialize ODE and Neural ODE Controller
    sf = ODE().to(device)
    shape_Node = NeuralODEController().to(device)
    
    
    # shape_Node.load_state_dict(torch.load("/home/mohammad/SoftRobot_CORL23/trainedModel/shape_node_temp_4000_20240907-161131.zip"))
    # shape_Node.load_state_dict(torch.load("/home/mohammad/SoftRobot_CORL23/trainedModel/shape_node_temp_1000_20240907-204244.zip"))
    # shape_Node.load_state_dict(torch.load("/home/mohammad/SoftRobot_CORL23/trainedModel/shape_node_temp_6000_20240908-073023.zip"))
    # shape_Node.load_state_dict(torch.load("/home/mohammad/SoftRobot_CORL23/trainedModel/shape_node_temp_1000_20240908-091928.zip"))
    # shape_Node.load_state_dict(torch.load("/home/mohammad/SoftRobot_CORL23/trainedModel/shape_node_temp_2000_20240908-094027.zip"))
    # shape_Node.load_state_dict(torch.load("trainedModel/shape_node_temp_9000_20240911-102604.zip"))
    shape_Node.load_state_dict(torch.load("trainedModel/shape_node_20240911-130255.zip"))
    
    
    
    
        
    fig = plt.figure( dpi=300)    
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_title('Visualizer-1.01')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_xlim([-0.08, 0.08])
    ax.set_ylim([-0.08, 0.08])
    ax.set_zlim([-0.0, 0.1])
    plt.rc('font', family='serif', size=10)
    plt.rc('axes', titlesize=12, labelsize=10)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('legend', fontsize=10)
    plt.rc('lines', linewidth=1.5, markersize=4)
    plt.rc('grid', color='#E5E5E5', linestyle='-', linewidth=0.7)
    
    ax.view_init(elev=28, azim=125)  # High elevation, rotated 30 degrees

    batch_size = 100
    n_seg = 1
    sf.number_of_segment = n_seg
    loss_fn    = nn.MSELoss()

    for t in range(1):
        
        # Forward pass through neural network to get actions
        actions = (torch.randint(-50,50,[batch_size,3*n_seg])/10000.0).to(device)  # Random targets for tip position
        # sf.ds = np.random.randint(2,100)/10000.0
        sf.ds = 0.005
        states_batch = sf.odeStepFull(actions)
        
        
         
        # actions[:,2] = 0.05 
        y0_batch = torch.zeros ([batch_size,5]).to(device)  # (batch_size, 14)
        l, ux, uy = sf.updateAction(actions[:,:3])
        y0_batch[:, 3] = ux.reshape(1,batch_size) #((ux.reshape(1,batch_size) + (torch.randint(-1500,1500,[1,batch_size])/100.0).to(device)))
        y0_batch[:, 4] = uy.reshape(1,batch_size) #((uy.reshape(1,batch_size) + (torch.randint(-1500,1500,[1,batch_size])/100.0).to(device)))
        
        
        # y0_batch[:, 12] = ux #((ux.reshape(1,batch_size) + (torch.randint(-1500,1500,[1,batch_size])/1000.0).to(device)))
        # y0_batch[:, 13] = uy #((uy.reshape(1,batch_size) + (torch.randint(-1500,1500,[1,batch_size])/1000.0).to(device)))
        # y0_batch[:, 12] = ((ux.reshape(1,batch_size) + (torch.randint(-1500,1500,[1,batch_size])/1000.0).to(device)))
        # y0_batch[:, 13] = ((uy.reshape(1,batch_size) + (torch.randint(-1500,1500,[1,batch_size])/1000.0).to(device)))
            
        sol = None    
        for n in range(n_seg):
            
            max_length = torch.max(l).detach().cpu().numpy()
            t_eval = torch.arange(0.0, max_length + sf.ds, sf.ds).to(device)
        
            sol_batch = odeint(shape_Node, y0_batch, t_eval)  # (timesteps, batch_size, 14)
            # sol_batch = odeint(sf.odeFunction, y0_batch, t_eval)  # (timesteps, batch_size, 14)
            
            # Mask out solutions for each trajectory after their respective lengths
            lengths = (l / sf.ds).long()
            
            sol_masked = sol_batch.to(device).clone()  # (timesteps, batch_size, 14)
        
            for i in range(batch_size):
                sol_masked[lengths[i]:, i ] = sol_batch[lengths[i], i]  # Masking with last one after trajectory ends
                
            if sol is None:
                sol = sol_masked
            else:                
                sol = torch.cat((sol, sol_masked), dim=0)
                    
            y0_batch = sol_masked[-1]  # (batch_size, 14)
            
            
            if n < n_seg-1:
                l, ux, uy = sf.updateAction(actions[:, (n+1)*3:(n+2)*3])
                # max_length = torch.max(l).detach().cpu().numpy()
                # t_eval = torch.arange(0.0, max_length + sf.ds, sf.ds).to(device)           
                y0_batch[:, 3] = ux #((ux.reshape(1,batch_size) + (torch.randint(-1500,1500,[1,batch_size])/1000.0).to(device)))
                y0_batch[:, 4] = uy #((uy.reshape(1,batch_size) + (torch.randint(-1500,1500,[1,batch_size])/1000.0).to(device)))
            
        
        states = states_batch[:, t, :].detach().cpu().numpy()
        states_p = sol[:, t, :].detach().cpu().numpy()
        # ax.plot(states[:, 0], states[:, 1], states[:, 2],linewidth = 1.8,color=(0.0, 0.0, 0., 0.8))
        # ax.plot(states_p[:, 0], states_p[:, 1], states_p[:, 2],linewidth = 1.8,color=(0.6, 0.2, 0.2, 0.3))
        
        print (f"num seg: {n_seg}, batch size: {batch_size}")
        print (f"loss-RMSE x: {torch.sqrt(loss_fn(states_batch[:,:,0], sol[:,:,0])):3.4f}")
        print (f"loss-STD  x: {torch.std(states_batch[:,:,0] - sol[:,:,0]):3.4f}")
        print (f"loss-VAR  x: {torch.var(states_batch[:,:,0] - sol[:,:,0]):3.4f}")
        print("----------------------")
        
        print (f"loss-RMSE y: {torch.sqrt(loss_fn(states_batch[:,:,1], sol[:,:,1])):3.4f}")
        print (f"loss-STD  y: {torch.std(states_batch[:,:,1] - sol[:,:,1]):3.4f}")
        print (f"loss-VAR  y: {torch.var(states_batch[:,:,1] - sol[:,:,1]):3.4f}")
        print("----------------------")
        
        print (f"loss-RMSE z: {torch.sqrt(loss_fn(states_batch[:,:,2], sol[:,:,2])):3.4f}")
        print (f"loss-STD  z: {torch.std(states_batch[:,:,2] - sol[:,:,2]):3.4f}")
        print (f"loss-VAR  z: {torch.var(states_batch[:,:,2] - sol[:,:,2]):3.4f}")
        
        for t in range(states_batch.size(1)):
            states = states_batch[:, t, :].detach().cpu().numpy()
            states_p = sol[:, t, :].detach().cpu().numpy()
            # ax.plot(states[:, 0], states[:, 1], states[:, 2],linewidth = 1.8,color=(0.0, 0.0, 0., 0.8))
            # ax.plot(states_p[:, 0], states_p[:, 1], states_p[:, 2],linewidth = 1.8,color=(0.6, 0.2, 0.2, 0.3))
            
            # print (f"num seg: {n_seg}, batch size: {batch_size}")
            # print (f"loss-RMSE x: {torch.sqrt(loss_fn(states_batch[:,:,0], sol[:,:,0]))}")
            # print (f"loss-STD  x: {torch.std(states_batch[:,:,0] - sol[:,:,0])}")
            # print (f"loss-VAR  x: {torch.var(states_batch[:,:,0] - sol[:,:,0])}")
            # print("----------------------")
            
            # print (f"loss-RMSE y: {torch.sqrt(loss_fn(states_batch[:,:,1], sol[:,:,1]))}")
            # print (f"loss-STD  y: {torch.std(states_batch[:,:,1] - sol[:,:,1])}")
            # print (f"loss-VAR  y: {torch.var(states_batch[:,:,1] - sol[:,:,1])}")
            # print("----------------------")
            
            # print (f"loss-RMSE z: {torch.sqrt(loss_fn(states_batch[:,:,2], sol[:,:,2]))}")
            # print (f"loss-STD  z: {torch.std(states_batch[:,:,2] - sol[:,:,2])}")
            # print (f"loss-VAR  z: {torch.var(states_batch[:,:,2] - sol[:,:,2])}")
            # print("----------------------")
            
            l1 = int((sf.l0+actions[0,0])/sf.ds)+1
            # l2 = l1+int((sf.l0+actions[0,3])/sf.ds)-1
            # l3 = l2+int((sf.l0+actions[0,6])/sf.ds)-1
            # l4 = l3+int((sf.l0+actions[0,9])/sf.ds)-1
            
            
            ax.plot(states[:l1, 0], states[:l1, 1], states[:l1, 2],linewidth = 0.5,color=(0.0, 0.0, 0., 0.8))
            # ax.plot(states[l1:l2, 0], states[l1:l2, 1], states[l1:l2, 2],linewidth = 0.5,color=(0.6, 0.2, 0.2, 0.8))
            # ax.plot(states[l2-1:l3, 0], states[l2-1:l3, 1], states[l2-1:l3, 2],linewidth = 0.5,color=(0.0, 0.0, 0.5, 0.8))
            # ax.plot(states[l3-1:l4, 0], states[l3-1:l4, 1], states[l3-1:l4, 2],linewidth = 0.5,color=(0.0, 0.5, 0.5, 0.8))
            
            ax.plot(states_p[:l1, 0], states_p[:l1, 1], states_p[:l1, 2],linewidth = 1.8,color=(0., 0., 0., 0.3))
            # ax.plot(states_p[l1:l2, 0], states_p[l1:l2, 1], states_p[l1:l2, 2],linewidth = 1.6,color=(0.6, 0.2, 0.2, 0.3))
            # ax.plot(states_p[l2-1:l3, 0], states_p[l2-1:l3, 1], states_p[l2-1:l3, 2],linewidth = 1.4,color=(0.0, 0.0, 0.5, 0.3))
            # ax.plot(states_p[l3-1:l4, 0], states_p[l3-1:l4, 1], states_p[l3-1:l4, 2],linewidth = 1.4,color=(0.0, 0.5, 0.5, 0.3))
            
            
            plt.pause(0.005)
        
    plt.show()
    exit()

    
    # Example: Training Loop
    batch_size = 500
    num_epochs = 10000
    optimizer  = torch.optim.Adam(shape_Node.parameters(), lr=0.001)
    loss_fn    = nn.MSELoss()
     
    # r0 = torch.zeros(3, 1).to(device)
    # R0 = torch.eye(3).reshape(9, 1).to(device)
    # y0 = torch.cat((r0, R0, torch.zeros([2, 1],device=device)), dim=0)
    # y0 = y0.squeeze()


    for epoch in range(num_epochs):
        # Forward pass through neural network to get actions
        actions = (torch.randint(-80,80,[batch_size,3])/10000.0).to(device)  # Random targets for tip position
        
        sf.ds = np.random.randint(1,100)/10000.0
    
        states_batch = sf.odeStepFull(actions)[:,:,:3]
        
        
        
        l, ux, uy = sf.updateAction(actions)
        max_length = torch.max(l).detach().cpu().numpy()
        # Solve ODE for all batch elements simultaneously
        t_eval = torch.arange(0.0, max_length + sf.ds, sf.ds).to(device)
        
         
        y0_batch = torch.zeros ([batch_size,5]).to(device)  # (batch_size, 14)
        y0_batch[:, 3] = ((ux.reshape(1,batch_size) + (torch.randint(-1500,1500,[1,batch_size])/100.0).to(device)))
        y0_batch[:, 4] = ((uy.reshape(1,batch_size) + (torch.randint(-1500,1500,[1,batch_size])/100.0).to(device)))
        
        sol_batch = odeint(shape_Node, y0_batch[:,:5], t_eval)[:,:,:3]  # (timesteps, batch_size, 14)
        
        # Mask out solutions for each trajectory after their respective lengths
        lengths = (l / sf.ds).long()
        sol_masked = sol_batch.clone()  # (timesteps, batch_size, 14)
    
        for i in range(batch_size):
            last = sol_masked[lengths[i], i].clone()
            sol_masked[lengths[i]:, i ] = last   # Masking with last one after trajectory ends

        
        loss = 1000*loss_fn(sol_masked[:,:,:2], states_batch[:,:,:2]) + 2000*loss_fn(sol_masked[:,:,2], states_batch[:,:,2])
               
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():3.8f}')
            
        if epoch >0 and epoch % 1000 == 0:
            timestr   = time.strftime("%Y%m%d-%H%M%S")
            modelName = f"trainedModel/shape_node_temp_{epoch}_"+ timestr+".zip"
            torch.save(shape_Node.state_dict(), modelName)
            print (f"model: {modelName} has been saved.")
                


    timestr   = time.strftime("%Y%m%d-%H%M%S")
    modelName = "trainedModel/shape_node_"+ timestr+".zip"
    torch.save(shape_Node.state_dict(), modelName)
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
