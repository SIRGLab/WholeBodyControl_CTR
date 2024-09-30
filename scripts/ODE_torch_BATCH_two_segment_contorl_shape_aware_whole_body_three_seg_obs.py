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
    def __init__(self):
        super(NeuralODEController, self).__init__()
        number_of_segment = 3
        down_sample = 10
        taget_pos_size = 3
        self.net = nn.Sequential(
            # nn.Linear(10*3 + 3, 256),  # 10*3 state variables + 3 target positions
            # nn.Linear(3, 256),  # 10*3 state variables + 3 target positions
            nn.Linear(number_of_segment*3+ 3*down_sample+taget_pos_size, 256),  # 3*2 current u, 10*current state,  state variables + 3 target positions
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
        # u = self.net(x)/20.0
        # return u
        
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
    
    # controller.load_state_dict(torch.load("/home/mohammad/SoftRobot_CORL23/trainedModel/model_controller_20240906-180632.zip"))
    # controller.load_state_dict(torch.load("/home/mohammad/SoftRobot_CORL23/trainedModel/model_controller_temp_8000_20240906-211303.zip"))
    controller.load_state_dict(torch.load("trainedModel/model_controller_temp_12000_20240907-104024.zip"))
    
            
    fig = plt.figure( dpi=300)    
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_title('Visualizer-1.01')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_xlim([-0.08, 0.08])
    ax.set_ylim([-0.08, 0.08])
    ax.set_zlim([-0.0, 0.15])
    plt.rc('font', family='serif', size=10)
    plt.rc('axes', titlesize=12, labelsize=10)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('legend', fontsize=10)
    plt.rc('lines', linewidth=1.5, markersize=4)
    plt.rc('grid', color='#E5E5E5', linestyle='-', linewidth=0.7)
    
    # ax.view_init(elev=28, azim=125)  # High elevation, rotated 30 degrees
    # ax.view_init(elev=65, azim=125)  # High elevation, rotated 30 degrees
    # ax.view_init(elev=26, azim=-146)  # High elevation, rotated 30 degrees
    # ax.view_init(elev=30, azim=-66)  # High elevation, rotated 30 degrees
    # ax.view_init(elev=32, azim=92)  # High elevation, rotated 30 degrees
    # ax.view_init(elev=34, azim=109)  # High elevation, rotated 30 degrees
    ax.view_init(elev=30, azim=129)  # High elevation, rotated 30 degrees
    
    plt.pause(0.001)
    time.sleep(5)

    batch_size = 1
    # reset_actions = (0*torch.randint(-150,150,[batch_size,9])/10000.0).to(device)  # Random targets for tip position
    # reset_actions = (0*torch.randint(-150,150,[batch_size,9])/10000.0).to(device)  # Random targets for tip position
    reset_actions = 1*torch.Tensor([[-0.0,0.0,-0.0,
                                  0.0,0.0,0.0,
                                  0.0,0.0,0.0]]).to(device)  # Random targets for tip position
    reset_states  = sf.odeStepFull(reset_actions)
    sample = 10
    reset_states_down_sample  = downsample_simple(reset_states,sample)[:,:,:3].reshape(batch_size,3*sample)   
    
    t = torch.linspace(0,100,100)
    # target_tip_position = [0.025*torch.sin(2*torch.pi*t/100.0),
    #                                         0.08*torch.cos(2*torch.pi*t/100.0),
    #                                         0.125+0.0*t/50.0 ]
    
    # target_tip_position = [0.025*torch.sin(2*torch.pi*t/100.0),
    #                                         0.04*torch.cos(2*torch.pi*t/200.0),
    #                                         0.105+0.0*t/50.0]
    
    target_tip_position = [-0.02 + 0.02*torch.sin(2*torch.pi*t/100.0),
                                        -0.01 + 0.02*torch.cos(1*torch.pi*t/100.0),
                                        0.1-0.0*t/50.0 +0.0*torch.sin(2*torch.pi*t/100.0)]
        
    ax.plot(target_tip_position[0][:], target_tip_position[1][:], target_tip_position[2][:],'g--')
    
    
    # target_tip_position = []
    # for t in range(100):
    #     T  = 25
    #     tt = (t*1.0)
    #     scale = 3
    #     x0 = np.array([-0.0,-0.0,0.13])
        
    #     if (tt<T):
    #         xd = (x0 + scale*np.array((-0.01+(0.02/T)*tt,0.01,0.0)))
    #     elif (tt<2*T):
    #         xd = (x0 + scale*np.array((0.01,0.01-((0.02/T)*(tt-T)),0.0)))
    #     elif (tt<3*T):
    #         xd = (x0 + scale*np.array((0.01-((0.02/T)*(tt-(2*T))),-0.01,0.0)))
    #     elif (tt<4*T):
    #         xd = (x0 + scale*np.array((-0.01,-0.01+((0.02/T)*(tt-(3*T))),0.0)))
    #     else:
    #         # t0 = time.time()+5
    #         gt = 0
            
    #     target_tip_position.append(xd)
    # target_tip_position = np.array(target_tip_position)    

    # ax.plot(target_tip_position[:,0], target_tip_position[:,1], target_tip_position[:,2],'g--')

    
    for t in range(100):
        # target_tip_position = torch.tensor([0.025*torch.sin(torch.tensor([2*torch.pi*t/100.0])),
        #                                     0.08*torch.cos(torch.tensor([2*torch.pi*t/100.0])),
        #                                     0.125+0.0*t/50.0 +0.0*torch.sin(torch.tensor([2*torch.pi*t/100.0]))]).unsqueeze(0).to(device)
        
        # target_tip_position = torch.tensor([0.025*torch.sin(torch.tensor([2*torch.pi*t/100.0])),
        #                                     0.04*torch.cos(torch.tensor([2*torch.pi*t/200.0])),
        #                                     0.105+0.0*t/50.0 +0.0*torch.sin(torch.tensor([2*torch.pi*t/100.0]))]).unsqueeze(0).to(device)
        
        target_tip_position = torch.tensor([-0.02 + 0.02*torch.sin(torch.tensor([2*torch.pi*t/100.0])),
                                            -0.01 + 0.02*torch.cos(torch.tensor([1*torch.pi*t/100.0])),
                                            0.1+0.0*t/50.0 +0.0*torch.sin(torch.tensor([2*torch.pi*t/100.0]))]).unsqueeze(0).to(device)
        
        # T  = 25
        # tt = (t*1.0)
        # scale = 3
        # x0 = np.array([-0.0,-0.0,0.13])

        # if (tt<T):
        #     xd = (x0 + scale*np.array((-0.01+(0.02/T)*tt,0.01,0.0)))
        # elif (tt<2*T):
        #     xd = (x0 + scale*np.array((0.01,0.01-((0.02/T)*(tt-T)),0.0)))
        # elif (tt<3*T):
        #     xd = (x0 + scale*np.array((0.01-((0.02/T)*(tt-(2*T))),-0.01,0.0)))
        # elif (tt<4*T):
        #     xd = (x0 + scale*np.array((-0.01,-0.01+((0.02/T)*(tt-(3*T))),0.0)))
        # else:
        #     # t0 = time.time()+5
        #     gt = 0
            
        # target_tip_position = torch.tensor(xd,dtype = torch.float32,device = device).unsqueeze(0)
       
        obs = [0.0,0.0,0.1] 

        # Forward pass through neural network to get actions
        actions = controller(reset_actions,reset_states_down_sample, target_tip_position)
    
        states_batch = sf.odeStepFull(actions)
        reset_states = states_batch
        reset_actions = actions
        reset_states_down_sample  = downsample_simple(reset_states,sample)[:,:,:3].reshape(batch_size,3*sample)   
        
        if t % 1 == 0 and t>0:   
            for t in range(states_batch.size(1)):
                states = states_batch[:, t, :].detach().cpu().numpy()
                targ = target_tip_position.detach().cpu().numpy()
                
                l1 = int((sf.l0+actions[0,0])/sf.ds)+1
                l2 = l1+int((sf.l0+actions[0,3])/sf.ds)-1
                
                s1 = ax.plot(states[:l1, 0], states[:l1, 1], states[:l1, 2],linewidth = 1.8,color=(0.0, 0.0, 0., 0.8))
                s2 = ax.plot(states[l1:l2, 0], states[l1:l2, 1], states[l1:l2, 2],linewidth = 1.6,color=(0.6, 0.2, 0.2, 0.8))
                s3 = ax.plot(states[l2-1:, 0], states[l2-1:, 1], states[l2-1:, 2],linewidth = 1.2,color=(0.0, 0.0, 0.5, 0.8))
                
                s4 = ax.plot(targ[0,0], targ[0,1], targ[0,2],'r-o')
                ax.plot(obs[0], obs[1], obs[2],'m-o')
                
                plt.pause(0.005)
                
                  
                l = s1.pop(0)
                l.remove()
                l = s2.pop(0)
                l.remove()
                l = s3.pop(0)
                l.remove()
                l = s4.pop(0)
                l.remove()
            
    plt.show()
    exit()

    
    # Example: Training Loop
    batch_size = 500
    num_epochs = 20000
    optimizer  = torch.optim.Adam(controller.parameters(), lr=0.001)
    loss_fn    = nn.MSELoss()

    for epoch in range(num_epochs):
        # reset
        # reset_actions = torch.tensor([0,0,0,0,0.0,0.0]).repeat(batch_size).reshape([batch_size,6])
        reset_actions = (torch.randint(-150,150,[batch_size,9])/10000.0).to(device)  # Random targets for tip position
        reset_states  = sf.odeStepFull(reset_actions)
        down_sample = 10
        reset_states_down_sample  = downsample_simple(reset_states,down_sample)[:,:,:3].reshape(batch_size,3*down_sample)   

        
        # reset_states  = downsample_simple(reset_states,10)[:,:,:3].reshape(batch_size,3*10)   
        target_tip_position = reset_states[-1,:,:3] + (torch.randint(-500,500,[batch_size,3])/100000.0).to(device)  # Random targets for tip position
        
        # Forward pass through neural network to get actions
        # actions = controller(reset_actions, reset_states[-1,:,:3], target_tip_position)
        actions = controller(reset_actions, reset_states_down_sample, target_tip_position)
    
    
        # with torch.no_grad():    
        states_batch = sf.odeStepFull(actions)
        down_sample = 10
        
        states_down_sample  = downsample_simple(states_batch,down_sample)[:,:,:3]   
        
        obs = torch.Tensor([0.0,0.0,0.1]).to(device) 
        s = states_down_sample[:,:,:3]
        dist = torch.linalg.norm(s-obs,dim=2)
        
        obs_loss = torch.mean(torch.lt(dist,0.01)*10.0)
        
        states_down_sample = states_down_sample.reshape(batch_size,3*down_sample)
       
        # Calculate loss using only the tip position (final state position)
        final_tip_position = states_batch[-1, :, :3]  # (batch_size, 3)
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
