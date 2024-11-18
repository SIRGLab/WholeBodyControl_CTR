import torch
import torch.nn as nn
import torch.nn.init as init

from torchdiffeq import odeint  # Ensure you have torchdiffeq installed
import matplotlib.pyplot as plt
import numpy as np
import time

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
from trainner import ODE, NeuralODEController, downsample_simple


def get_target_ref(traj='s', t=0):
    if traj == 's':
        target_tip_position = [-0.02 + 0.02*torch.sin(2*torch.pi*t/100.0),
                                            -0.01 + 0.02*torch.cos(1*torch.pi*t/100.0),
                                            0.1-0.0*t/50.0 +0.0*torch.sin(2*torch.pi*t/100.0)]  
    elif traj == 'elipse':
        target_tip_position = [0.025*torch.sin(2*torch.pi*t/100.0),
                                            0.08*torch.cos(2*torch.pi*t/100.0),
                                            0.125+0.0*t/50.0 ]
    
    elif traj == 'circle':
        target_tip_position = [0.025*torch.sin(2*torch.pi*t/100.0),
                                            0.025*torch.cos(2*torch.pi*t/100.0),
                                            0.105+0.0*t/50.0]
    elif traj == 'square':
        target_tip_position = []
        x0 = np.array([-0.02,0.02,0.13])
        scale = 3
        T  = 25
          
        if len(t.shape) >0:
            for t in range(100):
                tt = (t*1.0)
                if (tt<T):
                    xd = (x0 + scale*np.array((-0.01+(0.02/T)*tt,0.01,0.0)))
                elif (tt<2*T):
                    xd = (x0 + scale*np.array((0.01,0.01-((0.02/T)*(tt-T)),0.0)))
                elif (tt<3*T):
                    xd = (x0 + scale*np.array((0.01-((0.02/T)*(tt-(2*T))),-0.01,0.0)))
                elif (tt<4*T):
                    xd = (x0 + scale*np.array((-0.01,-0.01+((0.02/T)*(tt-(3*T))),0.0)))

                target_tip_position.append(xd)                
            target_tip_position = np.array(target_tip_position) 
        else:
            tt = (t*1.0)
            if (tt<T):
                xd = (x0 + scale*np.array((-0.01+(0.02/T)*tt,0.01,0.0)))
            elif (tt<2*T):
                xd = (x0 + scale*np.array((0.01,0.01-((0.02/T)*(tt-T)),0.0)))
            elif (tt<3*T):
                xd = (x0 + scale*np.array((0.01-((0.02/T)*(tt-(2*T))),-0.01,0.0)))
            elif (tt<4*T):
                xd = (x0 + scale*np.array((-0.01,-0.01+((0.02/T)*(tt-(3*T))),0.0)))
            target_tip_position = np.array(xd,dtype=np.float32)
        
    return target_tip_position

if __name__ == "__main__":
    
    # Initialize ODE and Neural ODE Controller
    sf = ODE().to(device)
    down_sample = 10
    traj_name = 's'
    # traj_name = 'elipse'
    # traj_name = 'circle'
    # traj_name = 'square'
    
    obs = [0.0,0.0,0.1] 

    controller = NeuralODEController().to(device)
    controller.load_state_dict(torch.load("trainedModel/model_controller.zip"))
            
    fig = plt.figure( dpi=300)    
    ax = fig.add_subplot(111, projection='3d')
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
    
    ax.view_init(elev=30, azim=129)  # High elevation, rotated 30 degrees
    
    plt.pause(0.001)
    time.sleep(5)

    batch_size = 1
    reset_actions = torch.Tensor([[-0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0]]).to(device)  # Random targets for tip position
    reset_states  = sf.odeStepFull(reset_actions)
    
    reset_states_down_sample  = downsample_simple(reset_states,down_sample)[:,:,:3].reshape(batch_size,3*down_sample)   
    
    t = torch.linspace(0,100,100)
   
    target_tip_position = get_target_ref(traj_name,t)    
    if traj_name == 'square':
        ax.plot(target_tip_position[:,0], target_tip_position[:,1], target_tip_position[:,2],'g--')
    else:
        ax.plot(target_tip_position[:][0], target_tip_position[:][1], target_tip_position[:][2],'g--')
    
    for t in range(100):
        
        target_tip_position = torch.tensor(get_target_ref(traj_name,torch.tensor(t))).to(device).unsqueeze(0)
    
        # Forward pass through neural network to get actions
        actions = controller(reset_actions,reset_states_down_sample, target_tip_position)
    
        states_batch = sf.odeStepFull(actions)
        reset_states = states_batch
        reset_actions = actions
        reset_states_down_sample  = downsample_simple(reset_states,down_sample)[:,:,:3].reshape(batch_size,3*down_sample)   
        
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

    