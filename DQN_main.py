# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:21:29 2018

@author: Administrator

DQN主程序
"""

import traci
from RL_brain import DeepQNetwork
from shixin_auxilliary import get_states,take_action,vehicle_delay

sumoBinary = "D:/code/SUMO/bin/sumo" 
sumoConfig = "D:/code/SUMO/my_code/Dyna/shixin_shanyin/shixin_shanyin.sumocfg"
sumoCmd=[sumoBinary, "-c", sumoConfig]

nX=9  #states的维度
nU=1   #下一相位的时长
actionMap=[15,25,35,45,55]

#*********************Main**********************
if __name__=='__main__':
    #get basic equipment lists
    traci.start(sumoCmd)
    tls = traci.trafficlight.getIDList()
    lanes=traci.trafficlight.getControlledLanes(''.join(tls))
    dets=traci.lanearea.getIDList()
        
    state_space_size = nX
    action_space_size = 5    
    RL = DeepQNetwork(action_space_size, state_space_size,
                     learning_rate=0.01,
                     reward_decay=0.9,
                     e_greedy=0.9,
                     replace_target_iter=50,
                     memory_size=200,
                     output_graph=True
                     )
    total_reward = []
    delays = []
    
    for episode in range(1):
        state = get_states(tls,dets)
        steps = 0
        while steps < 1000:
            action = RL.choose_action(state)
            u = list([actionMap[action]])
            delay1 = vehicle_delay(lanes)
            take_action(u,tls)
            delta_t=round((traci.trafficlight.getNextSwitch('center')-traci.simulation.getCurrentTime())/1000)
            for step in range(delta_t):
                traci.simulationStep()
            
            next_state = get_states(tls,dets)
            delay2 = vehicle_delay(lanes)
            reward = delay1-delay2
            
            total_reward.append(reward)
            delays.append(delay2)
            RL.store_transition(state, action, reward, next_state)
            
            if (steps > 200) and (steps % 5 == 0):
                RL.learn()
            state = next_state
            
            for step in range(3):
                traci.simulationStep() 
            steps += 1
    RL.plot_cost()
