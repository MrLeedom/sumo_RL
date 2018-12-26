# -*- coding: utf-8 -*-
# PDQN主程序
import traci
import tensorflow as tf
from Priority_RL_brain import DQNPrioritizedReplay
from shixin_auxilliary import get_states, take_action, vehicle_delay
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal

sumoBinary = "D:/code/SUMO/bin/sumo" 
sumoConfig = "D:/code/SUMO/my_code/Dyna/shixin_shanyin/shixin_shanyin.sumocfg"
sumoCmd = [sumoBinary, "-c", sumoConfig]

nX = 9  # states的维度
nU = 1   # 下一相位的时长
actionMap = [-5, -3, 1, 3, 5]
MEMORY_SIZE = 1000

# 原始版store_memory
def store_memory(RL, train_step):
    state = get_states(tls, dets)
    for step in range(train_step):
        action = RL.choose_action(state)
        u = list([actionMap[action]])
        take_action(u, tls)
        delta_t = round((traci.trafficlight.getNextSwitch('center')-traci.simulation.getCurrentTime())/1000)
        for steps in range(delta_t):
            traci.simulationStep()

        next_state = get_states(tls, dets)
        reward = vehicle_delay(lanes)
        RL.store_transition(state, action, reward, next_state)
        state = next_state
        for st in range(3):
            traci.simulationStep()
        step += 1

# 12.5修改版：reward改为当前相位执行之后，过度相位期间(3s)的平均车辆延误
def store_memory_2(RL, train_step):
    state = get_states(tls, dets)
    for step in range(train_step):
        action = RL.choose_action(state)
        u = list([actionMap[action]])
        take_action(u, tls)
        delta_t = round((traci.trafficlight.getNextSwitch('center') - traci.simulation.getCurrentTime()) / 1000)
        for steps in range(delta_t):
            traci.simulationStep()

        next_state = get_states(tls, dets)
        sum_re = 0
        for st in range(3):
            traci.simulationStep()
            reward = vehicle_delay(lanes)
            sum_re += reward
        sum_re = sum_re / 3
        RL.store_transition(state, action, sum_re, next_state)
        state = next_state
        step += 1

def write_result(name, data):
    df = pd.DataFrame({name: data})
    file_name = name + '.csv'
    df.to_csv(file_name, index=True, sep=',')

# 主函数
if __name__ == '__main__':
    sess = tf.Session()
    with tf.variable_scope('DQN_with_prioritized_replay'):
        RL_prio = DQNPrioritizedReplay(
            n_actions=5, n_features=nX, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.00005, sess=sess, prioritized=True,
        )
    sess.run(tf.global_variables_initializer())
    cost = []
    rewards = []
    action = []
    occ = []
    queue = []
    queue_var = []
# #################Training##########################
    traci.start(sumoCmd)
    tls = traci.trafficlight.getIDList()
    lanes = traci.trafficlight.getControlledLanes(''.join(tls))
    dets = traci.lanearea.getIDList()
    store_memory_2(RL_prio, MEMORY_SIZE)

    for episode in range(3):
        step = 0
        while step < 2000:
            RL_prio.learn()
            store_memory_2(RL_prio, 20)
            step += 20
        print("\nEposide %d finished." % episode)
    traci.close()
# #######################Testing############################
    traci.start(sumoCmd)
    tls = traci.trafficlight.getIDList()
    lanes = traci.trafficlight.getControlledLanes(''.join(tls))
    dets = traci.lanearea.getIDList()

    for episode in range(1):
        step = 0
        while step < 2000:
            RL_prio.learn()
            store_memory_2(RL_prio, 20)
            step += 20
        t_cost, t_rewards, t_actions, t_occ, t_queue, t_queue_var = RL_prio.return_results()

        cost.extend(t_cost)
        rewards.extend(t_rewards)
        action.extend(t_actions)
        occ.extend(t_occ)
        queue.extend(t_queue)
        queue_var.extend(t_queue_var)
    traci.close()

    occ = signal.medfilt(occ, 5)
    rewards = signal.medfilt(rewards, 5)
    queue = signal.medfilt(queue, 5)
    queue_var = signal.medfilt(queue_var, 5)

    df = pd.DataFrame({'rewards': rewards,
                       'occ': occ,
                       'queue_length': queue,
                       'queue_var': queue_var})
    df.to_csv('result.csv', index=True, sep=',')

    plt.plot(cost, label='DQN with prioritized replay')
    plt.legend(loc='best')
    plt.ylabel('Training times')
    plt.xlabel('training steps')
    plt.grid()
    plt.show()

    plt.plot(rewards, c='r', label='REWARDS')
    plt.plot(occ, c='b', label='OCC')
    plt.legend(loc='best')
    plt.ylabel('Rewards/OCC')
    plt.xlabel('transitions')
    plt.grid()
    plt.show()

    plt.plot(rewards, c='r', label='REWARDS')
    plt.plot(queue, c="green", label='QUEUE_LENGTH')
    plt.legend(loc='best')
    plt.ylabel('Rewards/Queue length')
    plt.xlabel('transitions')
    plt.grid()
    plt.show()





