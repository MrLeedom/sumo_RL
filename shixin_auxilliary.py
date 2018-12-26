# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 18:35:16 2018

@author: Administrator
"""

import numpy as np
import traci
import pandas as pd


sumoBinary = "D:/code/SUMO/bin/sumo" 
sumoConfig = "D:/code/SUMO/my_code/Dyna/shixin_shanyin/shixin_shanyin.sumocfg"
sumoCmd=[sumoBinary, "-c", sumoConfig]

# Input:detectorIDs
# Output:各流向的占有率和排队长度(以DataFrame的形式)
def get_states(tls,detectorIDs):
    states = []
    # phase index
    for tl in tls:
        ph=int(traci.trafficlight.getPhase(tl))
        states.append(ph/6) # 控制相位的序号：0,2,4,6

    # occ & queue length
    lane_direction = pd.DataFrame(pd.read_excel('lane_direction.xlsx'))
    dir = lane_direction.iloc[:,1]
    lanes=[]
    for det in detectorIDs:
        lane = traci.lanearea.getLaneID(det)
        lanes.append(lane)
    # 获取每个车道的参数
    occ = []
    queue_length = []
    for det in detectorIDs:
        occ_temp = traci.lanearea.getLastStepOccupancy(det)/100
        occ.append(occ_temp)
        queue = traci.lanearea.getJamLengthVehicle(det)/28
        queue_length.append(queue)
   
    # 合并相同流向的车道
    occ_df = pd.DataFrame({'direction':dir,'occupancy':occ})
    occ_df = occ_df.groupby(['direction']).mean()
    states.extend(occ_df.iloc[:, 0])
    queue_df = pd.DataFrame({'direction': dir, 'queue_length': queue_length})
    queue_df = queue_df.groupby(['direction']).sum()
    states.extend(queue_df.iloc[:, 0])
    states = np.array(states)
    return states

# Input:下一相位的时长，信号灯名称列表
# function:根据参数改变下一相位的时长
# 注意：应在过度相位的最后一秒执行该函数，函数执行完毕后，信号进入下一控制相位;
# 此时getPhaseDuration()的返回值仍是原始值，但是该相位的持续时间确实改变了，可由getNextSwitch(tlID)的数值验证
def take_action(u, tlIDs):
    raw_tsc = pd.DataFrame(pd.read_excel('raw_tsc.xlsx'))
    new_duration = []
    for ext, light in zip(u, tlIDs):
        index = traci.trafficlight.getPhase(light)
        #获取下一相位的序号和原始时长
        next_index = (index+1) % 8
        raw_duration = raw_tsc.iloc[next_index, 1]
        new_duration.append(raw_duration+ext)
    #进入下一相位
    traci.simulationStep()
    #改变当前相位的时长
    for delta, light in zip(new_duration, tlIDs):
        traci.trafficlight.setPhaseDuration(light, delta-1)


# 某一时刻的车辆平均延误
def vehicle_delay(laneIDs):
    average_delay=[]
    for l in laneIDs:
        lane_delay = 0
        vehicleIDs=traci.lane.getLastStepVehicleIDs(l)
        vehicleNum=traci.lane.getLastStepVehicleNumber(l)
        if vehicleNum !=0:
            for v in vehicleIDs:
                v_delay = 0
                allow_speed=traci.vehicle.getAllowedSpeed(v)
                real_speed=traci.vehicle.getSpeed(v)
                v_delay = 1-real_speed/allow_speed
                lane_delay += v_delay
            average_delay.append(lane_delay/vehicleNum)
    result=np.array(average_delay)
    return result.mean()
