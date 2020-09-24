'''
Description: the preprocessing module for the data and filter
Version: 2.0
Autor: mario
Date: 2020-09-24 16:34:31
LastEditors: mario
LastEditTime: 2020-09-24 20:58:13
'''


def MotionJointSelect(motiondata, datamode, featuremode):
    '''
    description: remove the no data joint in the data
    param {type} 
    return {type} 
    author: mario
    '''
    
    # the body mode motiondata has 18 joints of the whole body
    if datamode == 'body':
        jointindex = [i for i in range(18)]
        # only the upper body, is needed
        if featuremode == 0:
            # remove the lower-limbs joints
            lower_limbs_joints = [8, 9, 10, 11, 12, 13]
            for i in lower_limbs_joints:
                jointindex.remove(i)
        elif featuremode == 1:
            # only upper limbs joints is provided
            jointindex = jointindex[:8]
    
        return motiondata[:, jointindex, :]