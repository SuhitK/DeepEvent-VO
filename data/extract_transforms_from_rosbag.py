import numpy as np

import rospy
from rosbag import Bag

from cv_bridge import CvBridge
import cv2

import argparse
import tf
from geometry_msgs.msg import Pose, Point, Quaternion

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_zlim(-100, 100)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# plt.ion()

_EPS = np.finfo(float).eps * 4.0
print (_EPS)

def quaternion_matrix(quaternion):

    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> np.allclose(M, np.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> np.allclose(M, np.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


def pose_to_matrix(pose_msg):

    """ converts ros pose_stamped message to pose homogenous matrix"""

    translation = np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
    quaternion = np.array([pose_msg.orientation.w, pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z])

    rotation = quaternion_matrix(quaternion)
    rotation[:3,3] = translation

    pose_matrix = rotation

    return pose_matrix

def transform_to_se3(transform):

    """ converts transformation matrix to se(3) tangent space"""

    rotation_matrix = transform[:3,:3]
    translation = transform[:3,3]

    trace = rotation_matrix[0,0] + rotation_matrix[1,1] + rotation_matrix[2,2]
    theta = np.arccos((trace - 1)/2.0)

    omega_cross = (theta/(2.0*np.sin(theta)))*(rotation_matrix - rotation_matrix.T)
    omega = np.array([omega_cross[2,1], omega_cross[0,2], omega_cross[1,0]])

    return np.hstack((omega, translation))


def main():
    parser = argparse.ArgumentParser(
        description=("Extracts pose data from frames and converts to transforms"))
        
    parser.add_argument("--bag", dest="bag",
                        help="Path to ROS bag.",
                        required=True)
    parser.add_argument("--prefix", dest="prefix",
                        help="Output file prefix.",
                        required=True)
    parser.add_argument("--output_folder", dest="output_folder",
                        help="Output folder.",
                        required=True)
    parser.add_argument("--start_time", dest="start_time",
                        help="Time to start in the bag.",
                        type=float,
                        default = 0.0)
    parser.add_argument("--end_time", dest="end_time",
                        help="Time to end in the bag.",
                        type=float,
                        default = -1.0)

    args = parser.parse_args()
    bag_file = args.bag

    pose_array = np.zeros((12,))
    se3_array = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    point_array = np.array([[0.0, 0.0, 0.0, 1.0]])

    with Bag(bag_file, 'r') as ib:

        t_start = ib.get_start_time()
        
        print ("..Finished Reading Bag..")
        
        pose_prev = []
        iteration = 0
        new_point = np.array([[0.0, 0.0, 0.0, 1.0]]).T
        
        for topic, msg, t in ib.read_messages(start_time=rospy.Time(args.start_time + t_start)):

            if('pose' in topic):    
                
                if(iteration == 0):
                    pose_prev = pose_to_matrix(msg.pose)
                    R = pose_prev[:3,:3].reshape(9,)
                    t = pose_prev[:3,3].reshape(3,)

                    global_pose_data = np.hstack((R,t))
                    pose_array = np.vstack((pose_array, global_pose_data))

                    
                else:
                    pose_current = pose_to_matrix(msg.pose)
                    # for 2
                    # transformation = np.matmul(np.linalg.inv(pose_prev), pose_current)
                    
                    # for 1
                    transformation = np.matmul(pose_current, np.linalg.pinv(pose_prev))

                    R = pose_current[:3,:3].reshape(9,)
                    t = pose_current[:3,3].reshape(3,)
                    global_pose_data = np.hstack((R,t))
                    pose_array = np.vstack((pose_array, global_pose_data))

                    se3 = transform_to_se3(transformation)
                    new_point =  np.matmul(transformation, new_point)
                    pose_prev = pose_current

                    se3 = np.expand_dims(se3,0)

                    se3_array = np.vstack((se3_array, se3))
                    point_array = np.vstack((point_array, new_point.T))
                    
                print (iteration)
                iteration+=1     

            # fig.canvas.draw_idle()
            # fig.canvas.flush_events()
    
    ax.plot(point_array[:,0], point_array[:,1], point_array[:,2])            
    np.save("se3_transforms.npy", se3_array[1:,:])
    np.save("global_poses.npy", pose_array[1:,:])



    plt.show()
    

if __name__ == "__main__":
    main()
