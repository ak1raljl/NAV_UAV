import gym
import airsim
import cv2
import numpy as np
import torch
import math
import random
from gym import spaces


class AirsimEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        np.set_printoptions(formatter={'float': '{: 4.2f}'.format}, suppress=True)
        torch.set_printoptions(profile="short", sci_mode=False, linewidth=1000)
        print("init airsim_gym_env")
        self.model = None
        self.data_path = None
        self.navigation_3d = True
        self.using_velocity_state = False
        self.dt = 0.2

        self.start_position = [0, 0, 1]
        self.goal_position = [0, 0, 0]
        self.start_random_angle = 0
        self.goal_distance = 75
        self.goal_random_angle = 0
        self.goals_dis = [70, 60, 50, 40, 30, 20, 10, 0]
        self.level = 0

        self.work_space_x = [self.start_position[0] - 1, self.start_position[0] + 75]
        self.work_space_y = [self.start_position[1] - 3.5, self.start_position[1] + 3.5]
        self.work_space_z = [0, 7]
        self.max_episode_steps = 300
        self._max_episode_steps = 300

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.start_random_angle = 1
        self.goal_random_angle = None

        self.x = 0
        self.y = 0
        self.z = 0
        self.v_xy = 0
        self.v_z = 0
        self.yaw = 0
        self.yaw_rate = 0
        self.yaw_accumulated = 0.0  # 累计偏航角（弧度），用于超转检测

        self.v_xy_sp = 0
        self.v_z_sp = 0
        self.yaw_rate_sp = 0

        self.crash_distance = 0.1
        self.accept_radius = 1
        self.max_depth_meters = 8
        self.screen_height = 60
        self.screen_width = 90

        self.acc_xy_max = 2.0
        self.v_xy_max = 5.0
        self.v_xy_min = 0.2
        self.v_z_max = 1.0
        self.yaw_rate_max_deg = 30.0
        self.yaw_rate_max_rad = math.radians(self.yaw_rate_max_deg)
        self.max_vertical_difference = 5

        if self.navigation_3d:
            if self.using_velocity_state:
                self.state_feature_length = 6
            else:
                self.state_feature_length = 3
            self.action_space = spaces.Box(low=np.array([self.v_xy_min, -self.v_z_max, -self.yaw_rate_max_rad]),
                                           high=np.array([self.v_xy_max, self.v_z_max, self.yaw_rate_max_rad]),
                                           dtype=np.float32)
        else:
            if self.using_velocity_state:
                self.state_feature_length = 4
            else:
                self.state_feature_length = 2
            self.action_space = spaces.Box(low=np.array([self.v_xy_min, -self.yaw_rate_max_rad]),
                                           high=np.array([self.v_xy_max, self.yaw_rate_max_rad]),
                                           dtype=np.float32)

        self.observation_space = spaces.Dict({
            "depth": spaces.Box(low=0, high=255,
                                shape=(self.screen_height, self.screen_width, 1),
                                dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0,
                                shape=(self.state_feature_length,),
                                dtype=np.float32)
        })

        self.episode_num = 0
        self.total_step = 0
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.previous_distance_from_des_point = 0
        self.min_distance_to_obstacles = 0
        self.state_current_attitude = 0

    def reset(self):
        self.client.reset()
        self.update_goal_pose()
        yaw_noise = self.start_random_angle * np.random.random()
        pose = self.client.simGetVehiclePose()
        pose.position.x_val = self.start_position[0]
        pose.position.y_val = self.start_position[1]
        pose.position.z_val = -self.start_position[2]
        pose.orientation = airsim.to_quaternion(0, 0, 0)
        self.client.simSetVehiclePose(pose, True)

        self.client.simPause(False)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.client.moveToZAsync(-self.start_position[2], 2).join()

        self.client.simPause(True)

        self.episode_num += 1
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.goal_distance = self.get_distance()
        self.previous_distance_from_des_point = self.goal_distance
        self.level = 0
        self.yaw_accumulated = 0.0

        obs = self.get_obs()

        return obs

    def step(self, action):
        distance = self.get_distance()
        position_ue4 = self.get_position()
        self.set_action(action)
        obs = self.get_obs()

        is_not_inside_workspace_now = self.is_not_inside_workspace()
        has_reached_des_pose = self.is_in_desired_pose()
        too_close_to_obstable = self.is_crashed()

        is_timeout = self.step_num >= self.max_episode_steps
        is_over_rotated = abs(self.yaw_accumulated) > (math.pi / 2.0)
        done = (is_not_inside_workspace_now or has_reached_des_pose
                or too_close_to_obstable or is_timeout or is_over_rotated)
        reward = self.compute_reward_final(done, action)
        if is_not_inside_workspace_now:
            reward = -500
        if has_reached_des_pose:
            reward = 2000
        if too_close_to_obstable:
            reward = -500
        if is_timeout:
            reward = -500
        if is_over_rotated:
            reward = -500
        self.cumulated_episode_reward += reward
        info = {
            'is_success': has_reached_des_pose,
            'is_crash': too_close_to_obstable,
            'is_not_in_workspace': is_not_inside_workspace_now,
            'is_timeout': is_timeout,
            'is_over_rotated': is_over_rotated,
            'step_num': self.step_num,
            'reward': self.cumulated_episode_reward,
            'level': self.level
        }
        if done:
            print(info)
        self.print_train_info_airsim(action, obs, reward, position_ue4, distance)
        self.step_num += 1
        self.total_step += 1

        return obs, reward, done, info

    def render(self, mode='drone'):
        pass

    def close(self):
        pass

    def set_start(self, position, random_angle):
        self.start_position = position
        self.start_random_angle = random_angle

    def set_goal(self, distance, random_angle):
        self.goal_distance = distance
        self.goal_random_angle = random_angle

    def set_action(self, action):
        self.v_xy_sp = action[0] * 0.7
        self.yaw_rate_sp = action[-1] * 2
        if self.navigation_3d:
            self.v_z_sp = float(action[1])
        else:
            self.v_z_sp = 0
        prev_yaw = self.yaw
        self.yaw = self.get_attitude()[2]
        delta_yaw = self.yaw - prev_yaw
        # wrap delta to [-π, π] 避免跨越 ±180° 时跳变
        delta_yaw = (delta_yaw + math.pi) % (2 * math.pi) - math.pi
        self.yaw_accumulated += delta_yaw
        yaw_sp = self.yaw + self.yaw_rate_sp * self.dt

        if yaw_sp > math.radians(180):
            yaw_sp -= math.pi * 2
        elif yaw_sp < math.radians(-180):
            yaw_sp += math.pi * 2

        vx_local_sp = self.v_xy_sp * math.cos(yaw_sp)
        vy_local_sp = self.v_xy_sp * math.sin(yaw_sp)

        self.client.simPause(False)
        if len(action) == 2:
            self.client.moveByVelocityZAsync(
                vx_local_sp,
                vy_local_sp,
                -self.start_position[2],
                self.dt,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(self.yaw_rate_sp))
            ).join()
        elif len(action) == 3:
            self.client.moveByVelocityAsync(
                vx_local_sp,
                vy_local_sp,
                -self.v_z_sp,
                self.dt,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(self.yaw_rate_sp))
            ).join()

        self.client.simPause(True)

    def update_goal_pose(self):
        distance = self.goal_distance
        self.goal_position[0] = distance + self.start_position[0]
        self.goal_position[1] = self.start_position[1]
        self.goal_position[2] = self.start_position[2]

    def get_position(self):
        position = self.client.simGetVehiclePose().position
        return [position.x_val, position.y_val, -position.z_val]

    def get_distance(self):
        return self.goal_distance - self.get_position()[0]

    def get_attitude(self):
        self.state_current_attitude = self.client.simGetVehiclePose().orientation
        return airsim.to_eularian_angles(self.state_current_attitude)

    def get_velocity(self):
        states = self.client.getMultirotorState()
        linear_velocity = states.kinematics_estimated.linear_velocity
        angular_velocity = states.kinematics_estimated.angular_velocity

        velocity_xy = math.sqrt(pow(linear_velocity.x_val, 2) + pow(linear_velocity.y_val, 2))
        velocity_z = linear_velocity.z_val
        yaw_rate = angular_velocity.z_val

        return [velocity_xy, -velocity_z, yaw_rate]

    def get_obs(self):
        image = self.get_depth_image()
        image_resize = cv2.resize(image, (self.screen_width, self.screen_height))
        self.min_distance_to_obstacles = image.min()
        image_scaled = np.clip(image_resize, 0, self.max_depth_meters) / self.max_depth_meters * 255
        image_scaled = 255 - image_scaled
        depth = image_scaled.astype(np.uint8)[..., np.newaxis]  # (H, W, 1)

        state = self._get_state_feature()  # float32, [-1, 1]
        return {"depth": depth, "state": state}

    def get_depth_image(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
        ])

        # check observation
        while responses[0].width == 0:
            print("get_image_fail...")
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
            ])

        depth_img = airsim.list_to_2d_float_array(
            responses[0].image_data_float,
            responses[0].width,
            responses[0].height
        )

        depth_meter = depth_img * 100

        return depth_meter

    def _get_state_feature(self):
        distance = self.get_distance()
        relative_yaw = self._get_relative_yaw()
        relative_pose_z = self.get_position()[2] - self.goal_position[2]

        # normalize to [-1, 1]
        distance_norm     = np.clip(distance / self.goal_distance * 2 - 1, -1, 1)
        vertical_norm     = np.clip(relative_pose_z / self.max_vertical_difference, -1, 1)
        relative_yaw_norm = np.clip(relative_yaw / math.pi, -1, 1)

        velocity = self.get_velocity()
        linear_velocity_xy, linear_velocity_z, yaw_rate = velocity
        velocity_xy_norm  = np.clip((linear_velocity_xy - self.v_xy_min) / (self.v_xy_max - self.v_xy_min) * 2 - 1, -1, 1)
        velocity_z_norm   = np.clip(linear_velocity_z / self.v_z_max, -1, 1)
        yaw_rate_norm     = np.clip(yaw_rate / self.yaw_rate_max_rad, -1, 1)

        self.state_raw = np.array([distance, relative_pose_z, math.degrees(relative_yaw),
                                   linear_velocity_xy, linear_velocity_z, math.degrees(yaw_rate)])
        state_full = np.array([distance_norm, vertical_norm, relative_yaw_norm,
                               velocity_xy_norm, velocity_z_norm, yaw_rate_norm], dtype=np.float32)

        if self.navigation_3d:
            state = state_full[:6] if self.using_velocity_state else state_full[:3]
        else:
            state = state_full[[0, 2, 3, 5]] if self.using_velocity_state else state_full[[0, 2]]

        return state

    def _get_relative_yaw(self):
        # description: get relative yaw from current pose to goal in radian
        current_position = self.get_position()
        # get relative angle
        relative_pose_x = self.goal_position[0] - current_position[0]
        relative_pose_y = self.goal_position[1] - current_position[1]
        angle = math.atan2(relative_pose_y, relative_pose_x)

        # get current yaw
        yaw_current = self.get_attitude()[2]

        # get yaw error
        yaw_error = angle - yaw_current
        if yaw_error > math.pi:
            yaw_error -= 2*math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2*math.pi

        return yaw_error

    def is_not_inside_workspace(self):
        is_not_inside = False
        current_position = self.get_position()
        if current_position[0] < self.work_space_x[0] or current_position[0] > self.work_space_x[1] or \
                current_position[1] < self.work_space_y[0] or current_position[1] > self.work_space_y[1] or \
                current_position[2] < self.work_space_z[0] or current_position[2] > self.work_space_z[1]:
            is_not_inside = True

        return is_not_inside

    def is_in_desired_pose(self):
        in_desired_pose = False
        if self.get_distance() < self.accept_radius:
            in_desired_pose = True

        return in_desired_pose

    def is_crashed(self):
        is_crashed = False
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided or self.min_distance_to_obstacles < self.crash_distance:
            is_crashed = True

        return is_crashed

    def _process_reward(self):
        """过程奖励：距离进度 + 课程里程碑"""
        distance_now = self.get_distance()
        distance_delta = self.previous_distance_from_des_point - distance_now
        self.previous_distance_from_des_point = distance_now

        # 距离进度：靠近目标为正，远离为负
        r_distance = 300.0 * distance_delta / self.goal_distance

        # 课程里程碑：飞过检查点给递增奖励
        r_milestone = 0.0
        dis_remaining = self.get_position()[0]
        dis_remaining = self.goal_distance - dis_remaining
        if self.level < len(self.goals_dis) and dis_remaining < self.goals_dis[self.level]:
            self.level += 1
            r_milestone = 1.2 * (1 + self.level / len(self.goals_dis))

        return r_distance + r_milestone

    def _regularization_reward(self, action):
        """正则化奖励：抑制不必要的动作和姿态偏差"""
        # 偏航角误差（机头偏离目标方向）
        yaw_error_deg = self.state_raw[2]
        r_yaw_align = -0.8 * abs(yaw_error_deg / 90.0)

        # 偏航角速率（抑制原地打转）
        r_yaw_rate = -0.3 * abs(action[-1]) / self.yaw_rate_max_rad

        # 累计偏转惩罚（越接近 ±180° 惩罚越大）
        r_spin = -0.5 * np.clip(abs(self.yaw_accumulated) / math.pi / 2.0, 0, 1)

        r_vz = 0.0
        r_z_err = 0.0
        if self.navigation_3d:
            r_vz   = -0.1 * (abs(action[1]) / self.v_z_max) ** 2
            r_z_err = -0.2 * (abs(self.state_raw[1]) / self.max_vertical_difference) ** 2

        return r_yaw_align + r_yaw_rate + r_spin + r_vz + r_z_err

    def _safety_reward(self):
        """安全奖励：障碍物距离惩罚"""
        if self.min_distance_to_obstacles >= 3.0:
            return 0.0
        # 越近惩罚越大，< crash_distance 时为 -1
        proximity = 1.0 - np.clip(
            (self.min_distance_to_obstacles - self.crash_distance) / 5.0, 0, 1)
        return -0.5 * proximity

    def compute_reward_final(self, done, action):
        if done:
            return 0
        r_process = self._process_reward()
        r_safety  = self._safety_reward()
        r_reg     = self._regularization_reward(action)
        return r_process + r_safety + r_reg

    def print_train_info_airsim(self, action, obs, reward, pose, distance):
        msg_train_info = "EP: {} Step: {} Total_step: {}".format(self.episode_num, self.step_num, self.total_step)
        self.client.simPrintLogMessage('Train: ', msg_train_info)
        self.client.simPrintLogMessage('Action: ', str(action))
        self.client.simPrintLogMessage('reward: ', "{:4.4f} total: {:4.4f}".format(
            reward, self.cumulated_episode_reward))
        self.client.simPrintLogMessage('Min_depth: ', str(self.min_distance_to_obstacles))
        # self.client.simPrintLogMessage('position: ', str(pose))
        self.client.simPrintLogMessage('distance: ', str(distance))
        self.client.simPrintLogMessage('level: ', str(self.level))

