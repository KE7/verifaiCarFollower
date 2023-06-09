import math
from cv2 import imshow
import einops
import os
import sys
from typing import List, Optional
import numpy as np

from controller import Keyboard, Camera, Lidar, GPS, Display, Robot
from vehicle import Driver
# from deepbots.supervisor.controllers.deepbots_supervisor_env import RobotSupervisorEnv
from deepbots.supervisor.controllers.deepbots_supervisor_env import DeepbotsSupervisorEnv
from PPOAgent import PPOAgent, Transition
from gym.spaces import Box, Discrete

# Size of the yellow line angle filter
FILTER_SIZE = 3

# Line following PID
KP = 0.25
KI = 0.006
KD = 2

UNKNOWN = 99999.99
TIME_STEP = 50  # milliseconds

class VehicleController:
    def __init__(self, driver: Driver, robot: Robot):
        self.driver = driver
        self.robot = robot
        self.autodrive = True
        self.speed = 0.0
        self.steering_angle = 0.0
        self.max_speed = 250.0
        self.min_speed = -50.0
        self.manual_steering = 0
        self.PID_need_reset = False
        self.filter_angle_first_call = True
        self.previous_filtered_angles = [0.0] * FILTER_SIZE
        self.obstacle_distance = 0.0

        # Enable various features
        self.enable_collision_avoidance = False
        self.enable_display = False
        self.has_gps = False
        self.has_camera = False

        # Camera
        self.camera: Optional[Camera] = None
        self.camera_width = -1
        self.camera_height = -1
        self.camera_fov = -1.0

        # SICK laser
        self.sick: Optional[Lidar] = None
        self.sick_width = -1
        self.sick_range = -1.0
        self.sick_fov = -1.0

        # Speedometer
        self.display: Optional[Display] = None
        self.display_width = 0
        self.display_height = 0
        self.speedometer_image: Optional[Display] = None

        # GPS
        self.gps: Optional[GPS] = None
        self.gps_coords = [0.0, 0.0, 0.0]
        self.X = 0
        self.Y = 1
        self.Z = 2
        self.gps_speed = 0.0

        self.setup()


    def setup(self):
        self.detect_features()
        self.init_camera()
        self.init_sick()
        self.init_gps()
        self.init_display()
        self.start_engine()
        self.print_help()

    def detect_features(self):
        num_devices = self.robot.getNumberOfDevices()
        for i in range(num_devices):
            device = self.robot.getDeviceByIndex(i)
            name = device.getName()
            if name == "Sick LMS 291":
                self.enable_collision_avoidance = True
            elif name == "display":
                self.enable_display = True
            elif name == "gps":
                self.has_gps = True
            elif name == "camera":
                self.has_camera = True

    def init_camera(self):
        if self.has_camera:
            self.camera = self.robot.getDevice("camera")
            self.camera.enable(TIME_STEP)
            self.camera.recognitionEnable(TIME_STEP)
            self.camera_width = self.camera.getWidth()
            self.camera_height = self.camera.getHeight()
            self.camera_fov = self.camera.getFov()

    def init_sick(self):
        if self.enable_collision_avoidance:
            self.sick = self.robot.getDevice("Sick LMS 291")
            self.sick.enable(TIME_STEP)
            self.sick_width = self.sick.getHorizontalResolution()
            self.sick_range = self.sick.getMaxRange()
            self.sick_fov = self.sick.getFov()

    def init_gps(self):
        if self.has_gps:
            self.gps = self.robot.getDevice("gps")
            self.gps.enable(TIME_STEP)

    def init_display(self):
        if self.enable_display:
            self.display = self.robot.getDevice("display")
            self.speedometer_image = self.display.imageLoad("speedometer.png")

    def start_engine(self):
        if self.has_camera:
            self.set_speed(40.0)  # km/h

        self.driver.setHazardFlashers(True)
        self.driver.setDippedBeams(True)
        self.driver.setAntifogLights(True)
        self.driver.setWiperMode(Driver.SLOW)

    def print_help(self):
        print("Use the arrow keys to control the vehicle")
        print("[Left]/[Right] to steer")
        print("[Up]/[Down] to accelerate/brake")
        print("Press 'A' to switch back to auto-drive")

    def set_speed(self, speed: float):
        # check for max speed
        if speed > self.max_speed:
            print("Max speed reached. Clipping to {} km/h".format(self.max_speed))
            speed = self.max_speed

        # check for min speed
        if speed < self.min_speed:
            print("Min speed reached. Clipping to {} km/h".format(self.min_speed))
            speed = self.min_speed

        self.speed = speed

        self.driver.setCruisingSpeed(cruisingSpeed=self.speed)

    def set_auto_drive(self, autodrive: bool):
        if self.autodrive == autodrive:
            return
        
        self.autodrive = autodrive
        if autodrive:
            if self.has_camera:
                print("Autodrive enabled")
            else:
                print("Autodrive cannot be enabled: no camera found")
        else:
            print("Switching to manual drive")
            print("Use the arrow keys to control the vehicle")
            print("Press 'A' to switch back to auto-drive")

    def set_steering_angle(self, angle: float):
        # limit the difference with previous steering angle
        if angle - self.steering_angle > 0.1:
            angle = self.steering_angle + 0.1
        elif angle - self.steering_angle < -0.1:
            angle = self.steering_angle - 0.1
        self.steering_angle = angle

        # limit the range of the steering angle
        if self.steering_angle > 0.5:
            self.steering_angle = 0.5
        elif self.steering_angle < -0.5:
            self.steering_angle = -0.5
        
        self.driver.setSteeringAngle(self.steering_angle)

    def change_manual_steer_angle(self, delta: float):
        self.set_auto_drive(False)

        new_angle = self.manual_steering + delta
        if new_angle <= 25.0 and new_angle >= -25.0:
            self.manual_steering = new_angle
            self.set_steering_angle(self.manual_steering)

        if self.manual_steering == 0.0:
            print("Going straight")
        else:
            print("Turning %.2f rad (%s)", self.manual_steering, 
                  "left" if self.manual_steering < 0 else "right")

    def check_keyboard(self):
        key = Keyboard.getKey()
        if key == Keyboard.UP:
            self.set_speed(self.speed + 5.0)
        elif key == Keyboard.DOWN:
            self.set_speed(self.speed - 5.0)
        elif key == Keyboard.LEFT:
            self.change_manual_steer_angle(-1)
        elif key == Keyboard.RIGHT:
            self.change_manual_steer_angle(+1)
        elif key == ord("A"):
            self.set_auto_drive(True)


    def color_diff(self, a: List[int], b: List[int]) -> float:
        diff = 0
        for i in range(3):
            diff += abs(a[i] - b[i])
        return diff
    

    def convert_rgb_to_names(self, rgb_tuple):
        from scipy.spatial import KDTree
        from webcolors import hex_to_rgb
        import webcolors
        # a dictionary of all the hex and their respective names in css3
        css3_db = webcolors.CSS3_HEX_TO_NAMES
        names = []
        rgb_values = []
        for color_hex, color_name in css3_db.items():
            names.append(color_name)
            rgb_values.append(hex_to_rgb(color_hex))
        
        kdt_db = KDTree(rgb_values)
        distance, index = kdt_db.query(rgb_tuple)
        return f'closest match: {names[index]}'


    def process_camera_image(self, image: List[int]):
        """
        Returns the approximate angle of the yellow line
        or UNKNOWN if no line is detected
        """
        # print("image size: {}".format(len(image)))
        # num_image_pixels = self.camera_width * self.camera_height
        # RGB
        yellow = [95, 187, 203]

        sum_x = 0
        pixel_count = 0
        # Camera.saveImage(self.camera, "test.jpg", 100)
        for w in range(self.camera_width):
            for h in range(self.camera_height):
                middle = 64
                around_middle = 10
                # if w in range(middle - around_middle, middle + around_middle):
                    # print("image at w: {} h: {} is {}".format(w, h, image[w][h]))
                red   = Camera.imageGetRed(image, self.camera_width, w, h)
                green = Camera.imageGetGreen(image, self.camera_width, w, h)
                blue  = Camera.imageGetBlue(image, self.camera_width, w, h)
                colorname = self.convert_rgb_to_names((red, green, blue))
                othercolorname = self.convert_rgb_to_names(self.camera.getImageArray()[w][h])
                if colorname != 'darkslategray':
                    print("image at w: {} h: {} is {}".format(w, h, colorname))
                if othercolorname != 'darkslategray':
                    print("image at w: {} h: {} is {}".format(w, h, othercolorname))
                if self.color_diff([red, green, blue], yellow) < 50:
                    print("yellow found")
                    x = w
                    sum_x += x
                    pixel_count += 1
        
        if pixel_count == 0:
            return UNKNOWN
        
        return (float(sum_x) / float(pixel_count) / float(self.camera_width) - 0.5) * self.camera_fov

    def filter_angle(self, angle: float) -> float:
        """
        Filter the angle to avoid sudden changes
        """
        if angle == UNKNOWN or self.filter_angle_first_call:
            self.filter_angle_first_call = False
            self.previous_filtered_angles = [0.0] * FILTER_SIZE
        else:
            self.previous_filtered_angles.pop(0)

        if angle == UNKNOWN:
            return UNKNOWN
        
        self.previous_filtered_angles.append(angle)
        return np.mean(self.previous_filtered_angles)
    
    def process_sick_data(self, sick_data: float) -> float:
        """
        Returns the approximate angle of the obstacle
        or UNKNOWN if no obstacle is detected
        """
        half_area = 20  # check within +/- 20 degrees around the center
        sum_x = 0
        collision_count = 0
        for i in range(self.sick_width // 2 - half_area, self.sick_width // 2 + half_area):
            obstacle_range = sick_data[i]
            if obstacle_range < 20.0:
                sum_x += i
                collision_count += 1
                self.obstacle_distance += obstacle_range
                
        if collision_count == 0:
            return UNKNOWN
        self.obstacle_distance /= collision_count
        # TODO caller should set obstacle_distance before calling this function
        return (float(sum_x) / float(collision_count) / float(self.sick_width) - 0.5) * self.sick_fov

    def update_display(self):
        needle_length = 50.0

        # draw the background
        Display.imagePaste(self.display, self.speedometer_image, 0, 0, False)

        # draw the speedometer needle
        current_speed = self.driver.getCurrentSpeed()
        if current_speed < 0 or math.isnan(current_speed):
            current_speed = 0

        alpha = current_speed / 260.0 * 3.72 - 0.27
        x = - needle_length * math.cos(alpha)
        y = - needle_length * math.sin(alpha)
        Display.drawLine(self.display, 100, 95, 100 + x, 95 + y)

        # draw text
        txt = "GPS coordinates: (%.2f, %.2f)" % (self.gps_coords[self.X], self.gps_coords[self.Z])
        Display.drawText(self.display, txt, 10, 130)
        txt = "GPS Speed: %.2f" % self.gps_speed
        Display.drawText(self.display, txt, 10, 140)

    def compute_gps_speed(self):
        coords = self.gps.getValues()
        speed_ms = self.gps.getSpeed()

        self.gps_speed = speed_ms * 3.6  # convert to km/h
        self.gps_coords = coords

    def apply_PID(self, yellow_line_angle: float):
        PID_old_value = 0.0
        PID_integral = 0.0

        if self.PID_need_reset:
            PID_need_reset = False
            PID_integral = 0.0
            PID_old_value = yellow_line_angle

        # anti-windup mechanism
        if (yellow_line_angle < 0) != (PID_old_value < 0):
            PID_integral = 0.0

        diff = yellow_line_angle - PID_old_value
        # limit the integral term to avoid saturation
        if PID_integral < 30 and PID_integral > -30:
            PID_integral += yellow_line_angle

        PID_old_value = yellow_line_angle
        return self.KP * yellow_line_angle + self.KD * diff + self.KI * PID_integral
    

def normalizeToRange(value, minVal, maxVal, newMin, newMax, clip=False):
    """
    Normalize value to a specified new range by supplying the current range.

    :param value: value to be normalized
    :param minVal: value's min value, value ∈ [minVal, maxVal]
    :param maxVal: value's max value, value ∈ [minVal, maxVal]
    :param newMin: normalized range min value
    :param newMax: normalized range max value
    :param clip: whether to clip normalized value to new range or not
    :return: normalized value ∈ [newMin, newMax]
    """
    value = float(value)
    minVal = float(minVal)
    maxVal = float(maxVal)
    newMin = float(newMin)
    newMax = float(newMax)

    if clip:
        return np.clip((newMax - newMin) / (maxVal - minVal) * (value - maxVal) + newMax, newMin, newMax)
    else:
        return (newMax - newMin) / (maxVal - minVal) * (value - maxVal) + newMax


class FollowerCarRobot(DeepbotsSupervisorEnv):
    def __init__(self, image_width, image_height, image_channels=3, action_steps=10):
        super().__init__()
        self.observation_space = Box(low=0, high=255, 
                                     shape=(image_width, image_height, image_channels), 
                                     dtype=np.float32)
        self.action_sapce = Discrete(action_steps)

        self.steps_per_episode = 1000
        self.episode_score = 0
        self.episode_scores = []

    def get_reward(self):
        return 1
    
    def is_done(self, controller: VehicleController):
        # check if we have reached the end of the episode
        print("episode score: ", self.episode_score)
        if self.episode_score > 950:
            return True

        # check if we have veered off the road
        current_image = Camera.getImage(controller.camera)
        # current_image = controller.camera.getImageArray()
        # print("is equal: ", np.array_equal(current_image, other_image))
        # print("current image shape: ", np.array(current_image).shape)
        # print("other image shape: ", np.array(other_image).shape)
        # import ipdb; ipdb.set_trace()
        current_angle = controller.process_camera_image(current_image)
        print("current angle: ", current_angle)
        if current_angle == UNKNOWN or abs(current_angle) > 25:
            return True
        
        # check if we lost the lead car
        num_recognition_objects = Camera.getRecognitionNumberOfObjects(controller.camera)
        recognition_objects = Camera.getRecognitionObjects(controller.camera)
        lead_present = "BMW X5" in [recognition_objects[i].getModel() for i in range(num_recognition_objects)]
        print("lead present: ", lead_present)
        print("num recognition objects: ", num_recognition_objects)
        if num_recognition_objects == 0 or not lead_present:
            return True
        
        return False
    
    def solved(self):
        if len(self.episode_scores) > 100:
            return np.mean(self.episode_scores[-100:]) > 950
        
    def get_default_observation(self):
        return np.zeros(self.observation_space.shape)
    
    def convert_action(self, action: int):
        # convert discrete action to continuous action
        action = normalizeToRange(action, 0, self.action_space.n, -0.5, 0.5)
        return action


if __name__ == "__main__":
    env = FollowerCarRobot(image_width=128, image_height=64, image_channels=3, action_steps=10)
    robot = env
    driver = Driver()
    controller = VehicleController(driver=driver, robot=robot)

    if not controller.has_camera:
        print("Autonomous Car cannot drive without camera")
        sys.exit(1)
    else:
        print("Camera found")

    # print("camera image = {}", env.getDevice("camera").getImageArray())

    # camera_image = env.getDevice("camera").getImageArray()
    # print("camera image = {}", camera_image)
    image_width = env.getDevice("camera").getWidth()
    print("image width = {}", image_width)
    image_height = env.getDevice("camera").getHeight()
    print("image height = {}", image_height)
    agent = PPOAgent(image_height=image_height, image_width=image_width, image_channels=3, quantized_actions=10)
    # del env
    # env = FollowerCarRobot(image_width=image_width, image_height=image_height, image_channels=3, action_steps=10)

    # Let's see if there is a previously saved model
    try:
        agent.load(os.path.dirname(os.path.abspath(__file__)) + "/model")
        print("Loaded model from disk")
        solved = True
    except:
        print("No previous model found, training from scratch")
        solved = False

    i = 0
    episode_count = 0
    steps = 0

    # Main loop:
    # - perform simulation steps until Webots is stopping the controller
    while driver.step() != -1 and not solved:

        observation = env.reset()
        # print("observation = {}", np.array(observation).shape)
        env.episode_score = 0
        
        # update sensors only every TIME_STEP milliseconds
        if i % (TIME_STEP // robot.getBasicTimeStep()) == 0:
            steps += 1
            # read sensors
            if controller.has_camera:
                camera_image = np.array(robot.getDevice("camera").getImageArray(), dtype=np.float32)
                camera_image = einops.rearrange(camera_image, 'h w c -> c h w')
                assert camera_image.shape[0] == 3
                # print("camera image = {}", np.array(camera_image).shape)
            if controller.enable_collision_avoidance:
                sick_data = controller.sick.getRangeImage()

            num_recognition_objects = Camera.getRecognitionNumberOfObjects(controller.camera)
            recognition_objects = Camera.getRecognitionObjects(controller.camera)
            distance_to_lead = UNKNOWN
            for i in range(num_recognition_objects):
                if recognition_objects[i].getModel() == "BMW X5":
                    x = recognition_objects[i].getPosition()[0]
                    y = recognition_objects[i].getPosition()[1]
                    z = recognition_objects[i].getPosition()[2]
                    distance_to_lead = math.sqrt(x * x + y * y + z * z)
                    break

            if controller.autodrive and controller.has_camera:
                controller.set_speed(40.0)
                # orientation_to_lead = controller.process_camera_image(camera_image)
                # orientation_to_lead = controller.filter_angle(orientation_to_lead)
                # if orientation_to_lead != UNKNOWN:
                if distance_to_lead != UNKNOWN:
                    # too far behind the lead car
                    if distance_to_lead > 12.5:
                        controller.set_speed(45.0)
                        driver.setBrakeIntensity(0.0)
                    elif distance_to_lead < 7.5:
                        driver.setBrakeIntensity(1.0)
                    # controller.set_steering_angle(controller.apply_PID(orientation_to_lead))

            # In training mode the agent samples from the probability distribution, naturally implementing exploration
            selectedAction, actionProb = agent.work(camera_image, type_="selectAction")
            # Step the supervisor to get the current selectedAction's reward, 
            # the new observation and whether we reached
            # the done condition
            controller.set_steering_angle(selectedAction)
            reward = env.get_reward()
            done = env.is_done(controller)
            newObservation = np.array(robot.getDevice("camera").getImageArray(), dtype=np.float32)
            newObservation = einops.rearrange(camera_image, 'h w c -> c h w')
            # newObservation, reward, done, info = env.step([selectedAction])
            trans = Transition(camera_image, selectedAction, actionProb, reward, newObservation)
            agent.storeTransition(trans)
            if done:
                # Save the episode's score
                env.episode_scores.append(env.episode_score)
                agent.trainStep(batchSize=steps + 1)
                solved = env.solved()
            else:
                driver.setBrakeIntensity(0.5)
                controller.PID_need_reset = True

            env.episode_score += reward
            observation = newObservation

            # update stuff
            if controller.has_gps:
                controller.compute_gps_speed()
            if controller.enable_display:
                controller.update_display()

        i += 1

    if solved:
        print("Solved in {} episodes".format(episode_count))
        agent.save(os.path.dirname(os.path.abspath(__file__)) + "/model")
    else:
        print("Did not solve after {} episodes".format(episode_count))

    observation = robot.getDevice("camera").getImageArray()
    while driver.step() != -1:
        selectedAction, actionProb = agent.work(observation, type_="selectActionMax")
        observation, reward, done, info = env.step([selectedAction])

    # Enter here exit cleanup code
    driver.cleanup()