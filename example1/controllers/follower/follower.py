import math
from typing import List, Optional
import numpy as np

from controller import Driver, Keyboard, Camera, Lidar, GPS, Display, ImageRef, Robot

# Size of the yellow line angle filter
FILTER_SIZE = 3

# Line following PID
KP = 0.25
KI = 0.006
KD = 2

UNKNOWN = 99999.99
TIME_STEP = 50


class VehicleController:
    def __init__(self):
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
        self.speedometer_image: Optional[ImageRef] = None

        # GPS
        self.gps: Optional[GPS] = None
        self.gps_coords = [0.0, 0.0, 0.0]
        self.X = 0
        self.Y = 1
        self.Z = 2
        self.gps_speed = 0.0


    def setup(self):
        Driver.init()

        self.detect_features()
        self.init_camera()
        self.init_sick()
        self.init_gps()
        self.init_display()
        self.start_engine()
        self.print_help()

    def detect_features(self):
        num_devices = Robot.getNumberOfDevices()
        for i in range(num_devices):
            device = Robot.getDeviceByIndex(i)
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
            self.camera = Robot.getCamera("camera")
            self.camera.enable(TIME_STEP)
            self.camera.recognition_enable(TIME_STEP)
            self.camera_width = self.camera.get_width()
            self.camera_height = self.camera.get_height()
            self.camera_fov = self.camera.get_fov()

    def init_sick(self):
        if self.enable_collision_avoidance:
            self.sick = Robot.getDevice("Sick LMS 291")
            self.sick.enable(TIME_STEP)
            self.sick_width = self.sick.get_horizontal_resolution()
            self.sick_range = self.sick.get_max_range()
            self.sick_fov = self.sick.get_fov()

    def init_gps(self):
        if self.has_gps:
            self.gps = Robot.getDevice("gps")
            self.gps.enable(TIME_STEP)

    def init_display(self):
        if self.enable_display:
            self.display = Robot.getDevice("display")
            self.speedometer_image = self.display.image_load("speedometer.png")

    def start_engine(self):
        if self.has_camera:
            self.set_speed(40.0)  # km/h

        Driver.setHazardFlashers(True)
        Driver.setDippedBeams(True)
        Driver.setAntifogLights(True)
        Driver.setWiperMode(Driver.SLOW)

    def print_help(self):
        print("Use the arrow keys to control the vehicle")
        print("[Left]/[Right] to steer")
        print("[Up]/[Down] to accelerate/brake")
        print("Press 'A' to switch back to auto-drive")

    def set_speed(self, speed: float):
        # check for max speed
        if speed > self.max_speed:
            speed = self.max_speed

        # check for min speed
        if speed < self.min_speed:
            speed = self.min_speed

        self.speed = speed

        Driver.setCruisingSpeed(self.speed)

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
        
        Driver.setSteeringAngle(self.steering_angle)

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

    def process_camera_image(self, image: List[int]):
        """
        Returns the approximate angle of the yellow line
        or UNKNOWN if no line is detected
        """
        num_image_pixels = self.camera_width * self.camera_height
        yellow = [95, 187, 203]

        sum_x = 0
        pixel_count = 0
        for i in range(0, num_image_pixels, 4):
            if color_diff(image[i:i + 3], yellow) < 30:
                x = (i % self.camera_width)
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
        current_speed = Driver.getCurrentSpeed()
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
    


if __name__ == "__main__":
    driver = Driver()
    driver.init()
    controller = VehicleController()

    i = 0

    # Main loop:
    # - perform simulation steps until Webots is stopping the controller
    while driver.step() != -1:
        
        # update sensors only every TIME_STEP milliseconds
        if i % (TIME_STEP // Robot.getBasicTimeStep()) == 0:
            # read sensors
            if controller.has_camera:
                camera_image = Camera.getImage(controller.camera)
            if controller.enable_collision_avoidance:
                sick_data = Lidar.getRangeImage(controller.sick)

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
                orientation_to_lead = controller.process_camera_image(camera_image)
                orientation_to_lead = controller.filter_angle(orientation_to_lead)
                if orientation_to_lead != UNKNOWN:
                    if distance_to_lead != UNKNOWN:
                        # too far behind the lead car
                        if distance_to_lead > 12.5:
                            controller.set_speed(45.0)
                            Driver.setBrakeIntensity(0.0)
                        elif distance_to_lead < 7.5:
                            Driver.setBrakeIntensity(1.0)
                    controller.set_steering_angle(controller.apply_PID(orientation_to_lead))
                else:
                    Driver.setBrakeIntensity(0.5)
                    controller.PID_need_reset = True

            # update stuff
            if controller.has_gps:
                controller.compute_gps_speed()
            if controller.enable_display:
                controller.update_display()

        i += 1

    # Enter here exit cleanup code
    Driver.cleanup()