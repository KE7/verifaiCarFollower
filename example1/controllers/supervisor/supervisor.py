# TODOs:
#   1) Add the orientation specification. (Postpone)
#   2) Should we terminate as soon as we falsify? Both.
#   3) Run monolithic falsification.
#   4) Write seperate Scenic programs for each subscenarios.
#   5) Write a script that runs these Scenic programs by using VerifAI for each subscenario for falsifying, i.e., write a script implementing the falsification algorithm of Scenic-SMC.

# Problem: how can we have realistic physics for each subscenario?


from verifai.simulators.webots.webots_task import webots_task
from verifai.simulators.webots.client_webots import ClientWebots
from time import sleep
try:
    from controller import Supervisor
except ModuleNotFoundError:
    import sys
    sys.exit("This functionality requires webots to be installed")

from dotmap import DotMap
import numpy as np
import math
import sys

TIME_STEP = 10

class Task(webots_task):
    def __init__(self, N_SIM_STEPS, supervisor):
        super().__init__(N_SIM_STEPS, supervisor)
        self.trajectory = []
        self.epsilon = 0.001
        self.sample_ind = 0
        self.mode = None
        self.ind = None
        self.monolithic_scenario_convergence_flags = {"L": False, "S": False, "R": False}
        self.previous_sem = None

    def use_sample(self, sample):
        self.sample_ind += 1
        print("Sample:", self.sample_ind)
        self.trajectory = []
        ego = self.supervisor.getFromDef("FOLLOWER")
        lead = self.supervisor.getFromDef("LEAD")
        obstacle = self.supervisor.getFromDef("OBSTACLE")
        oil_barrel_1 = self.supervisor.getFromDef("OIL_BARREL_1")
        oil_barrel_2 = self.supervisor.getFromDef("OIL_BARREL_2")
        oil_barrel_3 = self.supervisor.getFromDef("OIL_BARREL_3")
        is_obstacle_found = False
        obstacle_positions = {}
        obstacle_headings = {}
        obstacle_colors = {}
        for obj in sample.objects:
            if obj.webotsName == "SUBSCENARIO":
                self.ind = obj.ind[-1]
                self.subscenario = "subscenario" + obj.ind
                self.mode = obj.mode
                controller_arg = lead.getField("controllerArgs").getMFString(0)
                if controller_arg == "" or controller_arg not in obj.ind:
                    if "L" in obj.ind:
                        lead.getField("controllerArgs").setMFString(0, "L")
                    elif "S" in obj.ind:
                        lead.getField("controllerArgs").setMFString(0, "S")
                    elif "R" in obj.ind:
                        lead.getField("controllerArgs").setMFString(0, "R")
                    ego.restartController()
                    lead.restartController()
            elif obj.webotsName == "SCENARIO":
                self.ind = obj.ind
                self.subscenario = "scenario"
                self.mode = obj.mode
                controller_arg = lead.getField("controllerArgs").getMFString(0)
                if controller_arg == "" or controller_arg not in obj.ind:
                    if "L" in obj.ind:
                        lead.getField("controllerArgs").setMFString(0, "L")
                    elif "S" in obj.ind:
                        lead.getField("controllerArgs").setMFString(0, "S")
                    elif "R" in obj.ind:
                        lead.getField("controllerArgs").setMFString(0, "R")
                    ego.restartController()
                    lead.restartController()
            elif obj.webotsName == "FOLLOWER":
                position = ego.getField("translation").getSFVec3f()
                position[0], position[1] = obj.position
                position[2] = 0.31
                ego.getField("translation").setSFVec3f(position)
                heading = ego.getField("rotation").getSFRotation()
                heading[3] = obj.heading
                ego.getField("rotation").setSFRotation(heading)
            elif obj.webotsName == "LEAD":
                position = lead.getField("translation").getSFVec3f()
                position[0], position[1] = obj.position
                position[2] = 0.31
                lead.getField("translation").setSFVec3f(position)
                heading = lead.getField("rotation").getSFRotation()
                heading[3] = obj.heading
                lead.getField("rotation").setSFRotation(heading)
            elif obj.webotsName == "OBSTACLE":
                is_obstacle_found = True
                obstacle_positions[obj.type[-1]] = obj.position
                obstacle_headings[obj.type[-1]] = obj.heading
                obstacle_colors[obj.type[-1]] = obj.color
            elif obj.webotsName == "OIL_BARREL_1":
                position = oil_barrel_1.getField("translation").getSFVec3f()
                position[0], position[1] = obj.position
                oil_barrel_1.getField("translation").setSFVec3f(position)
                heading = oil_barrel_1.getField("rotation").getSFRotation()
                heading[3] = obj.heading
                oil_barrel_1.getField("rotation").setSFRotation(heading)
            elif obj.webotsName == "OIL_BARREL_2":
                position = oil_barrel_2.getField("translation").getSFVec3f()
                position[0], position[1] = obj.position
                oil_barrel_2.getField("translation").setSFVec3f(position)
                heading = oil_barrel_2.getField("rotation").getSFRotation()
                heading[3] = obj.heading
                oil_barrel_2.getField("rotation").setSFRotation(heading)
            elif obj.webotsName == "OIL_BARREL_3":
                position = oil_barrel_3.getField("translation").getSFVec3f()
                position[0], position[1] = obj.position
                oil_barrel_3.getField("translation").setSFVec3f(position)
                heading = oil_barrel_3.getField("rotation").getSFRotation()
                heading[3] = obj.heading
                oil_barrel_3.getField("rotation").setSFRotation(heading)
        if is_obstacle_found:
            position = obstacle.getField("translation").getSFVec3f()
            position[0], position[1] = obstacle_positions[self.ind]
            obstacle.getField("translation").setSFVec3f(position)
            heading = obstacle.getField("rotation").getSFRotation()
            heading[3] = obstacle_headings[self.ind]
            obstacle.getField("rotation").setSFRotation(heading)
            obstacle.getField("appearance").getSFNode().getField("baseColor").setSFColor(obstacle_colors[self.ind])
        else:
            obstacle.getField("translation").setSFVec3f([-100.0, -100.0, -100.0])
        self.supervisor.step(TIME_STEP)
        ego.moveViewpoint()
        return ego, lead

    def _get_distance_value(self, ego, lead):
        ego_x, ego_y, _ = ego.getPosition()
        lead_x, lead_y, _ = lead.getPosition()
        distance = math.sqrt((ego_x - lead_x)**2 + (ego_y - lead_y)**2)
        value = 0.0
        if distance > 15:
            value = 15 - distance
        elif distance < 5:
            value = distance - 5
        else:
            value = 5 - abs(distance - 10)
        return value

    def _get_heading(self, robot):
        R = np.array(robot.getOrientation()).reshape((3, 3))
        # roll = math.atan2(R[2][1], R[2][2])
        # pitch = math.atan2(-R[2][0], math.sqrt(R[2][1]**2 + R[2][2]**2))
        yaw = math.atan2(R[1][0], R[0][0])
        return yaw

    def run_task(self, sample):
        ego, lead = self.use_sample(sample)
        with open(self.mode + "_csvs/" + self.subscenario + "_samples.csv", "a") as f:
            f.write(str(self.sample_ind) + "," + str(sample) + "\n")
        ego_x, ego_y, _ = ego.getPosition()
        lead_x, lead_y, _ = lead.getPosition()
        sim_steps = 0
        while ((self.subscenario != "subscenario1" or (ego_y < -78.5 and lead_y < -68.5)) and
               (self.subscenario != "subscenario2L" or (ego_x > 5.5 and lead_x > -4.5)) and
               (self.subscenario != "subscenario2S" or (ego_y < -5.5 and lead_y < 4.5)) and
               (self.subscenario != "subscenario2R" or (ego_y < -14.5 and lead_y < -4.5)) and 
               ((ego_y < -78.5 and lead_y < -68.5) or
                ((self.subscenario + self.ind != "scenarioL" or (ego_x > 5.5 and lead_x > -4.5)) and
                 (self.subscenario + self.ind != "scenarioS" or (ego_y < -5.5 and lead_y < 4.5)) and
                 (self.subscenario + self.ind != "scenarioR" or (ego_y < -14.5 and lead_y < -4.5))))):
            self.supervisor.step(TIME_STEP)
            sim_steps += 1
            ego_x, ego_y, _ = ego.getPosition()
            lead_x, lead_y, _ = lead.getPosition()
            distance_value = self._get_distance_value(ego, lead)
            self.trajectory.append(distance_value)
            if distance_value < 0.0:
                break
        with open(self.mode + "_csvs/" + self.subscenario + "_results.csv", "a") as f:
            f.write(str(self.sample_ind) + "," + str(min(self.trajectory)) + "," + str(sim_steps) + "\n")
        if self.trajectory[-1] >= 0.0:
            ego_x, ego_y, _ = ego.getPosition()
            ego_heading = self._get_heading(ego)
            lead_x, lead_y, _ = lead.getPosition()
            lead_heading = self._get_heading(lead)
            post_conditions = np.genfromtxt(self.mode + "_csvs/" + self.subscenario + "_post_conditions.csv", delimiter=",", skip_header=True)
            delta_sem = None
            point = np.array([ego_x, ego_y, ego_heading, lead_x, lead_y, lead_heading])
            if post_conditions.ndim == 2 and post_conditions.size > 1:
                sem = np.std(post_conditions, ddof=1, axis=0) / np.sqrt(np.size(post_conditions, axis=0))
                if self.previous_sem is not None:
                    delta_sem = abs(self.previous_sem - sem)
                self.previous_sem = sem
            print("Delta SEM:", delta_sem)
            with open(self.mode + "_csvs/" + self.subscenario + "_post_conditions.csv", "a") as f:
                f.write(str(ego_x) + "," + str(ego_y) + "," + str(ego_heading) + "," + str(lead_x) + "," + str(lead_y) + "," + str(lead_heading) + "\n")
            if delta_sem is not None and np.all(delta_sem <= self.epsilon):
                print("Post conditions stabilized.")
                return {} # This causes a ValueError that is catched by the analyze script.
        else:
            if self.mode == "falsify":
                print("Falsification ended successfully.")
                sys.exit() # This causes a EOFError that is catched by the analyze script.
            else:
                pass
        sim_results = {}
        sim_results["distance"] = [(j*0.010 + 0.010, b) for j, b in enumerate(self.trajectory)]
        return sim_results

PORT = 8888
BUFSIZE = 4096
N_SIM_STEPS = 2000
supervisor = Supervisor()
simulation_data = DotMap()
simulation_data.port = PORT
simulation_data.bufsize = BUFSIZE
simulation_data.task = Task(N_SIM_STEPS=N_SIM_STEPS, supervisor=supervisor)
client_task = ClientWebots(simulation_data)
print("Starting the simulation...")
supervisor.simulationResetPhysics()
supervisor.simulationReset()
supervisor.getFromDef("LEAD").restartController()
supervisor.getFromDef("FOLLOWER").restartController()
supervisor.step(TIME_STEP)
i = 0
while True:
    # try:
        # print("Connecting to the falsifier...")
        # result = client_task.run_client()
        # if not result:
        #     print("End of scene generation")
        #     supervisor.simulationResetPhysics()
        #     supervisor.simulationReset()
        #     supervisor.getFromDef("FOLLOWER").restartController()
        #     supervisor.getFromDef("LEAD").restartController()
        #     supervisor.step(TIME_STEP)
        
    # except:
    #     supervisor.simulationResetPhysics()
    #     supervisor.simulationReset()
    #     print("Waiting for new connection...", flush=True)
    #     supervisor.getFromDef("FOLLOWER").restartController()
    #     supervisor.getFromDef("LEAD").restartController()
    #     supervisor.step(TIME_STEP)
    #     sleep(70)
    i += 1
    supervisor.step(TIME_STEP)
