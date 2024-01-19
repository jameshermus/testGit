import numpy as np
import matplotlib.pyplot as plt
import os
from env1D import env1D
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from helperFunctions import defineDirectories
from modelClassical import modelClassical_f, modelClassical_submovement, modelClassical_x0
from stable_baselines3.common.evaluation import evaluate_policy
from helperFunctions import submovement

import imageio

import gym
import numpy as np
import cv2
from stable_baselines3 import PPO

class plotModels():
    def __init__(self,controllerType,dateInput = None, classicalBool = False,saveVideoBool = False):
        self.controllerType = controllerType
        self.dateInput = dateInput
        self.classicalBool = classicalBool
        self.saveVideoBool = saveVideoBool

        if not self.classicalBool:
            self.nameLearnClassic = 'Learn'
        elif self.classicalBool:
            self.nameLearnClassic = 'classical'
        _, self.log_path, self.save_path = defineDirectories(controllerType,self.dateInput)  

        self.env = env1D(self.controllerType,render_mode='rgb_array')

        # Add code to import a model
        if self.classicalBool == False:
            self.model = PPO.load(self.save_path+'/best_model')
            # self.model = PPO.load(self.save_path+'/model_final')
        elif self.classicalBool == True:
            if controllerType == 'f':
                self.model = modelClassical_f(controllerType, self.env.tolerance_x)
            elif controllerType == 'x0':
                self.model = modelClassical_x0(controllerType, self.env.tolerance_x)
            elif controllerType == 'submovement':
                submovementParam = {'thresholdLatency':self.env.robot.thresholdLatency,
                                    'A_high':self.env.robot.A_high,
                                    'A_low':self.env.robot.A_low,
                                    'D_high':self.env.robot.D_high}
                self.model = modelClassical_submovement(controllerType, tolerance_x = self.env.tolerance_x, submovementParam = submovementParam)

        # Hard code submovement as a sanity check
        self.N = int(np.ceil(self.env.timeMax/self.env.timeStep))
        # self.N = int(np.ceil(self.env.timeMax/(self.env.timeStep*self.env.downSampleFactor)))

        # evaluate_policy(self.model, self.env, n_eval_episodes=50)

        self.simulateModel()
        # self.record_video(video_length=self.N, video_file = self.save_path+"/video"+self.controllerType+"_"+self.nameLearnClassic+".mp4")
        self.plotResults()
        self.plotRewardHistogram() # Run last or it causes problems with submovementActionVec

        pass
    
    def simulateModel(self):
        
        obs = self.env.reset()[0]
        terminated = False
        self.score = 0
        episode = 0
        count = 0
        self.x = np.zeros(self.N)
        self.x_dot = np.zeros(self.N)
        self.f = np.zeros(self.N)
        self.target = obs[2]
        if self.controllerType == 'submovement':
            self.submovementActionVec = []

        # Function to record a video of the environment
        # Initialize video writer
        if self.saveVideoBool:
            fps = int(1/self.env.timeStep)
            # fps = int(self.downSampleFactor/self.env.timeStep)
            height, width, _ = self.env.render().shape
            video_writer = cv2.VideoWriter(self.save_path+"/"+self.controllerType+"_"+self.nameLearnClassic+"_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while not terminated:
            if self.saveVideoBool:
                # Render the environment
                frame = self.env.render()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # Save frame to video
                video_writer.write(frame)

            action,_ = self.model.predict(obs) # Use trained model

            if self.controllerType == 'submovement' and action > 0:
                self.submovementActionVec.append([action,self.env.time])

            obs, reward, terminated, truncated, info = self.env.step(action)
            self.f[count],_ = self.env.robot.get_force(obs, action, self.env.time) 
            self.x[count] = obs[0]
            self.x_dot[count] = obs[1]
            # actionList.append(action.append(env.time))
            self.score += reward
            count += 1
            print('Episode:{} Score:{} Action:{}'.format(episode, self.score, action))
        if self.saveVideoBool:
            # Release video writer
            video_writer.release()
        
        self.env.close()
        self.timeVec = np.arange(self.N) * self.env.timeStep
        # self.timeVec = np.arange(self.N) * (self.env.timeStep*self.env.downSampleFactor)

        pass

    def evaluate_policy_all_rewards(self,numEpisodes = 100):
        totalRewardList = []
        for episode in range(numEpisodes):
            terminated = False
            totalReward = 0
            obs,_ = self.env.reset()
            if self.classicalBool: 
                self.model.reset()
            while not terminated:
                action,_ = self.model.predict(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                totalReward += reward
            totalRewardList.append(totalReward)
        meanReward = np.mean(totalRewardList)
        stdReward = np.std(totalRewardList)
        return totalRewardList, meanReward, stdReward

    def plotResults(self):
        plt.rcParams.update({'font.size': 16})
        plt.rc('lines', linewidth=2.5)

        if self.controllerType == 'submovement' and len(self.submovementActionVec) > 0:
            n = len(self.submovementActionVec)
            submovements_x = np.zeros((n,len(self.timeVec)))
            submovements_v = np.zeros((n,len(self.timeVec)))
            for i, sub in enumerate(self.submovementActionVec):
                _, duration, amplitude = self.env.robot.get_subParam(sub[0])
                for j, t in enumerate(self.timeVec):
                    submovements_x[i][j], submovements_v[i][j] = submovement(duration,amplitude,sub[1],t)
            totalSubmovements_v = np.sum(submovements_v,0)

        plt.figure()
        plt.plot(self.timeVec,self.x)
        plt.plot(self.timeVec,self.target*np.ones(self.N),'k')
        plt.plot(self.timeVec,(self.target + self.env.tolerance_x)*np.ones(self.N),'--k')
        plt.plot(self.timeVec,(self.target - self.env.tolerance_x)*np.ones(self.N),'--k')
        plt.xlabel('Time(s)')
        plt.ylabel('x(m)')
        plt.subplots_adjust(left=0.15, bottom=0.15)
        # plt.show()
        plt.savefig(self.save_path+"/"+self.controllerType+"_"+self.nameLearnClassic+"_position.png")
        plt.close(plt.gcf().number)
                
        plt.figure()
        plt.plot(self.timeVec,self.x_dot)
        plt.xlabel('Time(s)')
        plt.ylabel('v(m/s)')
        if self.controllerType == 'submovement' and len(self.submovementActionVec) > 0:
            # plt.plot(self.timeVec,totalSubmovements_v)
            for i in range(n):
                plt.plot(self.timeVec,submovements_v[i][:])
        plt.subplots_adjust(left=0.15, bottom=0.15)
        plt.savefig(self.save_path+"/"+self.controllerType+"_"+self.nameLearnClassic+"_velocity.png")
        plt.close(plt.gcf().number)

        plt.figure()
        plt.plot(self.timeVec,self.f)
        plt.xlabel('Time(s)')
        plt.ylabel('f(N)')
        plt.subplots_adjust(left=0.15, bottom=0.15)
        plt.savefig(self.save_path+"/"+self.controllerType+"_"+self.nameLearnClassic+"_force.png")
        plt.close(plt.gcf().number)

        pass

    def plotRewardHistogram(self):
        totalRewardList, meanReward, stdReward = self.evaluate_policy_all_rewards(numEpisodes = 100)

        plt.figure()
        plt.hist(totalRewardList, bins=20, edgecolor='black', color='skyblue')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        mean_value = 0
        std_value = 0
        title_text = f'{meanReward:.2f} ± {stdReward:.2f}'
        plt.xlim([-50,600])
        plt.title(title_text)
        plt.subplots_adjust(left=0.15, bottom=0.15)
        plt.savefig(self.save_path+"/"+self.controllerType+"_"+self.nameLearnClassic+"_rewardHistogram.png")
        plt.close(plt.gcf().number)
        pass

    def makeExampleSubmovementPlot(self):
        submovementActionVec = [[1,0],[2,0],[3,0],[4,0]] # This is a local definition for plotting sake and is diffrent from self.submovmentActionVec
        n = len(submovementActionVec)
        submovements_x = np.zeros((n,len(self.timeVec)))
        submovements_v = np.zeros((n,len(self.timeVec)))
        for i, sub in enumerate(submovementActionVec):
            _, duration, amplitude = self.env.robot.get_subParam(sub[0])
            for j, t in enumerate(self.timeVec):
                submovements_x[i][j], submovements_v[i][j] = submovement(duration,amplitude,sub[1],t)

        plt.figure()
        for i in range(n):
            plt.plot(self.timeVec,submovements_x[i][:])
        plt.xlabel('Time(s)')
        plt.ylabel('x(m)')
        plt.xlim([0, self.env.robot.D_high])
        plt.subplots_adjust(left=0.15, bottom=0.15)
        plt.savefig(self.save_path+"/submovmentExample_position.png")
        plt.close(plt.gcf().number)
                
        plt.figure()
        for i in range(n):
            plt.plot(self.timeVec,submovements_v[i][:])
        plt.xlim([0, self.env.robot.D_high])
        plt.xlabel('Time(s)')
        plt.ylabel('v(m/s)')
        plt.subplots_adjust(left=0.15, bottom=0.15)
        plt.savefig(self.save_path+"/submovmentExample_velocity.png")
        plt.close(plt.gcf().number)        


dateInput = '23-08-03'
saveVideoBool = False
# plotModels('f', dateInput,saveVideoBool=saveVideoBool) 
# plotModels('x0', dateInput,saveVideoBool=saveVideoBool)
plotModels('submovement', dateInput,saveVideoBool=saveVideoBool) 

# plotModels('f', dateInput, classicalBool = True, saveVideoBool=saveVideoBool) 
# plotModels('x0', dateInput, classicalBool = True,saveVideoBool=saveVideoBool)
# subPlotObject = plotModels('submovement', dateInput, classicalBool = True,saveVideoBool=saveVideoBool)
# subPlotObject.makeExampleSubmovementPlot()

print('done')