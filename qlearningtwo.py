from torch.utils.data.dataloader import DataLoader
import configs
import torch
from torch import nn
import os
from dataset import Dataset
from encDec import encDec
from encDecMask import encDecMask
from torch.autograd import Variable
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F
import random
import numpy as np
from tensorforce.environments import Environment
# from tensorforce.agents import Agent
# import tensorforce.agents.dqn as dqnAgent
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# tensorforce 0.6.5 requires numpy==1.19.5, but you have numpy 1.23.5 which is incompatible.




# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))



def compute_avg_return(environment, policy, num_episodes):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

class CustomEnvironment(Environment):

    def __init__(self):
        super().__init__()

    def states(self):
        return dict(type='float', shape=(8,))

    def actions(self):
        #return dict(type='int', num_values=4)
        return dict(type='int', shape=(350,350)) #as large as the image - window size

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        #return super().max_episode_timesteps()
        return 1 #only 1 step for this guy as per paper

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        state = np.random.random(size=(8,))
        return state

    def execute(self, actions):
        #Q=Q+actions*(loss-Q)
        #next_state = np.random.random(size=(8,))
        next_state=actions
        terminal=True
        #terminal = False  # Always False if no "natural" terminal state
        #reward = np.random.random()
        reward=loss
        return next_state, terminal, reward


globalLoss=0
def rewardLossFcn(img, comparison):

    loss = -torch.mean((comparison-img)**2)
    print(loss)
    return loss

def criterionFcn(img, comparison):

    loss = torch.mean((comparison-img)**2)
    globalLoss=loss
    print('criterionFcn')
    print(loss)
    return loss

def punishLossFcn(img, comparison):

    print('globalLoss')
    print(globalLoss)
    loss = torch.mean((comparison-img)**2)*10^-8+globalLoss.np()
    print('punishLossFcn')
    print(loss)
    return loss

if __name__ == '__main__':
    print("Begin deepq")

    env=CustomEnvironment()
    env.reset()

    fc_layer_params = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])
    learning_rate=10^-5
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    random_policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(),
                                                    env.action_spec())

    compute_avg_return(env, random_policy, 10)

    #config = dict(device='GPU') this switches tensorforce to gpu
    CUDA_LAUNCH_BLOCKING = 1
    torch.autograd.set_detect_anomaly(True)
    if not os.path.exists(configs.resultPath):
        os.makedirs(configs.resultPath)
    torch.manual_seed(configs.seed)
    studentModel = encDec().cuda()
    studentModel = studentModel.to(configs.device)
    trainDataset = Dataset(configs.dataPath)
    validationDataset = Dataset(configs.validationDataPath)
    teacherModel = encDecMask().cuda()
    teacherModel = teacherModel.to(configs.device)

    #print(trainDataset)
    trainDataLoader = DataLoader(dataset=trainDataset, batch_size=configs.batch_size, shuffle=False, num_workers=configs.threads, pin_memory=True, drop_last=True)
    validationDataLoader = DataLoader(dataset=validationDataset, batch_size=configs.batch_size, shuffle=False, num_workers=configs.threads, pin_memory=True, drop_last=True)

    criterion = nn.MSELoss()
    studentOptimizer = torch.optim.Adam(studentModel.parameters(), lr=configs.learning_rate, weight_decay=1e-5)
    teacherOptimizer = torch.optim.Adam(teacherModel.parameters(), lr=configs.learning_rate, weight_decay=1e-5)
    counter=0
    # adapt these
    for epoch in range(configs.num_epochs):
        with tqdm(trainDataLoader) as tepoch:
            saveImg=0;
            debugImg=0;
            total_loss = 0
            totalTeacherLoss=0
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                img=Variable(data).cuda()
                tempimg=teacherModel(img)


                print(tempimg.shape)
                teacherImg=torch.sum(tempimg,1)
                print(teacherImg)
                maskargmax=torch.full((2,400,400),0).cuda()
                maskargmax[1,:,:]=torch.reshape(teacherImg,(400,400))
                mask = torch.argmax(maskargmax, dim=0) #if the temping value is > 0.5, then 1 is recorded in the mask, otherwise 0
                print(mask)

                maskedImg=(img*mask)
                maskedImg.requires_grad_(requires_grad=True)

                studentOutput = studentModel(maskedImg)
                loss = criterion(studentOutput, maskedImg)

                lossImg=teacherModel(studentOutput)
                teacherLoss=rewardLossFcn(tempimg, tempimg)

                if(counter%100==0):
                    save_image(tempimg, configs.resultPath + '/tempimg_{}.png'.format(counter))
                    save_image(maskedImg, configs.resultPath + '/maskedimg_{}.png'.format(counter))
                    save_image(studentOutput, configs.resultPath + '/output_{}.png'.format(counter))

                    print("saved")
                counter+=1

                studentOptimizer.zero_grad()
                loss.backward()
                studentOptimizer.step()

                teacherOptimizer.zero_grad()
                teacherLoss.backward()
                teacherOptimizer.step()

                total_loss += loss.data
                totalTeacherLoss+=teacherLoss.data
                tepoch.set_postfix(data="train",loss=loss.item(), total_loss=total_loss/trainDataset.__len__(),
                                   teacherLoss=teacherLoss,totalTeacherLoss=totalTeacherLoss/trainDataset.__len__())
                saveImg=studentOutput
            save_image(saveImg, configs.resultPath + '/image_train_{}.png'.format(epoch))
