SelfSupervisedFeatureLearning project set out to produce a network that could mask features such that it maximises the difficulty for the autoencoder in a self-supervised representation learning style of training.
This network includes the "student" which restores the masked image and the "teacher" which provides the masked image
Intitially, I tried various ways of passing the loss of the "student" to the "teacher" though I have pivited to think that reinforcement learning would be more appropriate to the task.
I now hope to train a Deep Q agent such as in Intelligent Masking: Deep Q-Learning for Context Encoding in Medical Image Analysis (https://arxiv.org/pdf/2203.13865.pdf)
I will start by implementing this paper for my current imagenet problem, then I will try to expand on it by increasing the amount of steps it does from 1 to n.
This change could see the agent tracing features more closly. With an element that punishes the agent for masking, perhaps it will be more conservative and only mask the feature so as to maximise loss in the autoencoder and minimise the punishment for its own masking.