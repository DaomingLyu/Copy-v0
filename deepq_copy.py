# Deep Q network -- Copy-v0

import gym
import numpy as np
import tensorflow as tf
import math
import random
import nplot

# HYPERPARMETERS
H = 30
H2 = 30
batch_number = 50
gamma = 0.99
explore = 1
num_of_episodes_between_q_copies = 50
learning_rate=1e-3
explore_decay = .9999
observe_size = 1
dropout_rate = 1

size1 = 2
size2 = 2
size3 = 5
total = 9
    
def one_hot_maker(x,y,z):
    if(x == 0):
        a = 1
        b = 0
    else:
        a = 0
        b = 1
    if(y == 0):
        c = 1
        d = 0
    else:
        c = 0
        d = 1
    if(z == 0):
        e = 1
        f = 0
        g = 0
        h = 0
        i = 0
    if(z == 1):
        e = 0
        f = 1
        g = 0
        h = 0
        i = 0
    if(z == 2):
        e = 0
        f = 0
        g = 1
        h = 0
        i = 0        
    if(z == 3):
        e = 0
        f = 0
        g = 0
        h = 1
        i = 0
    if(z == 4):
        e = 0
        f = 0
        g = 0
        h = 0
        i = 1
    return [a, b, c, d, e, f, g, h, i]
    
    
if __name__ == '__main__':

    env = gym.make('Copy-v0')
    print "Gym input is ", env.action_space
    print "Gym observation is ", env.observation_space
    env.monitor.start('training_dir', force=True)
    #Setup tensorflow
    
    tf.reset_default_graph()

    #First Q Network
    w1 = tf.Variable(tf.random_uniform([observe_size,H], -.10, .10))
    bias1 = tf.Variable(tf.random_uniform([H], -.10, .10))
    
    w2 = tf.Variable(tf.random_uniform([H,H2], -.10, .10))
    bias2 = tf.Variable(tf.random_uniform([H2], -.10, .10))
    
    w3 = tf.Variable(tf.random_uniform([H2,total], -.10, .10))
    bias3 = tf.Variable(tf.random_uniform([total], -.10, .10))
    
    states = tf.placeholder(tf.float32, [None, observe_size], name="states")  # This is the list of matrixes that hold all observations
    #actions = tf.placeholder(tf.float32, [None, env.action_space.n], name="actions")
    
    hidden_1 = tf.nn.relu(tf.matmul(states, w1) + bias1)
    hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2) + bias2)
    #dropout_1 = tf.nn.dropout(hidden_2, dropout_rate)
    action_values = tf.matmul(hidden_2, w3) + bias3
    
    
    actions = tf.placeholder(tf.float32, [None, 9], name="training_mask")
    #one_hot_actions = tf.one_hot(actions, total) # will need to compile three-hots to pass in
    Q = tf.reduce_sum(tf.mul(action_values, actions), reduction_indices=1) 
    
    #previous_action_masks = tf.placeholder(tf.float32, [None, env.action_space.n], name="p_a_m") # This holds all actions taken 
    #previous_values = tf.reduce_sum(tf.mul(action_values, previous_action_masks), reduction_indices=1) #Combination of action taken and resulting q
    
    #Is there a better way to do this?
    w1_prime = tf.Variable(tf.random_uniform([observe_size,H], -1.0, 1.0))
    bias1_prime = tf.Variable(tf.random_uniform([H], -1.0, 1.0))
    
    w2_prime = tf.Variable(tf.random_uniform([H,H2], -1.0, 1.0))
    bias2_prime = tf.Variable(tf.random_uniform([H2], -1.0, 1.0))

    
    w3_prime = tf.Variable(tf.random_uniform([H2,total], -1.0, 1.0))
    bias3_prime = tf.Variable(tf.random_uniform([total], -1.0, 1.0))
    
    #Second Q network
    
    next_states = tf.placeholder(tf.float32, [None, observe_size], name="n_s") # This is the list of matrixes that hold all observations
    hidden_1_prime = tf.nn.relu(tf.matmul(next_states, w1_prime) + bias1_prime)
    hidden_2_prime = tf.nn.relu(tf.matmul(hidden_1_prime, w2_prime) + bias2_prime)
    #dropout_1_prime = tf.nn.dropout(hidden_2_prime, dropout_rate)
    next_action_values =  tf.matmul(hidden_2_prime, w3_prime) + bias3_prime
    #next_values = tf.reduce_max(next_action_values, reduction_indices=1)   
    
     #need to run these to assign weights from Q to Q_prime
    w1_prime_update= w1_prime.assign(w1)
    bias1_prime_update= bias1_prime.assign(bias1)
    w2_prime_update= w2_prime.assign(w2)
    bias2_prime_update= bias2_prime.assign(bias2)
    w3_prime_update= w3_prime.assign(w3)
    bias3_prime_update= bias3_prime.assign(bias3)
    
    #Q_prime = rewards + gamma * tf.reduce_max(next_action_values, reduction_indices=1)
    
    #we need to train Q

    rewards = tf.placeholder(tf.float32, [None, ], name="rewards") # This holds all the rewards that are real/enhanced with Qprime
    #loss = (tf.reduce_mean(rewards - tf.reduce_mean(action_values, reduction_indices=1))) * one_hot
    loss = tf.reduce_sum(tf.square(rewards - Q)) #* one_hot  
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss) 
    
    #Setting up the enviroment
    
    max_episodes = 40000
    max_steps = 1000

    D = []
    priority_temp = []
    priority_replay = []
    explore = 1.0
    
    rewardList = []
    past_actions = []
    
    episode_number = 0
    episode_reward = 0
    reward_sum = 0
    
    current_priority_size = 0
    
    init = tf.initialize_all_variables()
   
    with tf.Session() as sess:
        sess.run(init)
        #Copy Q over to Q_prime
        sess.run(w1_prime_update)
        sess.run(bias1_prime_update)
        sess.run(w2_prime_update)
        sess.run(bias2_prime_update)
        sess.run(w3_prime_update)
        sess.run(bias3_prime_update)
    
        for episode in xrange(max_episodes):
            print 'Reward for episode %f is %f. Explore is %f. Priority Sample is %f' %(episode,reward_sum, explore, current_priority_size)
            reward_sum = 0
            new_state = []
            new_state.append(env.reset())
            
            for step in xrange(max_steps):
                #if(step == (max_steps-1)):
                #    print 'Made 199 steps!'
                
                #if episode % batch_number == 0:
                env.render()
                    
                #print new_state;
                
                state = list(new_state);
                #print state
                
                if explore > random.random():
                    action = env.action_space.sample()
                    #print action
                    #if(action == 1.0):
                    #    curr_action = [0.0,1.0] 
                    #else:
                    #    curr_action = [1.0,0.0]
                else:
                
                    #get action from policy
                    results = sess.run(action_values, feed_dict={states: np.array([new_state])})
                    #print results
                    # Build action 2, 2, 5
                    action1 = (np.argmax([results[0][0],results[0][1] ]))
                    action2 = (np.argmax([results[0][2],results[0][3] ]))
                    action3 = (np.argmax([results[0][4],results[0][5],results[0][4],results[0][5],results[0][4] ]))
                    #print action1
                    #print action2
                    #print action3
                    #action1 = 1
                    #action2 = 1
                    action = (action1, action2, action3)
                    #if(action == 1.0):
                    #    curr_action = [0.0,1.0]
                    #else:
                    #    cur_action = [1.0,0.0]
                    #print action
                curr_action = action;
                
                temp_state, reward, done, _ = env.step(action)
                new_state = []
                new_state.append(temp_state)
                reward_sum += reward
                
                
                
                D.append([state, curr_action, reward, new_state, done])
                priority_temp.append([state, curr_action, reward, new_state, done])
                #if done:
                #   # step through and increment until done
                #    help_factor = -10
                #    for x in reversed(xrange(len(D))):
                #        if x != (len(D)-1):
                #            if step == max_steps-1:
                #                help_factor = 20
                #                #print "made 200 steps @ reward"
                #            #print x
                #            (D[x])[2] += help_factor 
                #            help_factor += 1;
                #            #print x
                #            #print (D[x])[2]
                #            if (D[x])[4]:
                #                #print x
                #                break
                #    #print D
                
                
                if len(D) > 5000:
                    D.pop(0)
                #Training a Batch
                #samples = D.sample(50)
                sample_size = len(D)
                if sample_size > 500:
                    sample_size = 500
                else:
                    sample_size = sample_size
                 
                if True:
                    samples = [ D[i] for i in random.sample(xrange(len(D)), sample_size) ]
                    #print samples
                    new_states_for_q = [ x[3] for x in samples]
                    #print "Happy States:", new_states_for_q
                    all_q_prime = sess.run(next_action_values, feed_dict={next_states: np.array(new_states_for_q)})
                    y_ = []
                    states_samples = []
                    next_states_samples = []
                    actions_samples = []
                    for ind, i_sample in enumerate(samples):
                        #print i_sample
                        if i_sample[4] == True:
                            #print i_sample[2]
                            y_.append(reward)
                            #print y_
                        else:
                            this_q_prime = all_q_prime[ind]
                            maxq = max(this_q_prime)
                            y_.append(reward + (gamma * maxq))
                            #print y_
                        #y_.append(i_sample[2])
                        states_samples.append(i_sample[0])
                        next_states_samples.append(i_sample[3])
                        actions_samples.append(one_hot_maker(i_sample[1][0],i_sample[1][1],i_sample[1][2]))
                    
                    #print sess.run(loss, feed_dict={states: states_samples, next_states: next_states_samples, rewards: y_, actions: actions_samples, one_hot: actions_samples})
                    #print actions_samples
                    sess.run(train, feed_dict={states: states_samples, next_states: next_states_samples, rewards: y_, actions: actions_samples})
                
                #PRIORITY REPLAY SECTION
    
                if len(priority_replay) > 5000:
                    priority_replay.pop(0)
                #Training a Batch
                #samples = D.sample(50)
                sample_size = len(priority_replay)
                if sample_size > 500:
                    sample_size = 500
                else:
                    sample_size = sample_size
                 
                if sample_size != 0:
                    samples = [ priority_replay[i] for i in random.sample(xrange(len(priority_replay)), sample_size) ]
                    #print samples
                    new_states_for_q = [ x[3] for x in samples]
                    #print "Killer States:", new_states_for_q
                    all_q_prime = sess.run(next_action_values, feed_dict={next_states: np.array(new_states_for_q)})
                    y_ = []
                    states_samples = []
                    next_states_samples = []
                    actions_samples = []
                    for ind, i_sample in enumerate(samples):
                        #print i_sample
                        if i_sample[4] == True:
                            #print i_sample[2]
                            y_.append(reward)
                            #print y_
                        else:
                            this_q_prime = all_q_prime[ind]
                            maxq = max(this_q_prime)
                            y_.append(reward + (gamma * maxq))
                            #print y_
                        #y_.append(i_sample[2])
                        states_samples.append(i_sample[0])
                        next_states_samples.append(i_sample[3])
                        actions_samples.append(one_hot_maker(i_sample[1][0],i_sample[1][1],i_sample[1][2]))
                        current_priority_size = sample_size
                        #print "priority Replay Running with ", sample_size, "samples"
                    #print actions_samples
                    #print sess.run(loss, feed_dict={states: states_samples, next_states: next_states_samples, rewards: y_, actions: actions_samples, one_hot: actions_samples})
                    sess.run(train, feed_dict={states: states_samples, next_states: next_states_samples, rewards: y_, actions: actions_samples})
                    

 
                if done:
                    if reward_sum > 3:
                        priority_replay = priority_replay + priority_temp
                        #print "Priority Replay:", priority_replay
                    else:
                        priority_temp = []
                    break
                        
            if episode % num_of_episodes_between_q_copies == 0:
                sess.run(w1_prime_update)
                sess.run(bias1_prime_update)
                sess.run(w2_prime_update)
                sess.run(bias2_prime_update)
                sess.run(w3_prime_update)
                sess.run(bias3_prime_update)
            
            explore = explore * explore_decay
            if explore < .05:
                explore = .05
                
                
    env.monitor.close()