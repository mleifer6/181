# Imports.
import numpy as np
import numpy.random as npr
import math
from SwingyMonkey import SwingyMonkey

class Learner(object):
	'''
	This agent jumps randomly.
	'''

	def __init__(self):
		self.last_state  = None
		self.last_action = None
		self.last_reward = None
		self.values = {} # (s,a) : Q(s,a)
		self.visited = {}
		self.scores = []

		for vdist_bin in xrange(24):
			for hdist_bin in xrange(15):
				for gravity in [1,4]:
					for velocity in [0,1]:
						for action in xrange(2):
							self.values[((vdist_bin,hdist_bin,gravity, velocity), action)] = 0
							self.visited[((vdist_bin,hdist_bin,gravity, velocity), action)] = 0

		self.alpha = .7
		self.discount = 1
		self.epsilon = .9
		self.gravity = 1
		self.set_gravity = False

	def reset(self):
		self.last_state  = None
		self.last_action = None
		self.last_reward = None
		self.gravity = 1
		self.set_gravity = False

	def set_alpha(self, iter):
		self.alpha = 10.0/(10+iter)

	def set_exploration(self, iter):
		self.epsilon = 1 - (1.0/(1 + iter))
		#self.epsilon = 1 - 0.5*math.exp(-iter/500)
	def action_callback(self, state):
		'''
		Implement this function to learn things and take actions.
		Return 0 if you don't want to jump and 1 if you do.
		'''
		# You might do some learning here based on the current state and the last state.

		# You'll need to select and action and return it.
		# Return 0 to swing and 1 to jump.
		self.scores.append(state['score'])

		# Takes in s
		#vertical_dist = state['monkey']['bot'] - state['tree']['bot']
		vertical_dist = (state['tree']['top'] - state['monkey']['top'] + 200) / 25
		if vertical_dist < 0:
			vertical_dist = 0
		#monkey = (state['monkey']['top'] + state['monkey']['bot'])/(2.0) # 0 to 400
		horizontal_dist = state['tree']['dist'] # 0 to 600
		#gap = (state['tree']['top'] + state['tree']['bot'])/(2.0) # 0 to 400
		velocity = state['monkey']['vel']

		# Convert s to our state space

		#hdist_bin = horizontal_dist / 120
		hdist_bin = (horizontal_dist + 150)/50
		if hdist_bin < 0:
			hdist_bin = 0
		if hdist_bin > 15:
			print "horizonal binning ERROR"
		#monkey_bin = (int)(monkey / 40)
		#gap_bin = (int)(gap / 40)
		#vdist_bin = (vertical_dist + 400) / 160
		#if vdist_bin > 4 or vdist_bin < 0:
		#    print "vertical binning ERROR"
		if velocity > 0:
			velocity_bin = 1
		else:
			velocity_bin = 0
		new_state = (vertical_dist, hdist_bin, self.gravity, velocity_bin)
		if state["monkey"]["top"] >= 300 or state["monkey"]["top"] >= state['tree']['top']:
			new_action = 0 # don't jump for sure
		elif state["monkey"]["top"] <= 100:
			new_action = 1 # jump for sure
		#print vertical_dist
		#new_state = (vdist_bin, hdist_bin, vel_bin, self.gravity)
		#new_state = (vdist_bin, hdist_bin, self.gravity)
		elif self.values[(new_state,0)] >= self.values[(new_state,1)]:
			new_action = 0 # swing
		else:
			new_action = 1 # jump


		# Have s' and a' now so update
		if self.last_state != None and self.last_action != None:
			self.visited[(self.last_state, self.last_action)] += 1
			#self.alpha = math.exp(-1-(max(self.scores)/70.0))
			self.values[(self.last_state, self.last_action)] += (self.alpha)*(self.last_reward + self.discount*self.values[(new_state,new_action)] - self.values[(self.last_state, self.last_action)])
		else:
			self.set_gravity = True
		if self.set_gravity and velocity == -1:
			self.set_gravity = False
			self.gravity = 1
		elif self.set_gravity and velocity == -4:
			self.set_gravity = False
			self.gravity = 4
		#print "state gravity", self.gravity

		if npr.rand() > self.epsilon:
			new_action = 1 - new_action # explore with probablilty 1 - epsilon
		self.last_action = new_action
		self.last_state  = new_state

		return self.last_action

	def reward_callback(self, reward):
		'''This gets called so you can see what reward you get.'''
		# Takes in r(s,a)
		self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
	'''
	Driver function to simulate learning by having the agent play a sequence of games.
	'''

	for ii in range(iters):
		# Make a new monkey object.
		swing = SwingyMonkey(sound=False,                  # Don't play sounds.
							 text="Epoch %d" % (ii),       # Display the epoch on screen.
							 tick_length = t_len,          # Make game ticks super fast.
							 action_callback=learner.action_callback,
							 reward_callback=learner.reward_callback)

		# Loop until you hit something.
		while swing.game_loop():
			pass

		# Save score history.
		hist.append(swing.score)

		# Reset the state of the learner.
		learner.reset()
		learner.set_alpha(ii)
		learner.set_exploration(ii)
	return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games.
	run_games(agent, hist, 1000, 1)
	print sum(hist) / float(len(hist))
	print max(hist)
	# Save history.
	np.save('hist',np.array(hist))
