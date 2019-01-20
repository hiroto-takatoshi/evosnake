# This code defines the agent (as in the playable version) in a way that can be called and executed from an evolutionary algorithm. The code is partial and will not execute. You need to add to the code to create an evolutionary algorithm that evolves and executes a snake agent.
import curses
import random
import operator
from functools import partial

import math
import time
import os, sys
import copy
import uuid

import numpy as np
from deap import algorithms, base, creator, tools
import matplotlib.pyplot as plt

from multiprocessing import Pool

from hanging_threads import start_monitoring

from ctypes import *

S_UP, S_RIGHT, S_DOWN, S_LEFT = 0,1,2,3
XSIZE,YSIZE = 14,14
NFOOD = 1 # NOTE: YOU MAY NEED TO ADD A CHECK THAT THERE ARE ENOUGH SPACES LEFT FOR THE FOOD (IF THE TAIL IS VERY LONG)

snakeList = []
snakeList1 = []
phase = 500
succFlag = False

def randstr():
	return uuid.uuid4().hex

class PyStruct(Structure):
	_fields_ = [
		("a", c_int),
		("b", c_int),
		("c", c_int),
		("k", c_int),
		("mat", (c_int * 15) * 15),
		("q", (c_int * 2) * 200),
		("fx", c_int),
		("fy", c_int),
		("d", c_float),
		("e", c_float)
	]

def getDist(a, b):
	return np.linalg.norm(np.asarray(a) - np.asarray(b))
# faster interface of floodfill function
def ff(bd, fd):

	so = CDLL(os.path.abspath("floodfill.so"))
	so.floodfill.argtypes = [POINTER(PyStruct)]
	so.floodfill.restype = None

	ps = PyStruct()

	for x in bd[1:]:
		ps.mat[x[0]][x[1]] = -1
	ps.q[1][0] = bd[0][0]
	ps.q[1][1] = bd[0][1]
	ps.fx = fd[0]
	ps.fy = fd[1]

	so.floodfill(byref(ps))
	reta = 1
	if ps.c - 2 != 144 - len(bd):
		reta = -9000 - (144 - len(bd)) + (ps.c - 2)

	g = 0
	xx = bd[-1][0]
	yy = bd[-1][1]
	g = max(ps.mat[xx+1][yy], g)
	g = max(ps.mat[xx-1][yy], g)
	g = max(ps.mat[xx][yy+1], g)
	g = max(ps.mat[xx][yy-1], g)

	if g == 0:
		g = -200
	else : g = 0

	return reta, ps.b , ps.d, ps.a, ps.e, g, (ps.k-1) * (-200)

def nn(l, b):
	ret = b
	for x in l:
		ret += x[0] * x[1]
	return ret

# This class can be used to create a basic player object (snake agent)
class SnakePlayer(list):
	global S_RIGHT, S_LEFT, S_UP, S_DOWN
	global XSIZE, YSIZE

	def __init__(self):
		self.direction = S_RIGHT
		self.body = [ [4,10], [4,9], [4,8], [4,7], [4,6], [4,5], [4,4], [4,3], [4,2], [4,1],[4,0] ]
		
		self.score = 0
		self.ahead = []
		self.food = []
		
		# debug var
		self.aaa = 0.0
		self.bbb = 0.0
		self.ccc = 0.0
		self.ddd = 0.0
		self.aa = 0
		self.bb = 0
		self.cc = 0
		self.dd = 0

	def _reset(self):
		self.direction = S_RIGHT
		self.body[:] = [ [4,10], [4,9], [4,8], [4,7], [4,6], [4,5], [4,4], [4,3], [4,2], [4,1],[4,0] ]
		self.score = 0
		self.ahead = []
		self.food = []

	def getAheadLocation(self, dir):
		self.ahead = [ self.body[0][0] + (dir == S_DOWN and 1) + (dir == S_UP and -1), self.body[0][1] + (dir == S_LEFT and -1) + (dir == S_RIGHT and 1)] 

	def updatePosition(self):
		self.getAheadLocation(self.direction)
		self.body.insert(0, self.ahead )

	## You are free to define more sensing options to the snake

	def getReachableScore(self, bd):
		
		dir = [[1, 0], [-1, 0], [0, 1], [0, -1],]
		mat = {(bd[0][0], bd[0][1]) : 0, }
		q = [(bd[0][0], bd[0][1]), ]

		ans = 0

		while q:
			u = q.pop(0)
			ans = max(ans, mat[u])
			for i in range(4):
				v = (u[0] + dir[i][0], u[1] + dir[i][1])
				hitFlag = False
				if v[0] <= 0 or v[0] >= (YSIZE-1) or v[1] <= 0 or v[1] >= (XSIZE-1): hitFlag = True
				
				if [v[0], v[1]] in bd: 
					
					hitFlag = True
				if hitFlag : continue
				if v in mat: 
					if mat[v] > mat[u] + 1: print("happened")
					mat[v] = min(mat[v], mat[u] + 1)
					
					continue
				mat[v] = mat[u] + 1
				#print(v)
				q.append(v)

		foodDist = mat.get((self.food[0][0], self.food[0][1]), 200)
		
		
		return 1, 1 , 1, 1, 1, 1, 1
		


	def helperQuad(self, x, w1, w2, w3):
		return w1 * x * x + w2 * x + w3

	def getScore(self, w, vb=False):

		newBody = [self.ahead,]
		for x in self.body[:-1]:
			newBody.append(x)

		a, b, c, d, e, g, k = self.getReachableScore(newBody)
		

		f =  len(self.body)  / 144.0 


		part1 = self.helperQuad(f, 0, w[0], w[1]) * self.helperQuad(b, 0, w[2], w[3])
		
		part2 = self.helperQuad(f, 0, w[4], w[5]) * self.helperQuad(c, 0, w[6], w[7])
		part3 = self.helperQuad(f, 0, w[8], w[9]) * self.helperQuad(d, 0, w[10], w[11])
		part4 = self.helperQuad(f, 0, w[12], w[13]) * self.helperQuad(e, 0, w[14], w[15])
	
		part5 = g
		part6 = k


		score = part1 + part2 + part3 + part4 + part5 + part6

		return score, k

	def updateDirection(self, w, verb=False):
		bestScore = -1e100
		bestDir = self.direction
		cc = -10000
		
		for i in range(4):
			if i == self.reverseDirection():
				continue
			self.getAheadLocation(i)
			if self.sense_tail_ahead(): continue
			if self.sense_wall_ahead(): continue
			newBody = [self.ahead,]
			newBody.append(self.body[:-1])
			score, c = self.getScore(w)
			
			if c > cc:
				
				cc = c
				
				bestScore = score
				bestDir = i	
			elif c == cc and score > bestScore :
				bestScore = score
				bestDir = i			
		
		self.direction = bestDir
		self.getAheadLocation(bestDir)
		self.getScore(w, verb)
		


	def reverseDirection(self):
		return (self.direction + 2) % 4

	def snakeHasCollided(self):
		self.hit = False
		if self.body[0][0] == 0 or self.body[0][0] == (YSIZE-1) or self.body[0][1] == 0 or self.body[0][1] == (XSIZE-1): self.hit = True
		if self.body[0] in self.body[1:]: self.hit = True
		return( self.hit )

	def sense_wall_ahead(self):
		return( self.ahead[0] == 0 or self.ahead[0] == (YSIZE-1) or self.ahead[1] == 0 or self.ahead[1] == (XSIZE-1) )

	def sense_food_ahead(self):
		self.getAheadLocation(self.direction)
		return self.ahead in self.food

	def sense_tail_ahead(self):
		return self.ahead in self.body

# This function places a food item in the environment
def placeFood(snake):
	food = []
	st = time.time()
	while len(food) < NFOOD:
		if time.time() - st > 10: assert False, str(len(snake.body))
		potentialfood = [random.randint(1, (YSIZE-2)), random.randint(1, (XSIZE-2))]
		if not (potentialfood in snake.body) and not (potentialfood in food):
			food.append(potentialfood)
	snake.food = food  # let the snake know where the food is
	return( food )



# This outline function is the same as runGame (see below). However,
# it displays the game graphically and thus runs slower
# This function is designed for you to be able to view and assess
# your strategies, rather than use during the course of evolution
def displayStrategyRun(individual):

	curses.initscr()
	win = curses.newwin(YSIZE + 15, XSIZE + 15, 0, 0)
	win.keypad(1)
	curses.noecho()
	curses.curs_set(0)
	win.border(0)
	win.nodelay(1)
	win.timeout(60)

	snake = None

	if len(snakeList) > 0:
		snake = copy.deepcopy(random.choice(snakeList))
	else:
		snake = SnakePlayer()
	food = placeFood(snake)

	for f in food:
		win.addch(f[0], f[1], '@')

	timer = 0
	collided = False
	while not collided and not timer == ((2*XSIZE) * YSIZE):

		# Set up the display
		win.border(0)
		win.addstr(0, 2, 'Score : ' + str(snake.score) + ' ')		
		win.getch()

		## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##

		snake.updateDirection(individual, True)
		snake.updatePosition()

		if snake.body[0] in food:
			snake.score += 1
			for f in food: win.addch(f[0], f[1], ' ')
			food = placeFood(snake)
			for f in food: win.addch(f[0], f[1], '@')
			timer = 0
		else:	
			last = snake.body.pop()
			win.addch(last[0], last[1], ' ')
			timer += 1 # timesteps since last eaten
		win.addch(snake.body[0][0], snake.body[0][1], 'o')

		collided = snake.snakeHasCollided()
		hitBounds = (timer == ((2*XSIZE) * YSIZE))

	curses.endwin()

	print (collided)
	print (hitBounds)
	input("Press to continue...")

	return snake.score,


# This outline function provides partial code for running the game with an evolved agent
# There is no graphical output, and it runs rapidly, making it ideal for
# you need to modify it for running your agents through the game for evaluation
# which will depend on what type of EA you have used, etc.
# Feel free to make any necessary modifications to this section.
def runGame(individual, verb=False):
	global snakeList
	global snakeList1
	global phase
	

	totalScore = 0
	snake = None
	if not snakeList:
		snake = SnakePlayer()
	else:
		snake = copy.deepcopy(random.choice(snakeList))
	
	food = placeFood(snake)
	timer = 0

	st = time.time()

	while not snake.snakeHasCollided() and not timer == XSIZE * YSIZE and len(snake.body) < 144:

		## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##

		if time.time() - st > 10: assert False

		snake.updateDirection(individual, verb)
		snake.updatePosition()

		if snake.body[0] in food:
			snake.score += 1
			if len(snake.body) == 144:
				return snake.score
			food = placeFood(snake)
			
			timer = 0
		else:	
			snake.body.pop()
			timer += 1 # timesteps since last eaten

		totalScore = snake.score
	#print(totalScore)
	
	return totalScore

def runGameHelper(w):
	global phase
	totScore = []
	for i in range(5):
		x = runGame(w)
		
		totScore.append(x)
	return np.max(totScore), 

def eaMuPlusLambda1(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
				   stats=None, halloffame=None, verbose=__debug__):
	
	global phase
	global snakeList
	global snakeList1
	global succFlag

	succFlag = False
  
	logbook = tools.Logbook()
	logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

	# Evaluate the individuals with an invalid fitness
	#invalid_ind = [ind for ind in population if not ind.fitness.valid]
	t = time.time()
	fitnesses = toolbox.map(toolbox.evaluate, population)
	
	for ind, fit in zip(population, fitnesses):
		ind.fitness.values = fit
	print(time.time() - t)
	assert False	
	if halloffame is not None:
		halloffame.update(population)

	record = stats.compile(population) if stats is not None else {}
	logbook.record(gen=0, nevals=len(population), hof=0.0, besthofgen=0, **record)
	if verbose:
		print(logbook.stream)
		#print(record)

	k = 0.0
	kk = 0

	snakeList1[:] = []

	# Begin the generational process
	for gen in range(1, ngen + 1):
		# Vary the population
		offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

		# Evaluate the individuals with an invalid fitness
		#invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = toolbox.map(toolbox.evaluate, offspring)
		for ind, fit in zip(offspring, fitnesses):
			ind.fitness.values = fit

		# Update the hall of fame with the generated individuals
		if halloffame is not None:
			halloffame.update(offspring)

		# Select the next generation population
		population[:] = toolbox.select(population + offspring, mu)

		# Update the statistics with the new population
		record = stats.compile(population) if stats is not None else {}
		logbook.record(gen=gen, nevals=len([]), hof=k, besthofgen=kk, **record)

		if record['avg'] > k:
			k = record['avg']
			kk = gen
		
		if gen - kk > 10:
			return population, logbook
		if verbose:
			print(logbook.stream)

	return population, logbook

def showSnake(bd):
	for i in range(1, 13):
		for j in range(1, 13):
			if [i, j] in bd:
				print("o", end='')
			else:
				print(" ", end='')
		print("")

def main():


	## THIS IS WHERE YOUR CORE EVOLUTIONARY ALGORITHM WILL GO #

	creator.create("FitnessMax", base.Fitness, weights=(1.0,))
	creator.create("Individual", list, fitness=creator.FitnessMax)

	toolbox = base.Toolbox()
	toolbox.register("attr_float", random.random)
	toolbox.register("individual", tools.initRepeat, creator.Individual,
					lambda : random.random() * 2 - 1, n=16)
	
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	
	toolbox.register("evaluate", runGameHelper)
	toolbox.register("mate", tools.cxUniform, indpb=0.1)
	toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.05)
	toolbox.register("select", tools.selTournament, tournsize=2)
	
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.average)
	stats.register("std", np.std)
	stats.register("median", np.median)
	stats.register("max", np.max)

	global thePool
	global phase
	global succFlag
	thePool = Pool(16)
	lastPop = []

	for i in range(30):

		pop = toolbox.population(n=30)
		hof = tools.HallOfFame(3)

		pop, log = eaMuPlusLambda1(pop, toolbox=toolbox, mu=100, lambda_=200, 
			cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof, verbose=True)
		
		

		dt = log.select("avg")

		np.savetxt('single/'+str(i)+'.csv', dt, delimiter=',')

		print("Deme ", i, "ended,")

	return




if __name__ == "__main__":
	
	main()
