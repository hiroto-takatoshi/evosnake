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
resList = []

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
	#print(bd)
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

		#t = time.time()
		return ff(bd, self.food[0])

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
					#idx = bd.index([v[0], v[1]]) + 1
					#if idx <= mat[u] + 1: hitflag = True
					#print("you hit shit")
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
		
		'''
		for i in range(1,13):
			for j in range(1, 13):
				try:
					print("{:>3}".format(mat[(i,j)]), end=' ')
				except KeyError:
					print("{:>3}".format(0), end=' ')
			print()
		'''
		return ans, foodDist, len(mat) - 1


	def helperQuad(self, x, w1, w2, w3):
		return w1 * x * x + w2 * x + w3

	def getScore(self, w, vb=False):

		newBody = [self.ahead,]
		for x in self.body[:-1]:
			newBody.append(x)

		a, b, c, d, e, g, k = self.getReachableScore(newBody)
		#print("%3.3g %3.3g %3.3g %3.3g %3.3g" % (a, b, c, d, e))
		#time.sleep(1)

		f =  len(self.body)  / 144.0 

		"""
		#neural like topology
		n11 = nn([(a, w[0]), (b, w[1]), (c, w[2]), (d, w[3]), (e, w[4]), (f, w[5]),], w[6])
		n12 = nn([(a, w[7]), (b, w[8]), (c, w[9]), (d, w[10]), (e, w[11]), (f, w[12]),], w[13])
		n13 = nn([(a, w[14]), (b, w[15]), (c, w[16]), (d, w[17]), (e, w[18]), (f, w[19]),], w[20])
		n14 = nn([(a, w[21]), (b, w[22]), (c, w[23]), (d, w[24]), (e, w[25]), (f, w[26]),], w[27])
		n15 = nn([(a, w[28]), (b, w[29]), (c, w[30]), (d, w[31]), (e, w[32]), (f, w[33]),], w[34])
		n16 = nn([(a, w[35]), (b, w[36]), (c, w[37]), (d, w[38]), (e, w[39]), (f, w[40]),], w[41])
		n17 = nn([(a, w[42]), (b, w[43]), (c, w[44]), (d, w[45]), (e, w[46]), (f, w[47]),], w[48])
		n18 = nn([(a, w[49]), (b, w[50]), (c, w[51]), (d, w[52]), (e, w[53]), (f, w[54]),], w[55])
		
		n21 = nn([(n11, w[56]), (n12, w[57]), (n13, w[58]) , (n14, w[59]), 
			(n15, w[60]), (n16, w[61]), (n17, w[62]), (n18, w[63])], w[64])
		n22 = nn([(n11, w[65]), (n12, w[66]), (n13, w[67]) , (n14, w[68]), 
			(n15, w[69]), (n16, w[70]), (n17, w[71]), (n18, w[72])], w[73])
		n23 = nn([(n11, w[74]), (n12, w[75]), (n13, w[76]) , (n14, w[77]), 
			(n15, w[78]), (n16, w[79]), (n17, w[80]), (n18, w[81])], w[82])
		n24 = nn([(n11, w[83]), (n12, w[84]), (n13, w[85]) , (n14, w[86]), 
			(n15, w[87]), (n16, w[88]), (n17, w[89]), (n18, w[90])], w[91])
		n25 = nn([(n11, w[92]), (n12, w[93]), (n13, w[94]) , (n14, w[95]), 
			(n15, w[96]), (n16, w[97]), (n17, w[98]), (n18, w[99])], w[100])

		score = nn([(n21, w[101]), (n22, w[102]), (n23, w[103]) , (n24, w[104]), (n25, w[105])], w[106])
		"""

		part1 = self.helperQuad(f, 0, w[0], w[1]) * self.helperQuad(b, 0, w[2], w[3])
		#part1 = k
		part2 = self.helperQuad(f, 0, w[4], w[5]) * self.helperQuad(c, 0, w[6], w[7])
		part3 = self.helperQuad(f, 0, w[8], w[9]) * self.helperQuad(d, 0, w[10], w[11])
		part4 = self.helperQuad(f, 0, w[12], w[13]) * self.helperQuad(e, 0, w[14], w[15])
		#part5 = self.helperQuad(f, 0, w[16], w[17]) * self.helperQuad(e, 0, w[18], w[19])
		#part6 = self.helperQuad(f, 0, w[20], w[21]) * self.helperQuad(g, 0, w[22], w[23])
		#part6 = g
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
			#print(c)
			if c > cc:
				#if cc : print(cc, c)
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
	
	input("press me")

	#routine = gp.compile(individual, pset)

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
		win.addstr(19, 2, 'ScorA : ' + str("%4.3f" % (snake.aa)) + ' ')
		win.addstr(20, 2, 'part1 : ' + str("%4.3f" % (snake.aaa)) + ' ')
		win.addstr(21, 2, 'ScorB : ' + str("%4.3f" % (snake.bb)) + ' ')
		win.addstr(22, 2, 'part2 : ' + str("%4.3f" % (snake.bbb)) + ' ')
		win.addstr(23, 2, 'ScorC : ' + str("%4.3f" % (snake.cc)) + ' ')
		win.addstr(24, 2, 'part3 : ' + str("%4.3f" % (snake.ccc)) + ' ')
		win.addstr(25, 2, 'ScorD : ' + str("%4.3f" % (snake.dd)) + ' ')
		win.addstr(26, 2, 'part4 : ' + str("%4.3f" % (snake.ddd)) + ' ')
		
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
	#global snake
	global snakeList
	global snakeList1
	global phase
	

	totalScore = 0
	snake = None
	if not snakeList:
		snake = SnakePlayer()
	else:
		snake = copy.deepcopy(random.choice(snakeList))
	
	#assert(snake.score >= phase - 10)

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
				#snakeList1.append(copy.deepcopy(snake))
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
		#if x <= phase:
		#	return 0.0, 
		totScore.append(x)
	#assert(totScore / 2.0 >= phase)
	return np.max(totScore), np.min(totScore), np.average(totScore), totScore.count(133), 

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
	fitnesses = toolbox.map(toolbox.evaluate, population)
	#print(fitnesses)
	#assert False
	for ind, fit in zip(population, fitnesses):
		
		ind.fitness.values = fit
		#print(fit, ind.fitness.values)

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

		if record['avgbest'] > k:
			k = record['avgbest']
			kk = gen
		
		if gen - kk > 10:
			return population, logbook
		if verbose:
			print(logbook.stream)
		'''
		snakeList1[:] = []

		for x in range(100):
			if runGameHelper(population[x])[0] > 0.0:
				runGame(population[x], verb=True)
				if len(snakeList1) >= 10:
					succFlag = True
					snakeList[:] = snakeList1
					print("done", phase)
					return population, logbook
		print(len(snakeList1))
		if verbose:
			print(logbook.stream)
			#var = np.var(np.asarray(population))
			#print(var)
		'''

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
	#global snake
	global pset

	## THIS IS WHERE YOUR CORE EVOLUTIONARY ALGORITHM WILL GO #

	creator.create("FitnessMax", base.Fitness, weights=(1.0, 0.0, 1.0, 1.0))
	creator.create("Individual", list, fitness=creator.FitnessMax)

	toolbox = base.Toolbox()
	toolbox.register("attr_float", random.random)
	toolbox.register("individual", tools.initRepeat, creator.Individual,
					lambda : random.random() * 2 - 1, n=16)
	
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	
	toolbox.register("evaluate", runGameHelper)
	toolbox.register("mate", tools.cxUniform, indpb=0.1)
	toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.05)
	#toolbox.register("mutate", myMutate)
	#toolbox.register("select", tools.selTournament, tournsize=2)
	toolbox.register("select", tools.selNSGA2)
	
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avgbest", lambda x:np.average(x, axis=0)[0])
	stats.register("avgworst", lambda x: np.average(x, axis=0)[1] )
	stats.register("avgperform", lambda x:np.average(x, axis=0)[2])
	stats.register("avg133", lambda x:np.average(x, axis=0)[3])
	stats.register("maxbest", lambda x:np.amax(x, axis=0)[0])

	global thePool
	global phase
	global succFlag
	thePool = Pool(16)
	toolbox.register("map", thePool.map)

	'''
	bigList = []

	pop = toolbox.population(n=300)
	hof = tools.HallOfFame(2)
	cnt = 0
	while phase < 135:
		print(len(snakeList))
		#pop = toolbox.population(n=300)
		pop, log = eaMuPlusLambda1(pop, toolbox=toolbox, mu=100, lambda_=200, 
			cxpb=0.5, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=True)
		if not succFlag:
			cnt += 1
			pop = toolbox.population(n=300)
		else:
			phase += 1
			succFlag = False
			bigList += [pop]
			showSnake(random.choice(snakeList).body)
			print("")
		if cnt >= 3:
			print("Failed at challenging phase ", phase)
			break
	
	prev = None
	for x in bigList:
		now = np.mean(x, axis=0)
		print(now)
		if prev is not None:
			print(np.linalg.norm(now - prev))
		prev = now


	return
	'''
	
	#'''
	lastPop = []

	for i in range(30):

		pop = toolbox.population(n=300)
		hof = tools.ParetoFront()

		pop, log = algorithms.eaMuPlusLambda(pop, toolbox=toolbox, mu=100, lambda_=200, 
			cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof, verbose=True)
		
		#lastPop += [x for x in hof]

		#print(len(hof))

		dt = log.select("avgbest")

		np.savetxt('sol/'+str(i)+'.csv', dt, delimiter=',')
		print("Deme ", i, "ended")

	return

	hof = tools.HallOfFame(2)
	#print(bigList)
	lastPop, log = eaMuPlusLambda1(lastPop, toolbox=toolbox, mu=30, lambda_=60, 
		cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof, verbose=True)
	
	for i in range(3):
		displayStrategyRun(hof[0])
	print(runGameHelper(hof[0]))
	#'''
	return

	toolbox.register("migrate", tools.migRing, k=5, selection=tools.selBest)
	toolbox.register("variaton", algorithms.varAnd, toolbox=toolbox, cxpb=0.7, mutpb=0.3)

	
	NBR_DEMES = 1
	MU = 30
	LAMBDA = 60
	NGEN = 50
	CXPB = 0.5
	MUTPB = 0.2
	MIG_RATE = 5 
	
	demes = [toolbox.population(n=MU) for _ in range(NBR_DEMES)]
	hof = tools.HallOfFame(1)
	logbook = tools.Logbook()
	logbook.header = ['gen', ] + (stats.fields if stats else [])

	for idx, deme in enumerate(demes):
		fitnesses = toolbox.map(toolbox.evaluate, deme)
		for ind, fit in zip(deme, fitnesses):
			ind.fitness.values = fit
		
		hof.update(deme)
	
	record = demes[0]
	for deme in demes[1:]:
		record += deme
	record = stats.compile(record)
	logbook.record(gen=0, **record)
	print(logbook.stream)

	gen = 1

	while gen <= NGEN:

		for deme in demes:
			deme = algorithms.varAnd(deme, toolbox=toolbox, cxpb=0.5, mutpb=0.2)
		for idx, deme in enumerate(demes):

			offspring = algorithms.varOr(deme, toolbox, LAMBDA, CXPB, MUTPB)

			fitnesses = toolbox.map(toolbox.evaluate, offspring)
			for ind, fit in zip(offspring, fitnesses):
				ind.fitness.values = fit
			
			hof.update(offspring)

			demes[idx] = toolbox.select(deme + offspring, MU)
			

		if gen % MIG_RATE == 0:
			toolbox.migrate(demes)
		
		record = demes[0]
		for deme in demes[1:]:
			record += deme
		record = stats.compile(record)
		logbook.record(gen=gen, **record)
		print(logbook.stream)

		gen += 1



	for i in range(2):
	
		print(runGameHelper(hof[i]))

def debug():
	goodgene = [-0.02489863,-0.30308565,-0.52308796,0.29551958,-0.48911588,0.67214919, 
		-0.61397186,-1.09052873,-0.33509962,0.16349206,0.84844836,-0.35240072, 
		0.38884796,0.62850993,0.44859198,0.38609659,-0.03084196,-0.26513208,0.05987635, 
		0.74116524]
	
	displayStrategyRun(goodgene)

	sys.exit(0)

if __name__ == "__main__":
	#debug()
	
	#monitoring_thread = start_monitoring()
	main()

