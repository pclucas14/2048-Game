# 2048 Game written using the Pygame module
# 
# Lewis Deane
# 23/12/2014

import pygame, sys, time
import os.path
from pygame.locals import *
from colours import *
import time
from random import *

TOTAL_POINTS = 0
DEFAULT_SCORE = 2
BOARD_SIZE = 4

pygame.init()

SURFACE = pygame.display.set_mode((600, 700), 0, 32)
pygame.display.set_caption("2048")

myfont = pygame.font.SysFont("monospace", 25)
scorefont = pygame.font.SysFont("monospace", 50)

'''
helper methods 
'''
def floor(n):
	return int(n - (n % 1))

class Board: 
	def __init__(self):
		self.tileMatrix =  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
		self.undoMat = []

	def init_game(self, gui=False):
		self.reset()
		self.placeRandomTile()
		self.placeRandomTile()
		
		if gui : self.printMatrix()
		'''
		while True:
			if self.checkIfCanGo() == True:
				rotations = randint(0,3)
				print ('moved')
				self.play_move(rotations)
				if gui : self.printMatrix()

			else:
				self.printGameOver()
				print 'over'
				exit()#break

			pygame.display.update()
			if gui : self.printMatrix()
		'''
	
	# renaming a method without breaking everything. 
	def game_over(self):
		return not self.checkIfCanGo()

	# TODO : this method should return the move's reward
	def play_move(self, rotations):
		self.addToUndo()
		score = 0
		valid_move = False
		for i in range(0, rotations):
			self.rotateMatrixClockwise()

		if self.canMove():
			self.moveTiles()
			score = self.mergeTiles()
			self.placeRandomTile()
			valid_move = True

		for j in range(0, (4 - rotations) % 4):
			self.rotateMatrixClockwise()
		return score if valid_move else -1 


	def printMatrix(self):
		SURFACE.fill(BLACK)

		global BOARD_SIZE
		global TOTAL_POINTS

		for i in range(0, BOARD_SIZE):
			for j in range(0, BOARD_SIZE):
				pygame.draw.rect(SURFACE, getColour(self.tileMatrix[i][j]), (i*(600/BOARD_SIZE), j*(600/BOARD_SIZE) + 100, 600/BOARD_SIZE, 600/BOARD_SIZE))
				
				label = myfont.render(str(self.tileMatrix[i][j]), 1, WHITE)
				label2 = scorefont.render("Score: " + str(TOTAL_POINTS), 1, WHITE)

				offset = 0

				if self.tileMatrix[i][j] < 10:
					offset = -10
				elif self.tileMatrix[i][j] < 100:
					offset = -15
				elif self.tileMatrix[i][j] < 1000:
					offset = -20
				else:
					offset = -25

				if self.tileMatrix[i][j] > 0:
					SURFACE.blit(label, (i*(600/BOARD_SIZE) + (300/BOARD_SIZE) +offset, j*(600/BOARD_SIZE) + 100 + 300/BOARD_SIZE - 15))
				SURFACE.blit(label2, (10, 20))

	def printGameOver(self):
		global TOTAL_POINTS

		SURFACE.fill(BLACK)

		label = scorefont.render("Game Over!", 1, (255,255,255))
		label2 = scorefont.render("Score: " + str(TOTAL_POINTS), 1, (255,255,255))
		label3 = myfont.render("Press r to restart!", 1, (255,255,255))

		SURFACE.blit(label, (150, 100))
		SURFACE.blit(label2, (150, 300))
		SURFACE.blit(label3, (150, 500))

	def placeRandomTile(self):
		count = 0
		for i in range(0, BOARD_SIZE):
			for j in range(0, BOARD_SIZE):
				if self.tileMatrix[i][j] == 0:
					count += 1

		k = floor(random() * BOARD_SIZE * BOARD_SIZE)

		while self.tileMatrix[floor(k / BOARD_SIZE)][k % BOARD_SIZE] != 0:
			k = floor(random() * BOARD_SIZE * BOARD_SIZE)

		self.tileMatrix[floor(k / BOARD_SIZE)][k % BOARD_SIZE] = 2

	def moveTiles(self):
		# We want to work column by column shifting up each element in turn.
		for i in range(0, BOARD_SIZE): # Work through our 4 columns.
			for j in range(0, BOARD_SIZE - 1): # Now consider shifting up each element by checking top 3 elements if 0.
				while self.tileMatrix[i][j] == 0 and sum(self.tileMatrix[i][j:]) > 0: # If any element is 0 and there is a number to shift we want to shift up elements below.
					for k in range(j, BOARD_SIZE - 1): # Move up elements below.
						self.tileMatrix[i][k] = self.tileMatrix[i][k + 1] # Move up each element one.
					self.tileMatrix[i][BOARD_SIZE - 1] = 0

	def mergeTiles(self):
		global TOTAL_POINTS
		previous_total = TOTAL_POINTS
		for i in range(0, BOARD_SIZE):
			for k in range(0, BOARD_SIZE - 1):
					if self.tileMatrix[i][k] == self.tileMatrix[i][k + 1] and self.tileMatrix[i][k] != 0:
						self.tileMatrix[i][k] = self.tileMatrix[i][k] * 2
						self.tileMatrix[i][k + 1] = 0
						TOTAL_POINTS += self.tileMatrix[i][k]
						self.moveTiles()
		return TOTAL_POINTS - previous_total

	def checkIfCanGo(self):
		for i in range(0, BOARD_SIZE ** 2):
			if self.tileMatrix[floor(i / BOARD_SIZE)][i % BOARD_SIZE] == 0:
				return True

		for i in range(0, BOARD_SIZE):
			for j in range(0, BOARD_SIZE - 1):
				if self.tileMatrix[i][j] == self.tileMatrix[i][j + 1]:
					return True
				elif self.tileMatrix[j][i] == self.tileMatrix[j + 1][i]:
					return True
		return False

	def reset(self):
		global TOTAL_POINTS
		global tileMatrix

		TOTAL_POINTS = 0
		SURFACE.fill(BLACK)

		self.tileMatrix = [[0 for i in range(0, BOARD_SIZE)] for j in range(0, BOARD_SIZE)]

		#self.play_game()

	def canMove(self):
		for i in range(0, BOARD_SIZE):
			for j in range(1, BOARD_SIZE):
				if self.tileMatrix[i][j-1] == 0 and self.tileMatrix[i][j] > 0:
					return True
				elif (self.tileMatrix[i][j-1] == self.tileMatrix[i][j]) and self.tileMatrix[i][j-1] != 0:
					return True

		return False

	def saveGameState():
		f = open("savedata", "w")

		tiles = " ".join([str(tileMatrix[floor(x / BOARD_SIZE)][x % BOARD_SIZE]) for x in range(0, BOARD_SIZE**2)])
		
		f.write(str(BOARD_SIZE)  + "\n")
		f.write(tiles + "\n")
		f.write(str(TOTAL_POINTS))
		f.close()

	def loadGameState():
		if os.path.isfile("savedata"):
			global TOTAL_POINTS
			global BOARD_SIZE
			global tileMatrix
			f = open("savedata", "r")

			BOARD_SIZE = int(f.readline())
			mat = (f.readline()).split(' ', BOARD_SIZE ** 2)
			TOTAL_POINTS = int(f.readline())

			tileMatrix = [[0 for i in range(0, BOARD_SIZE)] for j in range(0, BOARD_SIZE)]

			for i in range(0, BOARD_SIZE ** 2):
				tileMatrix[floor(i / BOARD_SIZE)][i % BOARD_SIZE] = int(mat[i])

			f.close()

			play_game(True)

	def rotateMatrixClockwise(self):
		for i in range(0, int(BOARD_SIZE/2)):
			for k in range(i, BOARD_SIZE- i - 1):
				temp1 = self.tileMatrix[i][k]
				temp2 = self.tileMatrix[BOARD_SIZE - 1 - k][i]
				temp3 = self.tileMatrix[BOARD_SIZE - 1 - i][BOARD_SIZE - 1 - k]
				temp4 = self.tileMatrix[k][BOARD_SIZE - 1 - i]

				self.tileMatrix[BOARD_SIZE - 1 - k][i] = temp1
				self.tileMatrix[BOARD_SIZE - 1 - i][BOARD_SIZE - 1 - k] = temp2
				self.tileMatrix[k][BOARD_SIZE - 1 - i] = temp3
				self.tileMatrix[i][k] = temp4

	def isArrow(k):
		return(k == pygame.K_UP or k == pygame.K_DOWN or k == pygame.K_LEFT or k == pygame.K_RIGHT)

	def getRotations(k):
		if k == pygame.K_UP:
			return 0
		elif k == pygame.K_DOWN:
			return 2
		elif k == pygame.K_LEFT:
			return 1
		elif k == pygame.K_RIGHT:
			return 3
			
	def convertToLinearMatrix(self):
		mat = []

		for i in range(0, BOARD_SIZE ** 2):
			mat.append(self.tileMatrix[floor(i / BOARD_SIZE)][i % BOARD_SIZE])

		mat.append(TOTAL_POINTS)

		return mat

	def addToUndo(self):
		self.undoMat.append(self.convertToLinearMatrix())

	def undo():
		if len(self.undoMat) > 0:
			mat = self.undoMat.pop()

			for i in range(0, BOARD_SIZE ** 2):
				tileMatrix[floor(i / BOARD_SIZE)][i % BOARD_SIZE] = mat[i]

			global TOTAL_POINTS
			TOTAL_POINTS = mat[BOARD_SIZE ** 2]

			printMatrix()

	#play_game()