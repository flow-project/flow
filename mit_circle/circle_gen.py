from __future__ import division
import math
from math import pi
n = 22
r = 300/(2*pi)
points = [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in xrange(0,n+1)]

for i, point in enumerate(points[:-1]):
	print("<node id=\"n" + str(i) + "\" x=\"" + str(round(point[0], 2))  + "\" y=\"" + str(round(point[1], 2)) + "\"/>")

for i in range(n-1):
	print("<edge from=\"n"+ str(i) + "\" id=\"e"+str(i)+"\" to=\"n"+ str(i+1) + "\" type=\"edgeType\"/>")

print("<edge from=\"n"+ str(n-1) + "\" id=\"e"+str(n-1)+"\" to=\"n"+ str(0) + "\" type=\"edgeType\"/>")


def getRouteNodes(i):
	route = ""
	for j in range(n):
		route += "e" + str((i + j) % n) + " "
	return route[:-1]

for i in range(n-1):
	print("<route id=\"r"+ str(i) + "\" edges=\"" + getRouteNodes(i) + "\"/>")

for i in range(n):

	print("<vehicle id=\""+ str(i) +"\" type=\"car\" route=\"r"+str(i)+"\" depart=\"0\" color=\""+ str(1) +"," + str(round(i/(n-1), 2))+","+ str(round(i/(n-1), 2))+ "\"/>")