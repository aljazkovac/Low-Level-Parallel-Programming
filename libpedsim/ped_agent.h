//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// TAgent represents an agent in the scenario. Each
// agent has a position (x,y) and a number of destinations
// it wants to visit (waypoints). The desired next position
// represents the position it would like to visit next as it
// will bring it closer to its destination.
// Note: the agent will not move by itself, but the movement
// is handled in ped_model.cpp. 
//

#ifndef _ped_agent_h_
#define _ped_agent_h_ 1

#include <vector>
#include <deque>
#include <cmath>

using namespace std;

namespace Ped {
	class Twaypoint;

	class Tagent {
	public:
		Tagent(int posX, int posY);
		Tagent(double posX, double posY);

		// Reallocate coordinates in Memory
		void reallocate_coordinates(int* newX, int* newY);

		// Returns the coordinates of the desired position
		int getDesiredX() const { return desiredPositionX; }
		int getDesiredY() const { return desiredPositionY; }

	        int desiredPositionX;
		int desiredPositionY;

		// Sets the agent's position
		void setX(int newX) { *x = newX; }
		void setY(int newY) { *y = newY; }

		// Update the position according to get closer
		// to the current destination
		void computeNextDesiredPosition();

		// Position of agent defined by x and y
		int getX() const { return round(*x); };
		int getY() const { return round(*y); };

		// Adds a new waypoint to reach for this agent
		void addWaypoint(Twaypoint* wp);

		Twaypoint* getNextDestination();

		Twaypoint* getNextDestinationSpecial();
		
		// The current destination (may require several steps to reach)
		Twaypoint* destination;	

		Twaypoint* getDest() const { return destination; }	  
		void setDest(Twaypoint* dest) { destination = dest; }
 	
		deque<Twaypoint*> getWaypoints() const {return waypoints;}

		bool operator < (const Ped::Tagent& agent) const {
			return (*x < agent.getX());
		}
		

	private:
		Tagent() {};

		// The agent's current position
		int* x;
		int* y;

		// The agent's desired next position
		

		// The last destination
		Twaypoint* lastDestination;

		// The queue of all destinations that this agent still has to visit
		deque<Twaypoint*> waypoints;

		// Internal init function 
		void init(int posX, int posY);

		// Returns the next destination to visit
		
	};
}

#endif
