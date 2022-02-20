//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_

#include <vector>
#include <map>
#include <set>

#include "ped_agent.h"
#include <atomic>

// Thread function
namespace Ped{
  class Tagent;

  // The implementation modes for Assignment 1 + 2:
  // chooses which implementation to use for tick()
  enum IMPLEMENTATION { CUDA, VECTOR, OMP, PTHREAD, CTHREADS, SEQ, SIMD };

  class Model
  {
  public:
    // -------------- A3 ------------------------------------------------
    void populate_regions(int x0, int x1, int x2, int x3, int x4);
    void recalculate_regions(int x0, int x1, int x2, int x3, int x4);

    void populate_dynamic_regions();
    void repopulate_dynamic_regions();
    void create_two_boundaries(int count, int xBound, int boundary_counter);
    // ------------------------------------------------------------------
    // Sets everything up
    void setup(std::vector<Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario,IMPLEMENTATION implementation, int number_of_threads = 2);		
	
    // Coordinates a time step in the scenario: move all agents by one step (if applicable).
    void tick();

    // Returns the agents of this scenario
    const std::vector<Tagent*> getAgents() const { return agents; };

    // Adds an agent to the tree structure
    void placeAgent(const Ped::Tagent *a);

    // Cleans up the tree and restructures it. Worth calling every now and then.
    void cleanup();
    ~Model();

    // Returns the heatmap visualizing the density of agents
    int const * const * getHeatmap() const { return blurred_heatmap; };
    int getHeatmapSize() const;

  private:

    // Denotes which implementation (sequential, parallel implementations..)
    // should be used for calculating the desired positions of
    // agents (Assignment 1)
    IMPLEMENTATION implementation;

    // Denotes the number of threads to use in PTHREADS modes
    int number_of_threads;

    // Arrays
    int *xArray;
    int *yArray;
    
    float *destXarray;
    float *destYarray;
    float *destRarray;
    
    int *destReached;

    // Determine the region coordinates (4 regions)
    // I am basing this on the max coordinates I have seen in the 
    // hugeScenario
    int x0;
    int x1;
    int x2;
    int x3;
    int x4;	

    // The agents in this scenario
    std::vector<Tagent*> agents;

    // The waypoints in this scenario
    std::vector<Twaypoint*> destinations;
		
    // Moves an agent towards its next position
    void move(Ped::Tagent *agent);
    
    // Moves an agent towards its next destination in an atomic way
    void move_atomic(Ped::Tagent *agent);

    // The plane for Assignment 3
    std::vector<std::vector<Ped::Tagent*>> plane;
    std::vector<std::tuple<int, int>> xBounds;
    std::vector<std::vector<int>> boundaries;

    //--------------- CUDA -----------------
    int NUM_BLOCKS;
    int THREADS_PER_BLOCK;

    ////////////
    /// Everything below here won't be relevant until Assignment 3
    ///////////////////////////////////////////////

    // Returns the set of neighboring agents for the specified position
    set<const Ped::Tagent*> getNeighbors(int x, int y, int dist) const;

    


    ////////////
    /// Everything below here won't be relevant until Assignment 4
    ///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE*CELLSIZE

    // The heatmap representing the density of agents
    int ** heatmap;

    // The scaled heatmap that fits to the view
    int ** scaled_heatmap;

    // The final heatmap: blurred and scaled to fit the view
    int ** blurred_heatmap;

    void setupHeatmapSeq();
    void updateHeatmapSeq();
  };
}
#endif
