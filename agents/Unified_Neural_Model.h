
#ifndef UNIFIED_NEURAL_MODEL_H
#define UNIFIED_NEURAL_MODEL_H

#include"Reinforcement_Agent.h"
#include"../environments/Reinforcement_Environment.h"
#include"stdlib.h"
#include"stdio.h"
#include"time.h"
#include"useful/useful_utils.h"
#include"random/State_of_Art_Random.h"
#include"modules/Module.h"
#include"../parameters.h"
#include"self_organized_systems/Novelty_Map.h"
#include "array"

//#include"unistd.h"
//#include"sys/wait.h"
//subpopulations' size
//#define NEURON_EFFICIENT_SUBPOPULATION_SIZE 10
//#define CONNECTION_EFFICIENT_SUBPOPULATION_SIZE 10
//#define FITNESS_SUBPOPULATION_SIZE 10
//#define NEURON_RICH_SUBPOPULATION_SIZE 10
//#define CONNECTION_RICH_SUBPOPULATION_SIZE 10

typedef struct _nmap_cell
{
	Module* module;
	double fitness;
}nmap_cell;


class Unified_Neural_Model : public Reinforcement_Agent
{
	public:

		Unified_Neural_Model(Random* random);
		~Unified_Neural_Model();

		//All Reinforcement Agents have the following commented variables, although it is not declared here!
		//double* action;
		//int number_of_observation_vars;
		//int number_of_action_vars;
		
		Random* random;

		Module*** subpopulation;
		Module*** tmp_subpopulation;
		double** fitness;
		double** tmp_fitness;
		int testing_individual; // The current pop/nmapcell to give to the thread who asks.
		int testing_subpop; // The current individual to give to the thread who asks.
		int testing_individual_done; // to know what pop/nmapcell that is tested.
		int testing_subpop_done; // to know what individual that is tested.
		int best_index;
		int selected_individuals[NUMBER_OF_SUBPOPULATIONS][SUBPOPULATION_SIZE][2];	
		int generation;
		double step_counter;
		bool wait_for_sync;
		int curr_generation_trial;
		bool end_training;
		std::array<double*, NUMBER_OF_THREADS> SUNA_action;
		std::array<double, NUMBER_OF_THREADS> accumulated_rewards_per_thread;
		std::array<bool, NUMBER_OF_THREADS> rewards_initialized;

		//int curr_get_pop; // The current pop/nmapcell to give to the thread who asks.
		//int curr_get_individual; // The current individual to give to the thread who asks.
#ifdef	SPECTRUM_DIVERSITY
		Novelty_Map* nmap;
#endif
		//auxiliary
		void printBest();
		void evolve();
		void supremacistEvolve();
		void spectrumDiversityEvolve();
		double subpopulationObjective(Module* module, double fitness, int subpopulation_index);
		void findBestIndividual();
		void tryToInsertInSubpopulation(int subpopulation_index, int individual_index, int inserting_subpop);
		void calculateSpectrum(double* spectrum, int subpopulation_index, int individual_index);
		void endBestEpisode();

		//Implementing the Reinforcement Agent Interface
		void init(int number_of_observation_vars, int number_of_action_var);
		void step(double* observation, double reward);
		void step(int species, int individual, double* observation, double reward, int thread_id);
		std::array<int, 2> getNextIndividual();
		void print();
		double stepBestAction(double* observation);
		void endEpisode(double reward);
		void endEpisode(int species, int individual, double reward);
		void saveAgent(const char* filename);
		void loadAgent(const char* filename);
		void updateReward(double reward, int thread_id);
		
		//debug
		void printSubpop();

};

#endif
