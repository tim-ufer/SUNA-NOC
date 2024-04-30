
#include"stdio.h"
#include"stdlib.h"
#include "thread"
#include "vector"
#include "array"
#include <mutex>

//agents
#include"agents/Unified_Neural_Model.h"

//environments
#include"environments/Function_Approximation.h"
#include"environments/Single_Cart_Pole.h"
#include"environments/Double_Cart_Pole.h"
#include"environments/Mountain_Car.h"
#include"environments/Multiplexer.h"

#include"parameters.h"
#include <iostream>
FILE* main_log_file;
enum ENV_TYPE { mountain_car, function_approximation, single_cart_pole, double_cart_pole, multiplexer };
std::mutex mtx;

void setFeatures(Reinforcement_Environment* env)
{
	
#ifdef SET_NORMALIZED_INPUT
	bool feature_available;
	
	feature_available= env->set(NORMALIZED_OBSERVATION);

	if(feature_available == false)
	{
		printf("NORMALIZED_OBSERVATION feature not available\n");
		exit(1);
	}
	else
	{
		fprintf(main_log_file,"Normalized Observation enabled\n");
	}
#endif

#ifdef SET_NORMALIZED_OUTPUT
	bool feature_available;
	
	feature_available= env->set(NORMALIZED_ACTION);

	if(feature_available == false)
	{
		printf("NORMALIZED_ACTION feature not available\n");
		exit(1);
	}
	else
	{
		fprintf(main_log_file,"Normalized Action enabled\n");
	}
#endif

}

Reinforcement_Environment* setup_env(ENV_TYPE env_type, Random* random, int &number_of_observation_vars, int& number_of_action_vars)
{
	Reinforcement_Environment* env;
	switch (env_type) {
		case mountain_car:
			env = new Mountain_Car(random);
			break;
		case function_approximation:
			env = new Function_Approximation(random,1000,false);
			break;
		case single_cart_pole:
			env = new Single_Cart_Pole(random);
			break;
		case double_cart_pole:
			env = new Double_Cart_Pole(random);
			break;
		case multiplexer:
			env = new Multiplexer(3,8,random);
			break;	
		default:
			printf("You done goofed\n");
			exit(1);
	}
	setFeatures(env);
	env->start(number_of_observation_vars, number_of_action_vars);
	return env;
}

void ProcessTrials(Unified_Neural_Model* agent, Reinforcement_Environment* env, int thread_id) {
	int subpop;
	int individual;

	//starting reward 
	double reward= env->step(NULL);		
	double step_counter=1;

	bool do_testing = true;

	// We need to loop this threads live to ask for new individuals to trial.
	// Each loop means: 
		// 1) Getting and testing a new individual.
		// 2) Waiting for sync (evolution in SUNA)
	int i = env->trial;
	while (do_testing){
		// Lock the mutex using std::lock_guard
		{
			std::lock_guard<std::mutex> lock(mtx);
			std::array<int, 2> individual_path = agent->getNextIndividual(); // Returns {testing_subpop, testing_individual}
			subpop = individual_path[0];
			individual = individual_path[1];
		}

		if (subpop == -2){ // End training
			do_testing = false;
			std::cout << "END TRAINING Thread " << thread_id << std::endl;
			break;
		}else if (subpop == -1){ // Wait for sync
			continue;
		} 

		double accum_reward=reward;
		// do one trial (multiple steps until the environment finish the trial or the trial reaches its MAX_STEPS)

		while(env->trial==i && step_counter <= env->MAX_STEPS)
		{

			agent->step(subpop, individual, env->observation, reward, thread_id);

			reward = env->step(agent->SUNA_action[thread_id]);		
			
			accum_reward += reward;

			step_counter++;
		}


	#ifdef TERMINATE_IF_MAX_STEPS_REACHED
		//end evolution when the MAX_STEPS is reached
		if(step_counter > env->MAX_STEPS)
		{
			i=trials;
		}
	#endif

		agent->updateReward(accum_reward, thread_id);

		// Create a new function to end episode for this specific thread and trial
		{
			std::lock_guard<std::mutex> lock(mtx);
			agent->endEpisode(subpop, individual, reward);
		}
			
		//if env->trial is the same as i, it means that the internal state of the environment has not changed
		//then it needs a restart to begin a new trial
		if(env->trial==i)
		{
			reward= env->restart();
		}
		else
		{
			reward= env->step(NULL);
		}

		step_counter = 1;
		i++;
	}
}

int main()
{
	#pragma region
	main_log_file= fopen("log.txt","w");

	Random* random= new State_of_Art_Random(time(NULL));
	
	//Reinforcement_Agent* agent= new Dummy(env);
	Unified_Neural_Model* agent= new Unified_Neural_Model(random);
	//Self_Organizing_Neurons* b= (Self_Organizing_Neurons*)agent;
	
	// These two variables gets initial values from being passed as references to setup_env()
	int number_of_observation_vars;
	int number_of_action_vars;
	
	// Create the right number of environments. Each thread needs its own env.
	std::vector<Reinforcement_Environment*> environments;
	
	environments.reserve(NUMBER_OF_THREADS);  // Reserve memory for the vector

	for (int i = 0; i < NUMBER_OF_THREADS; ++i) {
        Reinforcement_Environment* env = setup_env(ENV_TYPE::mountain_car, random, number_of_observation_vars, number_of_action_vars);
        environments.push_back(env);  // Store the pointer to the newly created environment in the vector
    }

	////TODO: WRITING THE DEALLOCATION HERE SO I DO NOT FORGET!!
    

	agent->init(number_of_observation_vars, number_of_action_vars);
	
	#pragma endregion

	std::vector<std::thread> threads(NUMBER_OF_THREADS);
	// We must loop each trial for each thread and their individuals

	for (int t = 0; t < NUMBER_OF_THREADS; ++t) {
		threads[t] = std::thread(ProcessTrials, agent, environments[t], t);
	}
	// Wait for all threads to complete
	for (auto& th : threads) {
		th.join();
	}

	//agent->saveAgent("dna_best_individual");
	
	for (auto env : environments) {
		delete env;  // Assuming dynamic allocation in setup_env()
	}
	environments.clear();

	return 0;
}
