#include"Unified_Neural_Model.h"


Unified_Neural_Model::Unified_Neural_Model(Random* random)
{
	this->random= random;

	Module::setRandom(random);
	
	testing_individual= 0;	
	testing_individual_done=0;
	generation=1;
	best_index=0;
	wait_for_sync = false;
	curr_generation_trial = 0;
	end_training = false;

#ifdef	SPECTRUM_DIVERSITY
	nmap= new Novelty_Map(NOVELTY_MAP_SIZE , SPECTRUM_SIZE);
	
		
	if(NOVELTY_MAP_SIZE >= SUBPOPULATION_SIZE)
	{
		printf("ERROR: Novelty map's size is bigger or equal to the subpopulation size\n");
		exit(1);
	}

#endif

	step_counter=0;
}

// Full reset function
Unified_Neural_Model::~Unified_Neural_Model()
{
	free(action);

	for (int i = 0; i < NUMBER_OF_THREADS; i++)
	{
		free(SUNA_action[i]);
	}
	
	for(int j=0;j<SUBPOPULATION_SIZE; ++j)
	{
		delete subpopulation[j];
	}

	delete subpopulation;
	free(fitness);
}

		
void Unified_Neural_Model::saveAgent(const char* filename)
{
	//execute individual
	subpopulation[best_index]->saveDNA(filename);		

	subpopulation[best_index]->printGraph("best_individual.dot");		
}

void Unified_Neural_Model::loadAgent(const char* filename)
{
	//execute individual
	subpopulation[best_index]->loadDNA(filename);	
	
	subpopulation[best_index]->clearMemory();		
	}

void Unified_Neural_Model::init(int number_of_observation_vars, int number_of_action_vars)
{
	this->number_of_observation_vars= number_of_observation_vars;
	this->number_of_action_vars= number_of_action_vars;

	for (int i = 0; i < NUMBER_OF_THREADS; i++)
	{
		SUNA_action[i] = (double*) calloc(number_of_action_vars, sizeof(double));
		accumulated_rewards_per_thread[i] = 0;
		rewards_initialized[i] = false;
	}
	

	subpopulation= new Module*[SUBPOPULATION_SIZE];
	tmp_subpopulation= new Module*[SUBPOPULATION_SIZE];
	fitness= (double*)malloc(sizeof(double)*SUBPOPULATION_SIZE);
	tmp_fitness= (double*)malloc(sizeof(double)*SUBPOPULATION_SIZE);

	// Loop over cells in current subpopulation in novelty map
	for(int j=0;j<SUBPOPULATION_SIZE; ++j)
	{
		subpopulation[j]= new Module(number_of_observation_vars, number_of_action_vars, INITIAL_ALLOCATION_LENGTH);
		tmp_subpopulation[j]= new Module(number_of_observation_vars, number_of_action_vars, INITIAL_ALLOCATION_LENGTH);

		fitness[j]= EXTREME_NEGATIVE_REWARD;

		//starting mutations
		for(int k=0; k < NUMBER_OF_INITIAL_MUTATIONS; ++k)
		{
			subpopulation[j]->structuralMutation();
		}
		subpopulation[j]->updatePrimerList();
	}

	
}

// Legacy function from Reinforcement_Agent, don't use!
void Unified_Neural_Model::step(double* observation, double reward)
{
	//execute individual
	subpopulation[testing_individual]->process(observation, action);		

	//update reward
	tmp_fitness[testing_individual]+= reward;
}

void Unified_Neural_Model::step(int individual, double* observation, double reward, int thread_id)
{
	//execute individual
	subpopulation[individual]->process(observation, SUNA_action[thread_id]);		

	//update reward
	tmp_fitness[individual]+= reward;
}

// This function acts like a queue where a thread can get the next individual to run a trial.
// Purpose: This will limit the time a thread does nothing/is waiting.
int Unified_Neural_Model::getNextIndividual(){
	// If all generations have finished their training return NULL to end training.
	if (end_training){
		return -2;
	}

	// We need to wait for sync so that all individuals have done their trial so Evolution can be done.
	if (wait_for_sync){
		return -1;
	}

	//test a new individual from this subpopulation
	testing_individual++;

	//change subpopulation if all individuals from this subpop were already tested
	//evolve if all subpopulations and all individuals were tested (evaluated)
	if(testing_individual >= SUBPOPULATION_SIZE)
	{
		wait_for_sync=true;
		testing_individual=0;
	}
	return testing_individual;
}

void Unified_Neural_Model::updateReward(double reward, int thread_id){
	if (!rewards_initialized[thread_id]){
		accumulated_rewards_per_thread[thread_id] = reward;
		rewards_initialized[thread_id] = true;
	}
	else if (accumulated_rewards_per_thread[thread_id] < reward){
		accumulated_rewards_per_thread[thread_id] = reward;
	}
}

void Unified_Neural_Model::endEpisode(double reward)
{
	//update reward
	tmp_fitness[testing_individual]+= reward;

	step_counter++;
	

	//printf("%f\n",tmp_fitness[testing_individual]);

	//average 
	if(step_counter >= EPISODES_PER_INDIVIDUAL)
	{
		tmp_fitness[testing_individual]/= (double) step_counter;
		step_counter=0;

	}
	else
	{
		return;
	}
	
	subpopulation[testing_individual]->clearMemory();		
	
	//test a new individual from this subpopulation
	testing_individual++;

	//change subpopulation if all individuals from this subpop were already tested
	//evolve if all subpopulations and all individuals were tested (evaluated)
	if(testing_individual >= SUBPOPULATION_SIZE)
	{
		//Evolve
#ifdef	SPECTRUM_DIVERSITY
		spectrumDiversityEvolve();
#endif
		testing_individual=0;
	}

}

void Unified_Neural_Model::endEpisode(int individual, double reward)
{
	//update reward
	tmp_fitness[individual]+= reward;

	subpopulation[individual]->clearMemory();		
	
	//test a new individual from this subpopulation
	testing_individual_done++;

	//change subpopulation if all individuals from this subpop were already tested
	//evolve if all subpopulations and all individuals were tested (evaluated)
	if(testing_individual_done >= SUBPOPULATION_SIZE)
	{
		bool waitforsync = true;
		curr_generation_trial++;

		// Print average reward for this generation
		double best_reward = accumulated_rewards_per_thread[0];
		for (int i = 1; i < NUMBER_OF_THREADS; i++)
		{
			if (best_reward < accumulated_rewards_per_thread[i]){
				best_reward = accumulated_rewards_per_thread[i];
			}
		}
		printf("Best reward of generation %d, is:  %f \n", curr_generation_trial-1, best_reward);

		for (int i = 0; i < NUMBER_OF_THREADS; i++)
		{
			accumulated_rewards_per_thread[i] = 0;
			rewards_initialized[i] = false;
		}

		if (curr_generation_trial >= MAX_GENERATIONS_TRIALS){
			end_training = true;
		} else {

			//Evolve
#ifdef	SPECTRUM_DIVERSITY
			spectrumDiversityEvolve();
#endif
			waitforsync = false;
		}
		testing_individual_done=0;
		wait_for_sync = waitforsync;
	}

}

void Unified_Neural_Model::print()
{
	subpopulation[best_index]->printGraph("best_individual.dot");		
}

double Unified_Neural_Model::stepBestAction(double* observation)
{
	//execute individual
	subpopulation[best_index]->process(observation, action);		

	return 0;	
}

void Unified_Neural_Model::endBestEpisode()
{
	subpopulation[best_index]->clearMemory();		
}

bool Unified_Neural_Model::cell_insert_check(nmap_cell* cell, int individual, double this_fitness){
	if (cell->module==NULL ||
	 	cell->fitness < this_fitness || 
		(cell->fitness == this_fitness && 
			(cell->module)->number_of_neurons >= subpopulation[individual]->number_of_neurons)){
				return true;
			}
	return false;
}

bool Unified_Neural_Model::find_best_individual(int best_number_of_neurons, int individual, double best_fitness){
	if (tmp_fitness[individual] > best_fitness ||
		(tmp_fitness[individual] == best_fitness && 
			best_number_of_neurons >= subpopulation[individual]->number_of_neurons)){
				return true;
			}
	return false;
}

void Unified_Neural_Model::spectrumDiversityEvolve()
{
	//update fitness
	double avg_fitness=0;
	double fcounter=0;
	double best_fitness= tmp_fitness[0];
	best_index= 0;
	int best_number_of_neurons= subpopulation[0]->number_of_neurons;		
		
	//find best individual	

	for(int j=0;j<SUBPOPULATION_SIZE; ++j)
	{
		fitness[j]= tmp_fitness[j];

		if(find_best_individual(best_number_of_neurons, j, best_fitness)){
			best_fitness= tmp_fitness[j];
			best_index=j;
			best_number_of_neurons= subpopulation[j]->number_of_neurons;	
		}
					
		//reset the fitness
		tmp_fitness[j]= 0.0;

		avg_fitness+= fitness[j];
		fcounter++;
	}
	
	avg_fitness/= fcounter;

	//decide the parents

	for(int j=0;j<SUBPOPULATION_SIZE; ++j)
	{
		//calculate spectrum
		double spectrum[SPECTRUM_SIZE];	


		calculateSpectrum(spectrum, j);

		//insert in Novelty Map
		int index= nmap->input(spectrum);

		nmap_cell* cell= (nmap_cell*)(nmap->map[index]).pointer;
		double this_fitness= fitness[j];
		
		if(cell==NULL)
		{
			cell= (nmap_cell*)malloc(sizeof(nmap_cell));
			nmap->map[index].pointer= cell;
			cell->module=NULL;
		}	
		
		//check if nothing was inserted
		if (cell_insert_check(cell, j, this_fitness)){
			cell->module= subpopulation[j];		
			cell->fitness= this_fitness;
		}
	}
	
	
		
	nmap_cell* cell= (nmap_cell*)(nmap->map[0]).pointer;
	best_fitness= cell->fitness;
	best_index= 0;

	//copy parents (nmap's cells) to the population
	for(int i=0;i<NOVELTY_MAP_SIZE;++i)
	{	
		cell= (nmap_cell*)(nmap->map[i]).pointer;
		
		if(cell->module!=NULL)
		{
			if(cell->fitness> best_fitness)
			{
				best_fitness=cell->fitness;
				best_index=i;
			}
		}
		
		//prevent the algorithm from choosing empty cells (those cells appear when there is many cells with the same weight array)
		while(cell->module==NULL)
		{	
			int roullete= random->uniform(0,NOVELTY_MAP_SIZE-1);
			cell= (nmap_cell*)(nmap->map[roullete]).pointer;
		}	

		tmp_subpopulation[i]->clone(cell->module);
	}

	//reproduce 
	for(int i=NOVELTY_MAP_SIZE;i<SUBPOPULATION_SIZE;++i)
	{
		int random_individual= random->uniform(0,NOVELTY_MAP_SIZE-1);
		cell= (nmap_cell*)(nmap->map[random_individual]).pointer;
		
		while(cell->module==NULL)
		{	
			int roullete= random->uniform(0,NOVELTY_MAP_SIZE-1);
			cell= (nmap_cell*)(nmap->map[roullete]).pointer;
		}	
			
		tmp_subpopulation[i]->clone(cell->module);

		int number_of_mutations= NUMBER_OF_STEP_MUTATIONS;

		//structural mutation
		for(int k=0; k < number_of_mutations; ++k)
		{
			tmp_subpopulation[i]->structuralMutation();
		}
		tmp_subpopulation[i]->updatePrimerList();
		

		//weight mutation
		tmp_subpopulation[i]->weightMutation();
	}
	
	//remove all modules inserted in the novelty map
	for(int i=0;i<NOVELTY_MAP_SIZE;++i)
	{	
		cell= (nmap_cell*)(nmap->map[i]).pointer;
		cell->module=NULL;
		cell->fitness=EXTREME_NEGATIVE_REWARD;
	}


	//swap temporary population with the original one	

	for(int j=0;j<SUBPOPULATION_SIZE; ++j)
	{
		Module* swap_individual= subpopulation[j];
		subpopulation[j]= tmp_subpopulation[j];
		tmp_subpopulation[j]= swap_individual;
	}

	generation++;	

}

// Create the Spectrum of the DNA
// | Identity | Sigmoid | Threshold | Random | Control | Slow |
void Unified_Neural_Model::calculateSpectrum(double* spectrum, int individual_index)
{
	Module* mod= subpopulation[individual_index];		

	//clear spectrum
	for(int i=0;i<SPECTRUM_SIZE;++i)
	{
		spectrum[i]=0;
	}
	
	//execute all Control Neurons that are excited and not activated
	int counter=0;
	for(int i=0; mod->n[i].id >= 0;++i)
	{
		switch(mod->n[i].type)
		{
			case IDENTITY:
			{
				spectrum[0]++;
			}
			break;

			case SIGMOID:
			{
				spectrum[1]++;
			}
			break;
			
			case THRESHOLD:
			{
				spectrum[2]++;
			}
			break;
			
			case RANDOM:
			{
				spectrum[3]++;
			}
			break;
			
			case CONTROL:
			{
				spectrum[4]++;
			}
			break;
		}
		
		if(mod->n[i].firing_rate != 1)
		{
			spectrum[5]++;
		}
		counter++;
	}

#ifdef NORMALIZED_SPECTRUM_DIVERSITY
	for(int i=0;i<SPECTRUM_SIZE;++i)
	{
		spectrum[i]/=(double)counter;
	}
#endif
}
