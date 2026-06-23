/*
###############################################################################
# If you use PhysiCell in your project, please cite PhysiCell and the version #
# number, such as below:                                                      #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1].    #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# See VERSION.txt or call get_PhysiCell_version() to get the current version  #
#     x.y.z. Call display_citations() to get detailed information on all cite-#
#     able software used in your PhysiCell application.                       #
#                                                                             #
# Because PhysiCell extensively uses BioFVM, we suggest you also cite BioFVM  #
#     as below:                                                               #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1],    #
# with BioFVM [2] to solve the transport equations.                           #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# [2] A Ghaffarizadeh, SH Friedman, and P Macklin, BioFVM: an efficient para- #
#     llelized diffusive transport solver for 3-D biological simulations,     #
#     Bioinformatics 32(8): 1256-8, 2016. DOI: 10.1093/bioinformatics/btv730  #
#                                                                             #
###############################################################################
#                                                                             #
# BSD 3-Clause License (see https://opensource.org/licenses/BSD-3-Clause)     #
#                                                                             #
# Copyright (c) 2015-2021, Paul Macklin and the PhysiCell Project             #
# All rights reserved.                                                        #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
# this list of conditions and the following disclaimer.                       #
#                                                                             #
# 2. Redistributions in binary form must reproduce the above copyright        #
# notice, this list of conditions and the following disclaimer in the         #
# documentation and/or other materials provided with the distribution.        #
#                                                                             #
# 3. Neither the name of the copyright holder nor the names of its            #
# contributors may be used to endorse or promote products derived from this   #
# software without specific prior written permission.                         #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" #
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE   #
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE  #
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE   #
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR         #
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF        #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS    #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN     #
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)     #
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE  #
# POSSIBILITY OF SUCH DAMAGE.                                                 #
#                                                                             #
###############################################################################
*/

#include "./custom.h"

void create_cell_types( void )
{
	// set the random seed 
	if (parameters.ints.find_index("random_seed") != -1)
	{
		SeedRandom(parameters.ints("random_seed"));
	}
	
	/* 
	   Put any modifications to default cell definition here if you 
	   want to have "inherited" by other cell types. 
	   
	   This is a good place to set default functions. 
	*/ 
	
	initialize_default_cell_definition(); 
	cell_defaults.phenotype.secretion.sync_to_microenvironment( &microenvironment ); 
	
	cell_defaults.functions.volume_update_function = standard_volume_update_function;
	cell_defaults.functions.update_velocity = standard_update_cell_velocity;

	cell_defaults.functions.update_migration_bias = NULL; 
	cell_defaults.functions.update_phenotype = NULL; // update_cell_and_death_parameters_O2_based; 
	cell_defaults.functions.custom_cell_rule = NULL; 
	cell_defaults.functions.contact_function = NULL; 
	
	cell_defaults.functions.add_cell_basement_membrane_interactions = NULL; 
	cell_defaults.functions.calculate_distance_to_membrane = NULL; 
	
	/*
	   This parses the cell definitions in the XML config file. 
	*/
	
	initialize_cell_definitions_from_pugixml(); 

	/*
	   This builds the map of cell definitions and summarizes the setup. 
	*/
		
	build_cell_definitions_maps(); 

	/*
	   This intializes cell signal and response dictionaries 
	*/

	setup_signal_behavior_dictionaries(); 	

	/* 
	   Put any modifications to individual cell definitions here. 
	   
	   This is a good place to set custom functions. 
	*/ 
	
	cell_defaults.functions.update_phenotype = phenotype_function; 
	cell_defaults.functions.custom_cell_rule = custom_function; 
	cell_defaults.functions.contact_function = contact_function; 

	// ?? // 
	cell_defaults.parameters.o2_proliferation_saturation = 38.0;  
	cell_defaults.parameters.o2_reference = 38.0; 

	// cancer cells 
	Cell_Definition* pCD = find_cell_definition( "cancer cell");
	pCD->functions.update_phenotype = phenotype_function;  // MISO: O2-dependent complementary secretion

	pCD->parameters.o2_proliferation_saturation = 38.0;  
	pCD->parameters.o2_reference = 38.0; 

	// cargo cells 
	pCD = find_cell_definition( "cargo cell"); 

	// figure out mechanics parameters 
	
	pCD->phenotype.mechanics.relative_maximum_attachment_distance 
		= pCD->custom_data["max_attachment_distance"] / pCD->phenotype.geometry.radius ; 

	pCD->phenotype.mechanics.relative_detachment_distance 
		= pCD->custom_data["max_elastic_displacement"] / pCD->phenotype.geometry.radius ; 
		
	pCD->phenotype.mechanics.attachment_elastic_constant 
		= pCD->custom_data["elastic_coefficient"]; 
	
	// set functions 
	pCD->functions.update_phenotype = cargo_cell_phenotype_rule; 
	pCD->functions.custom_cell_rule = cargo_cell_rule; 
	pCD->functions.contact_function = biorobots_contact_function; 
	pCD->functions.update_migration_bias = NULL;	

	// worker cells 

	pCD = find_cell_definition( "worker cell");

	pCD->phenotype.mechanics.relative_maximum_attachment_distance 
		= pCD->custom_data["max_attachment_distance"] / pCD->phenotype.geometry.radius ; 

	pCD->phenotype.mechanics.relative_detachment_distance 
		= pCD->custom_data["max_elastic_displacement"] / pCD->phenotype.geometry.radius ; 
		
	pCD->phenotype.mechanics.attachment_elastic_constant 
		= pCD->custom_data["elastic_coefficient"]; 

	pCD->functions.update_phenotype = NULL; // worker_cell_rule; 
	pCD->functions.custom_cell_rule = worker_cell_rule;  
	pCD->functions.contact_function = biorobots_contact_function; 
	
	/*
	   This builds the map of cell definitions and summarizes the setup. 
	*/
		
	display_cell_definitions( std::cout ); 
	
	return; 
}

void setup_microenvironment( void )
{
	// set domain parameters 
	
	// put any custom code to set non-homogeneous initial conditions or 
	// extra Dirichlet nodes here. 
	
	// initialize BioFVM 
	
	initialize_microenvironment(); 	
	
	return; 
}

void setup_tissue( void )
{
	double Xmin = microenvironment.mesh.bounding_box[0]; 
	double Ymin = microenvironment.mesh.bounding_box[1]; 
	double Zmin = microenvironment.mesh.bounding_box[2]; 

	double Xmax = microenvironment.mesh.bounding_box[3]; 
	double Ymax = microenvironment.mesh.bounding_box[4]; 
	double Zmax = microenvironment.mesh.bounding_box[5]; 
	
	if( default_microenvironment_options.simulate_2D == true )
	{
		Zmin = 0.0; 
		Zmax = 0.0; 
	}
	
	double Xrange = Xmax - Xmin; 
	double Yrange = Ymax - Ymin; 
	double Zrange = Zmax - Zmin; 
	
	// create some of each type of cell 
	
	Cell* pC;
	
	for( int k=0; k < cell_definitions_by_index.size() ; k++ )
	{
		Cell_Definition* pCD = cell_definitions_by_index[k]; 
		std::cout << "Placing cells of type " << pCD->name << " ... " << std::endl; 
		for( int n = 0 ; n < parameters.ints("number_of_cells") ; n++ )
		{
			std::vector<double> position = {0,0,0}; 
			position[0] = Xmin + UniformRandom()*Xrange; 
			position[1] = Ymin + UniformRandom()*Yrange; 
			position[2] = Zmin + UniformRandom()*Zrange; 
			
			pC = create_cell( *pCD ); 
			pC->assign_position( position );
		}
	}
	std::cout << std::endl; 

	// custom placement 
	// place a cluster of tumor cells at the center 
	
	double cell_radius = cell_defaults.phenotype.geometry.radius; 
	double cell_spacing = 0.95 * 2.0 * cell_radius; 
	
	double tumor_radius = parameters.doubles("tumor_radius"); // 200.0;
	// MISO: randomized tumor center. Defaults to origin if unset.
	double cx = parameters.doubles("tumor_center_x");
	double cy = parameters.doubles("tumor_center_y");

	Cell_Definition* pCD_cancer = find_cell_definition( "cancer cell");

	// MISO emergent-growth mode (growth_mode=1): seed a SMALL cluster of cancer cells (1-3 small
	// foci near the hidden centre) and let the tumor grow its OWN morphology through O2-limited
	// proliferation + cell mechanics, instead of stamping a prescribed Fourier boundary. The
	// default (growth_mode=0, or parameter absent) reproduces the original benchmark exactly.
	int growth_mode = 0;
	if( parameters.doubles.find_index("growth_mode") != -1 )
		growth_mode = (int) parameters.doubles("growth_mode");
	if( growth_mode == 1 )
	{
		const double PI_g = 3.14159265358979323846;
		double seed_radius = 40.0;
		if( parameters.doubles.find_index("tumor_seed_radius") != -1 )
			seed_radius = parameters.doubles("tumor_seed_radius");
		int gseed = (int) parameters.doubles("tumor_shape_seed");
		unsigned long long rsg = (unsigned long long)(gseed) * 2654435761ULL + 12345ULL;
		auto urandg = [&]() -> double {
			rsg = rsg * 6364136223846793005ULL + 1442695040888963407ULL;
			return (double)((rsg >> 33) & 0x7fffffffULL) / 2147483647.0;
		};
		int n_foci = 1 + (int)( urandg() * 3.0 );   // 1..3 small foci -> emergent lumpy/multifocal mass
		for( int f = 0; f < n_foci; f++ )
		{
			double ang = urandg()*2.0*PI_g;
			double off = ( f == 0 ) ? 0.0 : ( 0.6 + 0.8*urandg() ) * seed_radius;
			double fxc = cx + off*cos(ang), fyc = cy + off*sin(ang);
			for( double yy = fyc - seed_radius; yy <= fyc + seed_radius; yy += cell_spacing*sqrt(3.0)/2.0 )
			{
				for( double xx = fxc - seed_radius; xx <= fxc + seed_radius; xx += cell_spacing )
				{
					if( (xx-fxc)*(xx-fxc) + (yy-fyc)*(yy-fyc) <= seed_radius*seed_radius )
					{
						Cell* pCg = create_cell(*pCD_cancer);
						pCg->assign_position( xx, yy, 0.0 );
					}
				}
			}
		}
		load_cells_from_pugixml();
		return;
	}

	// MISO 3D benchmark: when the domain is 3D, stamp an irregular SPHERE (low-order angular
	// perturbations) of cancer cells centred at (cx,cy,cz). The same O2-dependent phenotype then
	// yields a hypoxic core + oxygenated rim in 3D. Gated on simulate_2D, so the 2D path is untouched.
	if( default_microenvironment_options.simulate_2D == false )
	{
		const double PI3 = 3.14159265358979323846;
		double cz = 0.0;
		if( parameters.doubles.find_index("tumor_center_z") != -1 )
			cz = parameters.doubles("tumor_center_z");
		int sseed = (int) parameters.doubles("tumor_shape_seed");
		double rough = parameters.doubles("tumor_roughness");
		unsigned long long rs3 = (unsigned long long)(sseed) * 2654435761ULL + 12345ULL;
		auto ur3 = [&]() -> double {
			rs3 = rs3 * 6364136223846793005ULL + 1442695040888963407ULL;
			return (double)((rs3 >> 33) & 0x7fffffffULL) / 2147483647.0;
		};
		double a1 = rough*(2*ur3()-1), a2 = rough*(2*ur3()-1), a3 = rough*(2*ur3()-1);
		double p1 = ur3()*2*PI3, p2 = ur3()*2*PI3, p3 = ur3()*2*PI3;
		auto inside3 = [&]( double px, double py, double pz ) -> bool {
			double ddx=px-cx, ddy=py-cy, ddz=pz-cz;
			double rr = sqrt(ddx*ddx+ddy*ddy+ddz*ddz);
			if( rr < 1e-9 ) return true;
			double th = atan2(ddy,ddx), ph = acos( ddz / rr );
			double bnd = tumor_radius * (1.0 + a1*cos(2*th+p1) + a2*cos(3*ph+p2) + a3*cos(2*th+3*ph+p3));
			return rr <= bnd;
		};
		double ext = tumor_radius*(1.0+rough) + cell_spacing;
		for( double zz = cz-ext; zz <= cz+ext; zz += cell_spacing )
		 for( double yy = cy-ext; yy <= cy+ext; yy += cell_spacing )
		  for( double xx = cx-ext; xx <= cx+ext; xx += cell_spacing )
		   if( inside3(xx,yy,zz) )
		   {
		   	Cell* pC3 = create_cell(*pCD_cancer);
		   	pC3->assign_position( xx, yy, zz );
		   }
		load_cells_from_pugixml();
		return;
	}

	// MISO: IRREGULAR tumor boundary via radial Fourier harmonics + optional second focus.
	// Shape is deterministic in tumor_shape_seed so the harness can reproduce / hold out shapes.
	int shape_seed = (int) parameters.doubles("tumor_shape_seed");
	double roughness = parameters.doubles("tumor_roughness");   // ~0.1-0.45
	const double PI = 3.14159265358979323846;
	unsigned long long rs = (unsigned long long)(shape_seed) * 2654435761ULL + 12345ULL;
	auto urand = [&]() -> double {
		rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
		return (double)((rs >> 33) & 0x7fffffffULL) / 2147483647.0;
	};
	const int K = 6;
	double amp[K+1], pha[K+1];
	for( int k = 2; k <= K; k++ ){ amp[k] = roughness * (2.0*urand()-1.0) / (double)(k-1); pha[k] = urand()*2.0*PI; }
	bool multifocal = ( urand() < 0.5 );
	double ang2 = urand()*2.0*PI, dist2 = (0.5+0.5*urand())*tumor_radius;
	double fx2 = cx + dist2*cos(ang2), fy2 = cy + dist2*sin(ang2);
	double r2 = tumor_radius*(0.35 + 0.35*urand());

	auto inside = [&]( double px, double py ) -> bool {
		double ddx = px - cx, ddy = py - cy;
		double rr = sqrt(ddx*ddx + ddy*ddy), th = atan2(ddy, ddx);
		double bnd = tumor_radius;
		for( int k = 2; k <= K; k++ ){ bnd += tumor_radius*amp[k]*cos((double)k*th + pha[k]); }
		if( rr <= bnd ) return true;
		if( multifocal ){ double e = sqrt((px-fx2)*(px-fx2)+(py-fy2)*(py-fy2)); if( e <= r2 ) return true; }
		return false;
	};

	// scan bounding box covering main blob (+harmonics) and optional second focus
	double main_ext = tumor_radius*(1.0 + roughness) + cell_spacing;
	double xlo = fmin(cx - main_ext, fx2 - r2), xhi = fmax(cx + main_ext, fx2 + r2);
	double ylo = fmin(cy - main_ext, fy2 - r2), yhi = fmax(cy + main_ext, fy2 + r2);
	int row = 0;
	for( double yy = ylo; yy <= yhi; yy += cell_spacing*sqrt(3.0)/2.0 )
	{
		double xoff = (row % 2 == 1) ? 0.5*cell_spacing : 0.0;
		for( double xx = xlo + xoff; xx <= xhi; xx += cell_spacing )
		{
			if( inside(xx, yy) )
			{
				Cell* pC = create_cell(*pCD_cancer);
				pC->assign_position( xx, yy, 0.0 );
			}
		}
		row++;
	}

	
	// load cells from your CSV file (if enabled)
	load_cells_from_pugixml(); 	
	
	return; 
}

std::vector<std::string> my_coloring_function( Cell* pCell )
{ return paint_by_number_cell_coloring(pCell); }

void phenotype_function( Cell* pCell, Phenotype& phenotype, double dt )
{
	// MISO complementary biomarker channels via O2-dependent secretion:
	//   normoxic (proliferating RIM) cell  -> secretes chemoattractant, NOT therapeutic
	//   hypoxic  (necrotic CORE)     cell  -> secretes therapeutic,     NOT chemoattractant
	// oxygen is consumed by viable cells (broad depletion). => each channel reveals a different
	// compartment; fusing them is required to recover the full tumor (core + rim).
	static int o2_i    = microenvironment.find_density_index( "oxygen" );
	static int chemo_i = microenvironment.find_density_index( "chemoattractant" );
	static int ther_i  = microenvironment.find_density_index( "therapeutic" );
	static double hyp_thr = parameters.doubles("hypoxia_threshold");   // e.g. 12-18 mmHg
	static double rate    = parameters.doubles("biomarker_secretion"); // e.g. 6

	// standard O2-based death (lets a hypoxic/necrotic core emerge as the tumor grows)
	update_cell_and_death_parameters_O2_based( pCell, phenotype, dt );

	double o2 = pCell->nearest_density_vector()[o2_i];
	if( o2 < hyp_thr )   // hypoxic core
	{
		phenotype.secretion.secretion_rates[chemo_i] = 0.0;
		phenotype.secretion.secretion_rates[ther_i]  = rate;
		phenotype.secretion.saturation_densities[ther_i] = 1.0;
	}
	else                 // oxygenated rim
	{
		phenotype.secretion.secretion_rates[chemo_i] = rate;
		phenotype.secretion.saturation_densities[chemo_i] = 1.0;
		phenotype.secretion.secretion_rates[ther_i]  = 0.0;
	}
	return;
}

void custom_function( Cell* pCell, Phenotype& phenotype , double dt )
{ return; } 

void contact_function( Cell* pMe, Phenotype& phenoMe , Cell* pOther, Phenotype& phenoOther , double dt )
{ return; } 

std::vector<std::string> cancer_biorobots_coloring_function( Cell* pCell )
{
	std::vector< std::string > output( 4, "black" ); 
	
	double damage = get_single_signal( pCell, "damage"); 

	static double max_damage = 1.0 * get_single_signal(pCell,"custom:damage_rate") 
		/ (1e-16 + get_single_signal(pCell,"custom:repair_rate" ) );

	static Cell_Definition* pCD_cargo = find_cell_definition( "cargo cell"); 
	static Cell_Definition* pCD_cancer = find_cell_definition( "cancer cell"); 
	static Cell_Definition* pCD_worker = find_cell_definition( "worker cell"); 
	
	// cargo cell 
	if( pCell->type == pCD_cargo->type )
	{
		output[0] = "blue";
		output[1] = "blue";
		output[2] = "blue"; 
		output[3] = "none"; // no nuclear outline color 
		return output;
	}
	
	// worker cell 
	if( pCell->type == pCD_worker->type )
	{
		output[0] = "red";
		output[1] = "red";
		output[2] = "red"; 
		output[3] = "none"; // no nuclear outline color 
		return output;
	}
	
	// apoptotic tumor - cyan 
	if( get_single_signal( pCell, "apoptotic" ) > 0.5 )  // Apoptotic - cyan
	{
		output[0] = "cyan";
		output[2] = "darkcyan"; 
		return output; 
	}	
	
	// Necrotic tumor - Brown
	if( get_single_signal( pCell, "necrotic") > 0.5 )
	{
		output[0] = "rgb(250,138,38)";
		output[2] = "rgb(139,69,19)";
		return output; 
	}		
	
	// live tumor -- shade by level of damage 
	
	
	// if live: color by damage 
	if( get_single_signal( pCell, "dead") < 0.5 )
	{
		int damage_int = (int) round( damage * 255.0 / max_damage ); 
		
		char szTempString [128];
		sprintf( szTempString , "rgb(%u,%u,%u)" , damage_int , 255-damage_int , damage_int );
		output[0].assign( szTempString );
		output[1].assign( szTempString );
		sprintf( szTempString , "rgb(%u,%u,%u)" , damage_int/4 , (255-damage_int)/4 , damage_int/4 );
		output[2].assign( szTempString );
	}
	return output; 
}

void introduce_biorobots( void )
{
	// idea: we'll "inject" them in a little column
		
	static double worker_fraction = 
		parameters.doubles("worker_fraction"); // 0.10; /* param */ 
	static int number_of_injected_cells = 
		parameters.ints("number_of_injected_cells"); // 500; /* param */ 
	
	// make these vary with domain size 
	double left_coordinate = default_microenvironment_options.X_range[1] - 150.0; // 600.0; 
	double right_cooridnate = default_microenvironment_options.X_range[1] - 50.0; // 700.0;

	double bottom_coordinate = default_microenvironment_options.Y_range[0] + 50.0; // -700; 
	double top_coordinate = default_microenvironment_options.Y_range[1] - 50.0; // 700; 

	Cell_Definition* pCD_worker = find_cell_definition( "worker cell");
	Cell_Definition* pCD_cargo = find_cell_definition( "cargo cell");
		
	for( int i=0 ;i < number_of_injected_cells ; i++ )
	{
		std::vector<double> position = {0,0,0}; 
		position[0] = left_coordinate + (right_cooridnate-left_coordinate)*UniformRandom(); 
		position[1] = bottom_coordinate + (top_coordinate-bottom_coordinate)*UniformRandom(); 
		
		Cell* pCell;  
		if( UniformRandom() <= worker_fraction )
		{ pCell = create_cell( *pCD_worker ); }
		else
		{ pCell = create_cell( *pCD_cargo ); }
		pCell->assign_position( position ); 
	}
	
	return; 
}

void cargo_cell_rule( Cell* pCell, Phenotype& phenotype, double dt )
{
	if( get_single_signal( pCell, "dead" ) > 0.5 )
	{
		// the cell death functions don't automatically turn off custom functions, 
		// since those are part of mechanics. 
		
		// Let's just fully disable now. 
		pCell->functions.custom_cell_rule = NULL; 
		return; 
	}
	
	// if I'm docked
	if( pCell->state.number_of_attached_cells() > 0 )
	{
		set_single_behavior( pCell, "migration speed" , 0.0 ); 
		return; 
	}
	
	return; 
}

void cargo_cell_phenotype_rule( Cell* pCell, Phenotype& phenotype, double dt )
{
	// if dettached and receptor on, secrete signal 
	
	// if dettached and receptor off, secrete chemo

	double receptor = get_single_signal( pCell , "custom:receptor" ); 
	
	if( pCell->state.number_of_attached_cells() == 0 )
	{
		if( receptor > 0.1 )
		{
			set_single_behavior( pCell , "chemoattractant secretion" , 10); 
			set_single_behavior( pCell , "therapeutic secretion" , 0); 
		}
		else
		{
			set_single_behavior( pCell , "chemoattractant secretion" , 0); 
			set_single_behavior( pCell , "therapeutic secretion" , 10); 
		}
		return; 
	}
	
	// if you reach this point of the code, the cell is attached 
	

	// if attached and oxygen high, secrete nothing, receptor off 
	
	// if attached and oxygen low, dettach, start secreting chemo, receptor off   

	double o2 = get_single_signal( pCell, "oxygen"); 
	double o2_drop = get_single_signal( pCell , "custom:cargo_release_o2_threshold"); 
	
	if( o2 > o2_drop )
	{
		set_single_behavior( pCell , "chemoattractant secretion" , 0); 
		set_single_behavior( pCell , "therapeutic secretion" , 0); 
		set_single_behavior( pCell , "custom:receptor" , 0 ); 
	}
	else
	{
		set_single_behavior( pCell , "chemoattractant secretion" , 0); 
		set_single_behavior( pCell , "therapeutic secretion" , 10); 
		set_single_behavior( pCell , "custom:receptor" , 0 ); 
		
		pCell->remove_all_attached_cells(); 
	}
	
	return; 
}

void biorobots_contact_function( Cell* pActingOn, Phenotype& pao, Cell* pAttachedTo, Phenotype& pat , double dt )
{
	std::vector<double> displacement = pAttachedTo->position - pActingOn->position; 
	
	static double max_elastic_displacement = pao.geometry.radius * pao.mechanics.relative_detachment_distance; 
	static double max_displacement_squared = max_elastic_displacement*max_elastic_displacement; 
	
	// detach cells if too far apart 
	
	if( norm_squared( displacement ) > max_displacement_squared )
	{
		detach_cells( pActingOn , pAttachedTo );
		return; 
	}
	
	axpy( &(pActingOn->velocity) , pao.mechanics.attachment_elastic_constant , displacement ); 
	
	return; 
}

void tumor_cell_phenotype_with_therapy( Cell* pCell, Phenotype& phenotype, double dt )
{
	double damage = get_single_signal( pCell, "damage"); 

	double damage_rate = get_single_signal( pCell , "custom:damage_rate"); 
	double repair_rate = get_single_signal( pCell , "custom:repair_rate"); 
	double drug_death_rate = get_single_signal( pCell , "custom:drug_death_rate" ); 

	double drug = get_single_signal( pCell , "therapeutic"); 
	
	static double max_damage = 1.0 * damage_rate / (1e-16 + repair_rate );
	
	// if I'm dead, don't bother. disable my phenotype rule
	if( get_single_signal( pCell, "dead") > 0.5 )
	{
		pCell->functions.update_phenotype = NULL; 
		return; 
	}
	
	// first, vary the cell birth and death rates with oxygenation
	
	// std::cout << get_single_behavior( pCell , "cycle entry") << " vs "; 
	update_cell_and_death_parameters_O2_based(pCell,phenotype,dt);
	// std::cout << get_single_behavior( pCell , "cycle entry") << std::endl; 

	// the update the cell damage 
	
	// dD/dt = alpha*c - beta-D by implicit scheme 
	
	double temp = drug;
	
	// reuse temp as much as possible to reduce memory allocations etc. 
	temp *= dt; 
	temp *= damage_rate; 
	
	damage += temp; // d_prev + dt*chemo*damage_rate 
	
	temp = repair_rate;
	temp *= dt; 
	temp += 1.0; 
	damage /= temp;  // (d_prev + dt*chemo*damage_rate)/(1 + dt*repair_rate)
	
	// then, see if the cell undergoes death from the therapy 
	
	temp = dt; 
	temp *= damage; 
	temp *= drug_death_rate; 
	temp /= max_damage; // dt*(damage/max_damage)*death_rate 

	// make sure we write the damage (not current a behavior)
	pCell->phenotype.cell_integrity.damage = damage; 

	if( UniformRandom() <= temp )
	{
		// pCell->start_death( apoptosis_model_index );
		set_single_behavior( pCell, "apoptosis" , 9e99 ); 
		pCell->functions.update_phenotype = NULL; 		
		pCell->functions.custom_cell_rule = NULL; 
	}

	return; 
}

void worker_cell_rule( Cell* pCell, Phenotype& phenotype, double dt )
{
	// if I am dead, don't bother

	if( get_single_signal( pCell , "dead") > 0.5 )
	{
		// the cell death functions don't automatically turn off custom functions, 
		// since those are part of mechanics. 
		
		// Let's just fully disable now. 
		pCell->functions.custom_cell_rule = NULL; 
		return; 
	}
	
	// am I searching for cargo? if so, see if I've found it
	if( pCell->state.number_of_attached_cells() == 0 )
	{
		std::vector<Cell*> nearby = pCell->cells_in_my_container(); 
		bool attached = false; // want to limit to one attachment 
		int i =0;
		while( i < nearby.size() && attached == false )
		{
			// if it is expressing the receptor, dock with it 
			if( get_single_signal(nearby[i],"custom:receptor") > 0.5 && attached == false )
			{
				attach_cells( pCell, nearby[i] ); 
				// nearby[i]->custom_data["receptor"] = 0.0; // put into cargo cell rule instead? 
				// nearby[i]->phenotype.secretion.set_all_secretion_to_zero(); // put into cargo rule instead? 
				attached = true; 
			}
			i++; 
		}
	}

	// from prior motility function 

	double o2 = get_single_signal( pCell, "oxygen");
	double chemoattractant = get_single_signal( pCell , "chemoattractant"); 

	static double detection_threshold = get_single_signal( pCell, "custom:motility_shutdown_detection_threshold"); 
	
	// if attached, biased motility towards director chemoattractant 
	// otherwise, biased motility towards cargo chemoattractant 
	
	static double attached_worker_migration_bias = get_single_signal( pCell, "custom:attached_worker_migration_bias"); 
	static double unattached_worker_migration_bias = get_single_signal( pCell , "custom:unattached_worker_migration_bias"); 
	
	if( pCell->state.number_of_attached_cells() > 0 )
	{
		set_single_behavior( pCell , "migration bias" , attached_worker_migration_bias ); 

		set_single_behavior( pCell , "chemotactic response to oxygen" , -1 ); 
		set_single_behavior( pCell , "chemotactic response to chemoattractant" , 0 ); 
	}
	else
	{
		// if there is no detectable signal, shut down motility (permanently)
		if( chemoattractant < detection_threshold )
		{
			set_single_behavior( pCell, "migration speed" , 0 ); 
		}
		
		set_single_behavior( pCell , "migration bias" , unattached_worker_migration_bias ); 
		
 		set_single_behavior( pCell , "chemotactic response to oxygen" , 0 ); 
		set_single_behavior( pCell , "chemotactic response to chemoattractant" , 1 ); 
	}
	
	return; 
}

