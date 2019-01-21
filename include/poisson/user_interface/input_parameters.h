/*
 * input_parameters.h
 *
 *  Created on:
 *      Author:
 */

#ifndef INCLUDE_LAPLACE_INPUT_PARAMETERS_H_
#define INCLUDE_LAPLACE_INPUT_PARAMETERS_H_

#include "../../solvers_and_preconditioners/multigrid/multigrid_input_parameters.h"
#include "../../solvers_and_preconditioners/solvers/solver_data.h"
#include "../include/functionalities/print_functions.h"
#include "postprocessor/error_calculation_data.h"
#include "postprocessor/output_data.h"

#include "enum_types.h"

namespace Poisson
{
class InputParameters
{
public:
  // standard constructor that initializes parameters with default values
  InputParameters()
    : // MATHEMATICAL MODEL
      right_hand_side(false),

      // PHYSICAL QUANTITIES

      // SPATIAL DISCRETIZATION
      degree_mapping(1),
      IP_factor(1.0),
      spatial_discretization(SpatialDiscretization::DG),

      // SOLVER
      solver(Solver::Undefined),
      solver_data(SolverData(1e4, 1.e-20, 1.e-12)),
      compute_performance_metrics(false),
      preconditioner(Preconditioner::Undefined),
      multigrid_data(MultigridData()),
      enable_cell_based_face_loops(false),

      // OUTPUT AND POSTPROCESSING
      print_input_parameters(true)
  {
  }

  /*
   *  This function is implemented in the header file of the test case
   *  that has to be solved.
   */
  void
  set_input_parameters();

  void
  check_input_parameters()
  {
    // SPATIAL DISCRETIZATION
    AssertThrow(degree_mapping > 0, ExcMessage("Invalid parameter."));
  }

  void
  print(ConditionalOStream & pcout)
  {
    pcout << std::endl << "List of input parameters:" << std::endl;

    // MATHEMATICAL MODEL
    print_parameters_mathematical_model(pcout);

    // PHYSICAL QUANTITIES
    print_parameters_physical_quantities(pcout);

    // SPATIAL DISCRETIZATION
    print_parameters_spatial_discretization(pcout);

    // SOLVER
    print_parameters_solver(pcout);

    // NUMERICAL PARAMETERS
    print_parameters_numerical_parameters(pcout);

    // OUTPUT AND POSTPROCESSING
    print_parameters_output_and_postprocessing(pcout);
  }

  void
  print_parameters_mathematical_model(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Mathematical model:" << std::endl;

    // right hand side
    print_parameter(pcout, "Right-hand side", right_hand_side);
  }

  void
  print_parameters_physical_quantities(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Physical quantities:" << std::endl;
  }

  void
  print_parameters_spatial_discretization(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Spatial Discretization:" << std::endl;

    print_parameter(pcout, "Polynomial degree of mapping", degree_mapping);

    print_parameter(pcout, "IP factor viscous term", IP_factor);

    print_parameter(pcout, "Element type", enum_to_string(spatial_discretization));
  }

  void
  print_parameters_solver(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Solver:" << std::endl;

    print_parameter(pcout, "Solver", enum_to_string(solver));

    solver_data.print(pcout);

    print_parameter(pcout, "Preconditioner", enum_to_string(preconditioner));

    if(preconditioner == Preconditioner::Multigrid)
      multigrid_data.print(pcout);
  }


  void
  print_parameters_numerical_parameters(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Numerical parameters:" << std::endl;

    print_parameter(pcout, "Enable cell-based face loops", enable_cell_based_face_loops);
  }


  void
  print_parameters_output_and_postprocessing(ConditionalOStream & pcout)
  {
    pcout << std::endl << "Output and postprocessing:" << std::endl;

    output_data.print(pcout, false /*steady*/);
  }


  /**************************************************************************************/
  /*                                                                                    */
  /*                                 MATHEMATICAL MODEL                                 */
  /*                                                                                    */
  /**************************************************************************************/

  // if the rhs f is unequal zero, set right_hand_side = true
  bool right_hand_side;

  /**************************************************************************************/
  /*                                                                                    */
  /*                              SPATIAL DISCRETIZATION                                */
  /*                                                                                    */
  /**************************************************************************************/

  // Polynomial degree of shape functions used for geometry approximation (mapping from
  // parameter space to physical space)
  unsigned int degree_mapping;

  // Symmetric interior penalty Galerkin (SIPG) discretization
  // interior penalty parameter scaling factor: default value is 1.0
  double                IP_factor;
  SpatialDiscretization spatial_discretization;



  /**************************************************************************************/
  /*                                                                                    */
  /*                                       SOLVER                                       */
  /*                                                                                    */
  /**************************************************************************************/

  // description: see enum declaration
  Solver solver;

  // solver data
  SolverData solver_data;
  bool       compute_performance_metrics;

  // description: see enum declaration
  Preconditioner preconditioner;

  // description: see declaration of MultigridData
  MultigridData multigrid_data;

  /**************************************************************************************/
  /*                                                                                    */
  /*                                NUMERICAL PARAMETERS                                */
  /*                                                                                    */
  /**************************************************************************************/

  // By default, the matrix-free implementation performs separate loops over all cells,
  // interior faces, and boundary faces. For a certain type of operations, however, it
  // is necessary to perform the face-loop as a loop over all faces of a cell with an
  // outer loop over all cells, e.g., preconditioners operating on the level of
  // individual cells (for example block Jacobi). With this parameter, the loop structure
  // can be changed to such an algorithm (cell_based_face_loops).
  bool enable_cell_based_face_loops;

  /**************************************************************************************/
  /*                                                                                    */
  /*                               OUTPUT AND POSTPROCESSING                            */
  /*                                                                                    */
  /**************************************************************************************/

  // print a list of all input parameters at the beginning of the simulation
  bool print_input_parameters;

  // writing output
  OutputData output_data;
};

} // namespace Poisson
#endif /* INCLUDE_LAPLACE_INPUT_PARAMETERS_H_ */
