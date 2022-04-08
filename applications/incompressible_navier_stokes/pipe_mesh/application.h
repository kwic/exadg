/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2022 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PIPE_MESH_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PIPE_MESH_H_

#include <deal.II/grid/grid_in.h>

namespace ExaDG
{
namespace IncNS
{

template<int dim>
class InflowProfileVelocity : public dealii::Function<dim>
{
public:
  InflowProfileVelocity(double const max_velocity, double const radius, double const x, double const y)
    : dealii::Function<dim>(dim, 0.0), max_velocity(max_velocity), radius(radius), x(x), y(y)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const
  {
    double result = 0.0;

    if(component == 2) {
      double dx = p[0] - x;
      double dy = p[1] - y;
      double fac = (dx*dx + dy*dy) / (radius*radius);
      result = std::max(0.0, max_velocity * (1.0 - fac));
    }
    return result;
  }

private:
  double const max_velocity, radius, x, y;
};


template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Application");
      prm.add_parameter("MaxInflowVelocity", input_max_vel, "Maximum inflow velocity");
      prm.add_parameter("InflowRadius", input_radius, "Radius of parabolic inflow profile");
      prm.add_parameter("InflowPosX", input_x, "X position of parabolic inflow profile center");
      prm.add_parameter("InflowPosY", input_y, "Y position of parabolic inflow profile center");
      prm.add_parameter("InputMesh", input_mesh_path, "Path to mesh VTU file");
    prm.leave_subsection();
    // clang-format on
  }

private:
  void
  parse_parameters() final
  {
    ApplicationBase<dim, Number>::parse_parameters();
    max_velocity = 2.0 * input_max_vel;
  }

  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type                   = ProblemType::Unsteady;
    this->param.equation_type                  = EquationType::NavierStokes;
    this->param.formulation_viscous_term       = formulation_viscous_term;
    this->param.formulation_convective_term    = FormulationConvectiveTerm::ConvectiveFormulation;
    this->param.use_outflow_bc_convective_term = false;
    this->param.right_hand_side                = false;


    // PHYSICAL QUANTITIES
    this->param.start_time = start_time;
    this->param.end_time   = end_time;
    this->param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    this->param.solver_type                   = SolverType::Unsteady;
    this->param.temporal_discretization       = TemporalDiscretization::BDFDualSplittingScheme;
    this->param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Explicit;
    this->param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    this->param.adaptive_time_stepping        = true;
    this->param.max_velocity                  = max_velocity;
    this->param.cfl                           = 2.0e-1;
    this->param.time_step_size                = 1.0e-1;
    this->param.order_time_integrator         = 2;    // 1; // 2; // 3;
    this->param.start_with_low_order          = true; // true; // false;

    this->param.convergence_criterion_steady_problem =
      ConvergenceCriterionSteadyProblem::SolutionIncrement; // ResidualSteadyNavierStokes;
    this->param.abs_tol_steady = 1.e-12;
    this->param.rel_tol_steady = 1.e-8;

    // output of solver information
    this->param.solver_info_data.interval_time =
      (this->param.end_time - this->param.start_time) / 10;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = this->param.degree_u;
    this->param.degree_p                = DegreePressure::MixedOrder;

    // convective term
    if(this->param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      this->param.upwind_factor = 0.5;

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // PROJECTION METHODS

    // pressure Poisson equation
    this->param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    this->param.solver_data_pressure_poisson    = SolverData(1000, 1.e-20, 1.e-6, 100);
    this->param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

    // projection step
    this->param.solver_projection         = SolverProjection::CG;
    this->param.solver_data_projection    = SolverData(1000, 1.e-20, 1.e-12);
    this->param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;


    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    this->param.order_extrapolation_pressure_nbc =
      this->param.order_time_integrator <= 2 ? this->param.order_time_integrator : 2;

    // viscous step
    this->param.solver_viscous         = SolverViscous::CG;
    this->param.solver_data_viscous    = SolverData(1000, 1.e-20, 1.e-6);
    this->param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; // Multigrid;

    // PRESSURE-CORRECTION SCHEME

    // formulation
    this->param.order_pressure_extrapolation = 1;
    this->param.rotational_formulation       = true;

    // momentum step

    // Newton solver
    this->param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-14, 1.e-6);

    // linear solver
    this->param.solver_momentum                = SolverMomentum::GMRES;
    this->param.solver_data_momentum           = SolverData(1e4, 1.e-20, 1.e-6, 100);
    this->param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    this->param.update_preconditioner_momentum = false;


    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    this->param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-10, 1.e-6);

    // linear solver
    this->param.solver_coupled      = SolverCoupled::FGMRES; // GMRES;
    this->param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-2, 200);

    // preconditioning linear solver
    this->param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    this->param.update_preconditioner_coupled = true;

    // preconditioner velocity/momentum block
    this->param.preconditioner_velocity_block          = MomentumPreconditioner::Multigrid;
    this->param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;
    this->param.multigrid_data_velocity_block.smoother_data.smoother =
      MultigridSmoother::Chebyshev; // Jacobi; //Chebyshev; //GMRES;
    this->param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi; // PointJacobi; //BlockJacobi;
    this->param.multigrid_data_velocity_block.smoother_data.iterations        = 5;
    this->param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
    this->param.multigrid_data_velocity_block.coarse_problem.solver =
      MultigridCoarseGridSolver::GMRES;

    // preconditioner Schur-complement block
    this->param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
  }

  void
  create_grid() final
  {
    dealii::GridIn<dim> grid_in;
    grid_in.attach_triangulation(*this->grid->triangulation);
    std::ifstream input_file(input_mesh_path);
    grid_in.read_vtk(input_file);

    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // fill boundary descriptor velocity

    // no-slip walls
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));

    // inflow
    this->boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(1, new InflowProfileVelocity<dim>(input_max_vel, input_radius, input_x, input_y)));

    // outflow
    this->boundary_descriptor->velocity->neumann_bc.insert(
      pair(2, new dealii::Functions::ZeroFunction<dim>(dim)));

    // fill boundary descriptor pressure

    // no-slip walls
    this->boundary_descriptor->pressure->neumann_bc.insert(0);

    // inflow
    this->boundary_descriptor->pressure->neumann_bc.insert(1);

    // outflow
    this->boundary_descriptor->pressure->dirichlet_bc.insert(
      pair(2, new dealii::Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output              = this->output_parameters.write;
    pp_data.output_data.directory                 = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename                  = this->output_parameters.filename;
    pp_data.output_data.start_time                = start_time;
    pp_data.output_data.interval_time             = (end_time - start_time) / 10;
    pp_data.output_data.write_vorticity           = true;
    pp_data.output_data.write_divergence          = true;
    pp_data.output_data.write_velocity_magnitude  = true;
    pp_data.output_data.write_vorticity_magnitude = true;
    pp_data.output_data.write_processor_id        = true;
    pp_data.output_data.write_q_criterion         = true;
    pp_data.output_data.degree                    = this->param.degree_u;
    pp_data.output_data.write_higher_order        = true;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  std::string input_mesh_path;
  double input_max_vel = 1.0;
  double input_radius = 1.0;
  double input_x = 1.0;
  double input_y = 1.0;

  FormulationViscousTerm const formulation_viscous_term =
    FormulationViscousTerm::LaplaceFormulation;

  double max_velocity = input_max_vel;
  double const viscosity    = 1.0e-1;

  double const H = 2.0;
  double const L = 4.0;

  double const start_time = 0.0;
  double const end_time   = 100.0;
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PIPE_MESH_H_ */
