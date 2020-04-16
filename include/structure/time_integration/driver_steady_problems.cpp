/*
 * driver_steady_problems.cpp
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#include "driver_steady_problems.h"

namespace Structure
{
template<int dim, typename Number>
DriverSteady<dim, Number>::DriverSteady(
  std::shared_ptr<Operator<dim, Number>>      operator_in,
  std::shared_ptr<PostProcessor<dim, Number>> postprocessor_in,
  InputParameters const &                     param_in,
  MPI_Comm const &                            mpi_comm_in)
  : pde_operator(operator_in),
    postprocessor(postprocessor_in),
    param(param_in),
    mpi_comm(mpi_comm_in),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm_in) == 0),
    computing_times(1)
{
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::setup()
{
  // initialize global solution vectors (allocation)
  initialize_vectors();

  // initialize solution by interpolation of initial data
  initialize_solution();
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::solve_problem()
{
  postprocessing();

  solve();

  postprocessing();
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::initialize_vectors()
{
  pde_operator->initialize_dof_vector(solution);
  pde_operator->initialize_dof_vector(rhs_vector);
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::initialize_solution()
{
  double time = 0.0;
  pde_operator->prescribe_initial_conditions(solution, time);
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::solve()
{
  pcout << std::endl << "Solving steady state problem ..." << std::endl;

  Timer timer;
  timer.restart();

  unsigned int N_iter_nonlinear = 0;
  unsigned int N_iter_linear    = 0;

  if(param.large_deformation) // nonlinear problem
  {
    VectorType const_vector;
    pde_operator->solve_nonlinear(solution,
                                  const_vector,
                                  /* time */ 0.0,
                                  /* update_preconditioner = */ true,
                                  N_iter_nonlinear,
                                  N_iter_linear);
  }
  else // linear problem
  {
    // calculate right-hand side vector
    pde_operator->compute_rhs_linear(rhs_vector);

    N_iter_linear = pde_operator->solve_linear(solution,
                                               rhs_vector,
                                               /* time */ 0.0,
                                               /* update_preconditioner = */ true);
  }

  computing_times[0] += timer.wall_time();

  // solver info output
  if(param.large_deformation)
  {
    double N_iter_linear_avg =
      (N_iter_nonlinear > 0) ? double(N_iter_linear) / double(N_iter_nonlinear) : N_iter_linear;

    pcout << std::endl
          << "Solve nonlinear problem:" << std::endl
          << "  Newton iterations:      " << std::setw(12) << std::right << N_iter_nonlinear
          << std::endl
          << "  Linear iterations (avg):" << std::setw(12) << std::scientific
          << std::setprecision(4) << std::right << N_iter_linear_avg << std::endl
          << "  Linear iterations (tot):" << std::setw(12) << std::scientific
          << std::setprecision(4) << std::right << N_iter_linear << std::endl
          << "  Wall time [s]:          " << std::setw(12) << std::scientific
          << std::setprecision(4) << computing_times[0] << std::endl;
  }
  else
  {
    pcout << std::endl
          << "Solve linear problem:" << std::endl
          << "  Iterations:   " << std::setw(12) << std::right << N_iter_linear << std::endl
          << "  Wall time [s]:" << std::setw(12) << std::scientific << std::setprecision(4)
          << computing_times[0] << std::endl;
  }

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::postprocessing() const
{
  postprocessor->do_postprocessing(solution);
}

template<int dim, typename Number>
void
DriverSteady<dim, Number>::get_wall_times(std::vector<std::string> & name,
                                          std::vector<double> &      wall_time) const
{
  name.resize(1);
  std::vector<std::string> names = {"(Non-)linear system"};
  name                           = names;

  wall_time.resize(1);
  wall_time[0] = computing_times[0];
}

template class DriverSteady<2, float>;
template class DriverSteady<2, double>;

template class DriverSteady<3, float>;
template class DriverSteady<3, double>;

} // namespace Structure
