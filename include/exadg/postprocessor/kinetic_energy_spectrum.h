/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_POSTPROCESSOR_KINETIC_ENERGY_SPECTRUM_H_
#define INCLUDE_EXADG_POSTPROCESSOR_KINETIC_ENERGY_SPECTRUM_H_

// deal.II
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
// forward declaration
class DealSpectrumWrapper;

struct KineticEnergySpectrumData
{
  KineticEnergySpectrumData()
    : calculate(false),
      write_raw_data_to_files(false),
      do_fftw(true),
      start_time(0.0),
      calculate_every_time_steps(-1),
      calculate_every_time_interval(-1.0),
      directory("output/"),
      filename("energy_spectrum"),
      clear_file(true),
      degree(0),
      evaluation_points_per_cell(0),
      exploit_symmetry(false),
      n_cells_1d_coarse_grid(1),
      refine_level(0),
      length_symmetric_domain(dealii::numbers::PI)
  {
  }

  void
  print(dealii::ConditionalOStream & pcout)
  {
    if(calculate == true)
    {
      pcout << std::endl << "  Calculate kinetic energy spectrum:" << std::endl;
      print_parameter(pcout, "Write raw data to files", write_raw_data_to_files);
      print_parameter(pcout, "Do FFTW", do_fftw);
      print_parameter(pcout, "Start time", start_time);
      if(calculate_every_time_steps >= 0)
        print_parameter(pcout, "Calculate every timesteps", calculate_every_time_steps);
      if(calculate_every_time_interval >= 0.0)
        print_parameter(pcout, "Calculate every time interval", calculate_every_time_interval);
      print_parameter(pcout, "Directory of output files", directory);
      print_parameter(pcout, "Filename", filename);
      print_parameter(pcout, "Clear file", clear_file);

      print_parameter(pcout, "Evaluation points per cell", evaluation_points_per_cell);

      print_parameter(pcout, "Exploit symmetry", exploit_symmetry);
      if(exploit_symmetry)
      {
        print_parameter(pcout, "n_cells_1d_coarse_grid", n_cells_1d_coarse_grid);
        print_parameter(pcout, "refine_level", refine_level);
        print_parameter(pcout, "length_symmetric_domain", length_symmetric_domain);
      }
    }
  }

  bool   calculate;
  bool   write_raw_data_to_files;
  bool   do_fftw;
  double start_time;
  int    calculate_every_time_steps;
  double calculate_every_time_interval;

  // these parameters are only relevant if do_fftw = true
  std::string directory;
  std::string filename;
  bool        clear_file;

  unsigned int degree;
  unsigned int evaluation_points_per_cell;

  // exploit symmetry for Navier-Stokes simulation and mirror dof-vector
  // according to Taylor-Green symmetries for evaluation of energy spectrum.
  bool         exploit_symmetry;
  unsigned int n_cells_1d_coarse_grid;
  unsigned int refine_level;
  double       length_symmetric_domain;
};

template<int dim, typename Number>
class KineticEnergySpectrumCalculator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  KineticEnergySpectrumCalculator(MPI_Comm const & mpi_comm);

  void
  setup(dealii::MatrixFree<dim, Number> const & matrix_free_data_in,
        dealii::DoFHandler<dim> const &         dof_handler_in,
        KineticEnergySpectrumData const &       data_in);

  void
  evaluate(VectorType const & velocity, double const & time, int const & time_step_number);

private:
  bool
  needs_to_be_evaluated(double const time, unsigned int const time_step_number);

  void
  do_evaluate(VectorType const & velocity, double const time);

  MPI_Comm const mpi_comm;

  bool                      clear_files;
  KineticEnergySpectrumData data;
  unsigned int              counter;
  bool                      reset_counter;
  unsigned int const        precision = 12;

  std::shared_ptr<DealSpectrumWrapper> deal_spectrum_wrapper;

  dealii::SmartPointer<dealii::DoFHandler<dim> const> dof_handler;

  std::shared_ptr<VectorType>                 velocity_full;
  std::shared_ptr<dealii::Triangulation<dim>> tria_full;
  std::shared_ptr<dealii::FESystem<dim>>      fe_full;
  std::shared_ptr<dealii::DoFHandler<dim>>    dof_handler_full;
};
} // namespace ExaDG

#endif /* INCLUDE_EXADG_POSTPROCESSOR_KINETIC_ENERGY_SPECTRUM_H_ */
