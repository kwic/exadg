/*
 * boundary_descriptor.h
 *
 *  Created on: Aug 10, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_

using namespace dealii;

#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

#include "../../functionalities/function_interpolation.h"

namespace IncNS
{
//clang-format off
/*
 *
 *   Boundary conditions:
 *
 *   +----------------------+---------------------------+------------------------------------------------+
 *   |     example          |          velocity         |               pressure                         |
 *   +----------------------+---------------------------+------------------------------------------------+
 *   |     inflow, no-slip  |   Dirichlet(Mortar):      |  Neumann:                                      |
 *   |                      | prescribe g_u             | prescribe dg_u/dt in case of dual-splitting    |
 *   +----------------------+---------------------------+------------------------------------------------+
 *   |     symmetry         |   Symmetry:               |  Neumann:                                      |
 *   |                      | no BCs to be prescribed   | prescribe dg_u/dt = 0 in case of dual-splitting|
 *   +----------------------+---------------------------+------------------------------------------------+
 *   |     outflow          |   Neumann:                |  Dirichlet:                                    |
 *   |                      | prescribe F(u)*n          | prescribe g_p                                  |
 *   +----------------------+---------------------------+------------------------------------------------+
 *
 *   Divergence formulation: F(u) = F_nu(u) / nu = ( grad(u) + grad(u)^T )
 *   Laplace formulation:    F(u) = F_nu(u) / nu = grad(u)
 */
//clang-format on

enum class BoundaryTypeU
{
  Undefined,
  Dirichlet,
  DirichletMortar,
  Neumann,
  Symmetry
};

enum class BoundaryTypeP
{
  Undefined,
  Dirichlet,
  Neumann
};

template<int dim>
struct BoundaryDescriptorU
{
  // Dirichlet: prescribe all components of the velocity
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet_bc;

  // another type of Dirichlet boundary condition where the Dirichlet value comes
  // from the solution on another domain that is in contact with the actual domain
  // of interest at the given boundary (this type of Dirichlet boundary condition
  // is required for fluid-structure interaction problems)
  std::map<types::boundary_id, std::shared_ptr<FunctionInterpolation<1, dim>>> dirichlet_mortar_bc;

  // Neumann: prescribe all components of the velocity gradient in normal direction
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> neumann_bc;

  // Symmetry: For this boundary condition, the velocity normal to boundary is set to zero
  // (u*n=0) as well as the normal velocity gradient in tangential directions
  // (grad(u)*n - [(grad(u)*n)*n] n = 0). This is done automatically by the code.
  // The user does not have to prescribe a boundary condition, simply use ZeroFunction<dim>,
  // it is not relevant because this function will not be evaluated by the code.
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> symmetry_bc;

  // add more types of boundary conditions


  // return the boundary type
  inline DEAL_II_ALWAYS_INLINE //
    BoundaryTypeU
    get_boundary_type(types::boundary_id const & boundary_id) const
  {
    if(this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
      return BoundaryTypeU::Dirichlet;
    else if(this->dirichlet_mortar_bc.find(boundary_id) != this->dirichlet_mortar_bc.end())
      return BoundaryTypeU::DirichletMortar;
    else if(this->neumann_bc.find(boundary_id) != this->neumann_bc.end())
      return BoundaryTypeU::Neumann;
    else if(this->symmetry_bc.find(boundary_id) != this->symmetry_bc.end())
      return BoundaryTypeU::Symmetry;

    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));

    return BoundaryTypeU::Undefined;
  }

  inline DEAL_II_ALWAYS_INLINE //
    void
    verify_boundary_conditions(types::boundary_id const             boundary_id,
                               std::set<types::boundary_id> const & periodic_boundary_ids) const
  {
    unsigned int counter = 0;
    if(this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
      counter++;

    if(this->dirichlet_mortar_bc.find(boundary_id) != this->dirichlet_mortar_bc.end())
      counter++;

    if(this->neumann_bc.find(boundary_id) != this->neumann_bc.end())
      counter++;

    if(this->symmetry_bc.find(boundary_id) != this->symmetry_bc.end())
      counter++;

    if(periodic_boundary_ids.find(boundary_id) != periodic_boundary_ids.end())
      counter++;

    AssertThrow(counter == 1, ExcMessage("Boundary face with non-unique boundary type found."));
  }
};

template<int dim>
struct BoundaryDescriptorP
{
  // Dirichlet: prescribe pressure value
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet_bc;

  // Neumann: It depends on the chosen Navier-Stokes solver how this map has to be filled
  //
  //  - coupled solver: do nothing (one only has to discretize the pressure gradient for
  //                    this solution approach)
  //
  //  - pressure-correction: this solver always prescribes homogeneous Neumann BCs in the
  //                         pressure Poisson equation. Hence, no function has to be specified.
  //
  //  - dual splitting: Specify a Function<dim> with dim components for the boundary condition
  //                    dg_u/dt that has to be evaluated for the dual splitting scheme. But this
  //                    is only necessary if the parameter store_previous_boundary_values == false.
  //                    Otherwise, the code automatically determines the time derivative dg_u/dt
  //                    numerically and no boundary condition has to be set by the user.
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> neumann_bc;

  // add more types of boundary conditions


  // return the boundary type
  inline DEAL_II_ALWAYS_INLINE //
    BoundaryTypeP
    get_boundary_type(types::boundary_id const & boundary_id) const
  {
    if(this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
      return BoundaryTypeP::Dirichlet;
    else if(this->neumann_bc.find(boundary_id) != this->neumann_bc.end())
      return BoundaryTypeP::Neumann;

    AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));

    return BoundaryTypeP::Undefined;
  }

  inline DEAL_II_ALWAYS_INLINE //
    void
    verify_boundary_conditions(types::boundary_id const             boundary_id,
                               std::set<types::boundary_id> const & periodic_boundary_ids) const
  {
    unsigned int counter = 0;
    if(this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
      counter++;

    if(this->neumann_bc.find(boundary_id) != this->neumann_bc.end())
      counter++;

    if(periodic_boundary_ids.find(boundary_id) != periodic_boundary_ids.end())
      counter++;

    AssertThrow(counter == 1, ExcMessage("Boundary face with non-unique boundary type found."));
  }
};


} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_ */
