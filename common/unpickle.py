#!/usr/bin/env python

class Params(object):
  def __init__(self, **kwds):
    self.__dict__.update(kwds)

def parse_file(fname):
  with open(fname, 'r') as f:
    def get_scalar_array(convert, n):
      return [convert(f.next()) for x in xrange(n)]
    def get_vector_array(convert, n):
      return [[convert(f.next()),
               convert(f.next()),
               convert(f.next())] for x in xrange(n)]
    # constants
    dt         = float(f.next())
    nktv2p     = float(f.next())
    ntype      = int(f.next())
    yeff       = [float(f.next()) for x in xrange(ntype*ntype)]
    geff       = [float(f.next()) for x in xrange(ntype*ntype)]
    betaeff    = [float(f.next()) for x in xrange(ntype*ntype)]
    coeffFrict = [float(f.next()) for x in xrange(ntype*ntype)]
    # per-particle data
    nnode = int(f.next())
    pos = get_vector_array(float, nnode)
    v = get_vector_array(float, nnode)
    omega = get_vector_array(float, nnode)
    radius = get_scalar_array(float, nnode)
    mass = get_scalar_array(float, nnode)
    ty = get_scalar_array(int, nnode)
    force = get_vector_array(float, nnode)
    torque = get_vector_array(float, nnode)
    # per-neighbor data
    nedge = int(f.next())
    edge = [(int(f.next()), int(f.next())) for x in xrange(nedge)]
    shear = get_vector_array(float, nedge)
    # expected results
    expected_force = get_vector_array(float, nnode)
    expected_torque = get_vector_array(float, nnode)
    expected_shear = get_vector_array(float, nedge)
    return Params(dt=dt, nktv2p=nktv2p,
        ntype=ntype,
          yeff=yeff, geff=geff, betaeff=betaeff, coeffFrict=coeffFrict,
        nnode=nnode,
          x=pos, v=v, omega=omega, radius=radius, mass=mass, ty=ty,
          force=force, torque=torque,
        nedge=nedge,
          edge=edge, shear=shear,
        expected_force=expected_force,
        expected_torque=expected_force,
        expected_shear=expected_shear)
