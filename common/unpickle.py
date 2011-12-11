#!/usr/bin/env python

import struct

class Params(object):
  def __init__(self, **kwds):
    self.__dict__.update(kwds)

def parse_file(fname):
  with open(fname, 'rb') as f:
    nbytes_of_fmt = { '<i': 4, '<d': 8 }
    def get_scalar_array(fmt, n):
      return [ struct.unpack(fmt, f.read(nbytes_of_fmt[fmt]))[0] for x in xrange(n) ]
    def get_vector_array(fmt, n):
      return [[struct.unpack(fmt, f.read(nbytes_of_fmt[fmt]))[0],
               struct.unpack(fmt, f.read(nbytes_of_fmt[fmt]))[0],
               struct.unpack(fmt, f.read(nbytes_of_fmt[fmt]))[0]] for x in xrange(n)]
    # magic number
    EXPECTED_MAGIC = 0xDEADBEEF
    magic, = struct.unpack('<I', f.read(4))
    if magic != EXPECTED_MAGIC:
      print "Error with file [%s]: magic number is [%x] expecting [%x]\n" % (fname, magic, EXPECTED_MAGIC)

    # constants
    dt, nktv2p = struct.unpack('<2d', f.read(16))
    ntype,     = struct.unpack('<i', f.read(4))
    yeff       = [ struct.unpack('<d', f.read(8))[0] for x in xrange(ntype*ntype)]
    geff       = [ struct.unpack('<d', f.read(8))[0] for x in xrange(ntype*ntype)]
    betaeff    = [ struct.unpack('<d', f.read(8))[0] for x in xrange(ntype*ntype)]
    coeffFrict = [ struct.unpack('<d', f.read(8))[0] for x in xrange(ntype*ntype)]
    # per-particle data
    nnode, = struct.unpack('<i', f.read(4))
    pos = get_vector_array('<d', nnode)
    v = get_vector_array('<d', nnode)
    omega = get_vector_array('<d', nnode)
    radius = get_scalar_array('<d', nnode)
    mass = get_scalar_array('<d', nnode)
    ty = get_scalar_array('<i', nnode)
    force = get_vector_array('<d', nnode)
    torque = get_vector_array('<d', nnode)
    # per-neighbor data
    nedge = struct.unpack('<i', f.read(4))[0]
    edge = [(struct.unpack('<i', f.read(4))[0], struct.unpack('<i', f.read(4))[0]) for x in xrange(nedge)]
    shear = get_vector_array('<d', nedge)
    # expected results
    expected_force = get_vector_array('<d', nnode)
    expected_torque = get_vector_array('<d', nnode)
    expected_shear = get_vector_array('<d', nedge)
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
