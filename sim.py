"""Code for Local density determines nuclear movements during syncytial
blastoderm formation in a cricket
"""
import numpy as np
import skfmm


SCALE = 1.0
out_dir = "/home/jordanhoffmann/sim_run/"


def make_nbhs():
  """List of neighbor directions.

     Args:
     Out:
      List of all points in a 3x3 grid, relative to the center.
  """
  nbhd = []
  for i in range(-1, 2):
      for j in range(-1, 2):
          for k in range(-1, 2):
              nbhd.append([i, j, k])
  nbhd = np.array(nbhd)
  new_array = [tuple(row) for row in nbhd]
  uniques = np.array(list(set(new_array)))
  return np.array(uniques)


def center_func(x):
  """Return a center height, fit based on data.

     Args:
      x: Position from -100 to 100.
     Out:
      Center (float)
  """
  return (0.065889993013125383591876982336543733 - \
          0.03527527490855846686867991479630291 * x + \
          0.001382483883967444270729485467086306 * x**2 + \
          0.00016103918870186853073804555958048468 * x**3 + \
          1.2923803007491026100287626748874814e-6 * x**4 - \
          3.1751183884855637492929700087251899e-7 * x**5 - \
          1.1103942353163466165900610075636078e-9 * x**6 + \
          3.3374901244531390674330155133784064e-10 * x**7 + \
          1.9336978449445920336193829468191267e-13 * x**8 - \
          2.1186853536155994639094406169744052e-13 * x**9 + \
          1.7918017199832011406356535593453922e-16 * x**10 + \
          8.6979225607065446225523603806916015e-17 * x**11 - \
          1.1925282593167411286548644745097342e-19 * x**12 - \
          2.3963826184042011884227247606825531e-20 * x**13 + \
          3.4165468357924512805775115653656522e-23 * x**14 + \
          4.5066783711521707109192253744895895e-24 * x**15 - \
          5.676840893007110929968759802952267e-27 * x**16 - \
          5.7844399222791372084271826301637292e-28 * x**17 + \
          5.8202337831119144319785630520023643e-31 * x**18 + \
          4.9704908929403045606176177105688512e-32 * x**19 - \
          3.6379119026498338675793426864803187e-35 * x**20 - \
          2.7261520758401733132218541055951668e-36 * x**21 + \
          1.273360485915671306013620988361208e-39 * x**22 + \
          8.5956237633974552807728882687054761e-41 * x**23 - \
          1.915325425798642016729727654752415e-44 * x**24 - \
          1.1803626059746943569401145361145106e-45 * x**25)


def rad_bent(x):
  """Return a radius, fit based on data.

     Args:
      x: Position from -100 to 100.
     Out:
      A radius (float)
  """
  return 1.040396156714501 * (-100. + x) * (100. + x) * \
      (-0.002240441472956154 + \
       4.278069012746345e-6 * x - 3.012545652944636e-7 * x**2 + \
       1.966685001231935e-9 * x**3 + 4.295866006332965e-10 * x**4 - \
       1.221956393922583e-11 * x**5 - 1.013482141443765e-12 * x**6 + \
       1.937031028506722e-14 * x**7 + 1.202390672111997e-15 * x**8 - \
       1.538230210366295e-17 * x**9 - 8.148363448241872e-19 * x**10 + \
       6.866261677305087e-21 * x**11 + 3.384371372733242e-22 * x**12 - \
       1.793521761010485e-24 * x**13 - 8.998793814444871e-26 * x**14 + \
       2.638206291748053e-28 * x**15 + 1.561280576750857e-29 * x**16 - \
       1.631625076960794e-32 * x**17 - 1.758145812716118e-33 * x**18 - \
       9.152166767640361e-37 * x**19 + 1.237984231742942e-37 * x**20 + \
       2.301407629890382e-40 * x**21 - 4.949852424225521e-42 * x**22 - \
       1.498466183821452e-44 * x**23 + 8.573404157603941e-47 * x**24 + \
       3.482311667439197e-49 * x**25)


def closest(vec, x, y, z):
  """Return a list of distances from x, y, z to points in vec.

     Args:
      vec: list of x, y, z coordinates
      x: x coordinate
      y: y coordinate
      z: z coordinate
     Out:
      L2 distance from each coordinate to vec
  """
  coord = np.array([x, y, z])
  return np.linalg.norm(coord - vec)


def downsample(shell):
  """Downsample function used for saving data.

     Args:
      shell: size in x
     Out:
      Either shell or a random selection of points in shell
  """
  length = len(shell)
  if length < 2400:
    return shell
  else:
    permuted = np.random.permutation(range(length))[:2399]
    smaller = np.array([shell[j] for j in permuted])
    return smaller


def setup(dx, dy, dz, number):
  """Setup the simulation.

     Args:
      dx: size in x
      dy: size in y
      dz: size in z
      number: how many nuclei to add
     Out:
      locations: array of locations
      phi: initialise a level set with 0s around each nucleus
  """
  locations = np.zeros((number, 3))
  for i in range(number):
    locations[i][0] = 80. * SCALE + 5 * np.random.rand() - 2.5
    locations[i][1] = 45. * SCALE + 2.5 * np.random.rand() - 1.25
    locations[i][2] = 31. * SCALE + 2.5 * np.random.rand() - 1.25
    loc = [locations[i][0], locations[i][1], locations[i][2]]
    print('Set nucleus to : ', loc)
  phi = np.ones((dx, dy, dz))
  for i in range(number):
    rx = int(locations[i][0])
    ry = int(locations[i][1])
    rz = int(locations[i][2])
    phi[rx, ry, rz] = -1
  return locations, phi


def refine_shells(locations, shells, cents, nbhs, interior):
  """Refine the shells created.

     Args:
      locations: Where the locations are
      cents: Where the point of emination is
      nbhs: The nbhs
      interior: how many nuclei to add
     Out:
      shells_new: The refined, new shells
  """
  cents = np.array(cents)
  num = len(locations)
  shells_new = [[] for i in range(num)]
  for i in range(num):
    me = locations[i]
    Near_1 = me + cents[i]
    Anti_1 = me - cents[i]
    for j in range(len(shells[i])):
      them = shells[i][j]
      dis_1 = np.linalg.norm(them - Near_1)
      dis_2 = np.linalg.norm(them - Anti_1)
      if dis_1 < 1.055 * dis_2:
        shells_new[i].append(shells[i][j])
  return shells_new


def divide_new(locations, divide_at, thetwo):
  """Setup the division times.

     Args:
      locations:
      divide_at:
      thetwo:
     Out:
      divide_at:
  """
  rad_max = 15
  for i in range(2):
    me = locations[thetwo[i]]
    count = 0
    for j in range(len(locations)):
      if np.linalg.norm(me - locations[j]) < rad_max:
        count += 1
      else:
        den = (0.1 * rad_max)
        count += np.exp(-(np.linalg.norm(me - locations[j]) - rad_max) / den)
    if count <= 3:
      divide_at[thetwo[i]] = 52 + int(np.random.normal(0, 1.))
    elif count <= 4:
      divide_at[thetwo[i]] = 54 + int(np.random.normal(0, 2.1))
    elif count <= 5:
      divide_at[thetwo[i]] = 55 + int(np.random.normal(0, 2.1))
    elif count <= 6:
      divide_at[thetwo[i]] = 60 + int(np.random.normal(0, 2.1))
    elif count <= 8:
      divide_at[thetwo[i]] = 67 + int(np.random.normal(0, 2.1))
    elif count <= 10:
      divide_at[thetwo[i]] = 72 + int(np.random.normal(0, 2.2))
    elif count <= 13:
      divide_at[thetwo[i]] = 79 + int(np.random.normal(0, 2.5))
    elif count <= 15:
      divide_at[thetwo[i]] = 86 + int(np.random.normal(0, 2.5))
    elif count <= 17:
      divide_at[thetwo[i]] = 96 + int(np.random.normal(0, 3.))
    elif count <= 20:
      divide_at[thetwo[i]] = 104 + int(np.random.normal(0, 3.))
    elif count <= 23:
      divide_at[thetwo[i]] = 122 + int(np.random.normal(0, 3.5))
    elif count <= 25:
      divide_at[thetwo[i]] = 151 + int(np.random.normal(0, 3.5))
    elif count <= 30:
      divide_at[thetwo[i]] = 183 + int(np.random.normal(0, 3.5))
    elif count <= 35:
      divide_at[thetwo[i]] = 193 + int(np.random.normal(0, 4.))
    elif count <= 40:
      divide_at[thetwo[i]] = 222 + int(np.random.normal(0, 4.))
    else:
      divide_at[thetwo[i]] = 347 + int(np.random.normal(0, 5.5))
  return divide_at


if __name__ == "__main__":
  print("Running sim.py")
  interior = np.zeros((202, 92, 92))
  for time_step in range(2 * 400 + 1):
    print('I am on time step: ' + str(time_step))

    NC = len(locations)
    print('Length of locations is: ' + str(NC))
    shell_bound = tsd_Shell * micro_vel
    velocity = speed(dx, dy, dz)
    t = skfmm.travel_time(phi, velocity, dx=1.0, order=1)
    if time_step < 100:
      shells = make_shell(locations, t, shell_bound, UPPERMAX)
      shells = refine_shells(locations, shells, cents, nbhds, interior)
      cents = shift_cents(cents, locations, shells)
      steps = step2(locations, shells, speed_ratio, tsd, tsd, interior,
                    tsd, alpha_middle, fall, time_step, rank)
    else:
      shells = make_shell(locations, t, shell_bound, UPPERMAX)
      shells = refine_shells(locations, shells, cents, nbhds, interior)
      cents = shift_cents(cents, locations, shells)
      steps = step2(locations, shells, speed_ratio, tsd, tsd, interior, tsd,
                    alpha_surface, fall, time_step, rank)
    locations = update_positions(locations, steps)
    phi = update_phi(dx, dy, dz, locations, cents)
    tsd_Shell += 1
    local_fates = np.zeros(len(locations))
    for i in range(len(locations)):
      tsd[i] += 1
      local_fates[i] = i
      if tsd[i] == divide_at[i]:
        local_fates = np.append(local_fates, i)
        locations = divide2(locations, i)
        rr = np.random.randint(26)
        cents[i] = nbhds[rr]
        cents.append(-1.0 * cents[i])
        tsd[i] = 0
        tsd[len(cents) - 1] = 0
        thetwo = [i, len(cents) - 1]
        divide_at = divide_new(locations, divide_at, thetwo)
        phi = update_phi(dx, dy, dz, locations, cents)
