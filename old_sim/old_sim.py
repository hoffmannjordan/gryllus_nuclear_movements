import numpy as np
import skfmm
from scipy.optimize import minimize
from mpi4py import MPI


SCALE = 1.0


def make_nbhs():
  """List of neighbor directions."""
  nbhd = []
  for i in range(-1, 2):
      for j in range(-1, 2):
          for k in range(-1, 2):
              nbhd.append([i, j, k])
  nbhd = np.array(nbhd)
  new_array = [tuple(row) for row in nbhd]
  uniques = np.array(list(set(new_array)))
  return np.array(uniques)


def con(vec):
  """Constraint for optimisation is using an ellipsoidal shape."""
  x = vec[0] - 100 * SCALE
  y = vec[1] - 25 * SCALE
  z = vec[2] - 25 * SCALE
  rx = 100.0 * SCALE
  ry = 25.0 * SCALE
  rz = 25.0 * SCALE
  return (x**2 / rx**2 + y**2 / ry**2 + z**2 / rz**2) - 0.995


def closest(vec, x, y, z):
  me = np.array([x, y, z])
  return np.linalg.norm(me - vec)


def project(x, y, z):
  vec = [100. * SCALE, 0., 0.]
  cons = {'type': 'eq', 'fun': con}
  result = minimize(closest, vec, constraints=cons, args=(x, y, z),
                    method='SLSQP')
  closest_point = result.x
  return closest_point


def setup(dx, dy, dz, number):
  locations = np.zeros((number, 3))
  for i in range(number):
    # locations[i][0] = 80.*SCALE+ 5*np.random.rand()-2.5
    locations[i][0] = 80. * SCALE + 5 * np.random.rand() - 2.5
    locations[i][1] = 45. * SCALE + 2.5 * np.random.rand() - 1.25
    locations[i][2] = 31. * SCALE + 2.5 * np.random.rand() - 1.25
    print('Set nucleus to : ', locations[i][0], locations[i][1], locations[i][2])
    print(shell(locations[i][0], locations[i][1], locations[i][2]))
  phi = np.ones((dx, dy, dz))
  for i in range(number):
    rx = int(locations[i][0])
    ry = int(locations[i][1])
    rz = int(locations[i][2])
    phi[rx, ry, rz] = -1
  return locations, phi


def speed(dx, dy, dz):
  return np.random.uniform(low=0.6, high=1., size=dx * dy * dz).reshape((dx, dy, dz))


def find_closest_id(locations, x, y, z):
  options = np.zeros(len(locations))
  for i in range(len(locations)):
    options[i] = np.linalg.norm([locations[i] - np.array([x, y, z])])
  return np.argmin(options)


def find_boundaries(t, ratio, number, locations):
  t = t.astype(int)
  [xs, ys, zs] = np.where(t == speed_ratio)
  boundary_list = [[] for i in range(number)]
  for i in range(len(xs)):
    closest = find_closest_id(locations, xs[i], ys[i], zs[i])
    boundary_list[closest].append([xs[i], ys[i], zs[i]])
  print('Length of 1: ', len(boundary_list[0]))
  [xs, ys, zs] = np.where(t == (speed_ratio - 1))
  for i in range(len(xs)):
    closest = find_closest_id(locations, xs[i], ys[i], zs[i])
    boundary_list[closest].append([xs[i], ys[i], zs[i]])
  return boundary_list


def move(locations, boundary_list, vel):
  moves = np.zeros(np.shape(locations))
  for i in range(len(locations)):
    movement = np.zeros(3)
    me = locations[i]
    them = boundary_list[i]
    for j in range(len(them)):
      distance = np.linalg.norm(me - them[j])
      vector = them[j] - me
      weight = np.e**(distance / 10)
      movement += weight * vector
    movement = movement / np.linalg.norm(movement)
    moves[i] += movement
  return moves


def update_positions(locations, moves):
  new = locations + moves
  for i in range(len(new)):
    if not shell(new[i][0], new[i][1], new[i][2]):
      print('Moved one back in.')
      x = new[i][0]
      y = new[i][1]
      z = new[i][2]
      nearest_on_surface = project_con(x - 100.0 * SCALE, y - 41.0 * SCALE, z - 41.0 * SCALE)
      nearest_on_surface += [100. * SCALE, 41. * SCALE, 41. * SCALE]

      new[i] = nearest_on_surface
      print('Started at: ', new[i])
  return new


def update_phi(dx, dy, dz, locations, cents):
  phi = np.ones((dx, dy, dz))
  for i in range(len(locations)):
    rx = int(locations[i][0] + cents[i][0])
    ry = int(locations[i][1] + cents[i][1])
    rz = int(locations[i][2] + cents[i][2])
    if ry >= 91:
      ry = 91
    if rz >= 91:
      rz = 91
    phi[rx, ry, rz] = -1
  return phi


def make_shell(locations, t, upper, UPPERMAX):
  upper = UPPERMAX
  px, py, pz = np.where(t <= upper)
  shells = [[] for i in range(len(locations))]
  for i in range(len(px)):
    start = np.array([px[i], py[i], pz[i]])
    if shell(px[i], py[i], pz[i]) == True:
      x = start - locations
      norms = np.linalg.norm(x, axis=1)
      min_p = np.argmin(norms)
      shells[min_p].append(start)
  return shells


def sigmoid(r, all):
  return 1.0 - 1.0 / (1.0 + np.e**(-r / fall))


def project_move(x, y, z):
  vec = [100. * SCALE, 0., 0.]
  cons = {'type': 'eq', 'fun': con2}
  result = minimize(closest, vec, constraints=cons, args=(x, y, z), method='SLSQP')
  closest_point = result.x
  return closest_point + [100. * SCALE, 25. * SCALE, 25. * SCALE]


def project_con(x, y, z):
  vec = [100. * SCALE, 0., 0.]
  cons = {'type': 'eq', 'fun': con_con}
  result = minimize(closest, vec, constraints=cons, args=(x, y, z), method='SLSQP')
  closest_point = result.x
  return closest_point


def con_con(vec):
  x = vec[0]
  y = vec[1]
  z = vec[2]
  center = center_func(x)
  max_dist = rad_bent(x)
  rr = np.sqrt((y - center) * (y - center) + z * z)
  return max_dist - rr


def project_move_sphere(x, y, z):
  #vec = [100.*SCALE,0.,0.]
  vec = [100. * SCALE, 0., 0.]
  cons = {'type':'eq', 'fun': con_sphere}
  result = minimize(closest, vec, constraints=cons, args=(x, y, z), method='SLSQP')
  closest_point = result.x
  return closest_point + [100. * SCALE, 25. * SCALE, 25. * SCALE]


def con_sphere(vec):
  x = vec[0] / (100. * SCALE)
  y = vec[1] / (25. * SCALE)
  z = vec[2] / (25. * SCALE)
  rx = 100. * SCALE / (100. * SCALE)
  ry = 25. * SCALE / (25. * SCALE)
  rz = 25. * SCALE / (25. * SCALE)
  return (x**2 / rx**2 + y**2 / ry**2 + z**2 / rz**2) - 0.995


def closest_sphere(vec, x, y, z):
  me = np.array([x, y, z])
  return np.linalg.norm(me - vec)


def step2(locations, shells, speed_ratio, velocity, velterm, interior, tsd, alpha, fall, tt, rank):
  showQ = True
  if showQ:
    show_interior = np.zeros(np.shape(interior))
  rand_weight = 0.5
  stepps = np.zeros(np.shape(locations))
  for i in range(len(locations)):
    me = locations[i]
    for_projection = me - [100. * SCALE, 31. * SCALE, 31. * SCALE]
    nearest_on_surface = project_con(for_projection[0], for_projection[1], for_projection[2])
    nearest_on_surface += [100. * SCALE, 31. * SCALE, 31. * SCALE]
    move_to_surface = nearest_on_surface - me
    move_to_surface = move_to_surface / np.linalg.norm(move_to_surface)
    time_since_division = tsd[i]
    extra = np.random.normal(0, 0.2 * np.sqrt(len(locations)))
    cutoff = 37
    if time_since_division < cutoff:
      me = locations[i]
      mean_coordinate = np.mean(shells[i], axis=0)
      diff = mean_coordinate - me
      norm = np.linalg.norm(diff)
      movement = np.zeros(3)
      for them in shells[i]:
        XX = int((them[0]))
        YY = int((them[1]))
        ZZ = int((them[2]))
        interior_weight = interior[XX, YY, ZZ]
        distance = np.ceil(np.linalg.norm(me - them))
        vec_tmp = them - me
        rrr = np.random.rand()
        if len(locations) >= min(np.random.normal(192, 50), 256) and np.linalg.norm(nearest_on_surface - me) > 1.5 and rrr < 0.75:
          vec_tmp = 0.25 * vec_tmp + 0.75 * move_to_surface
          vec_tmp = vec_tmp / np.linalg.norm(vec_tmp)
        vec_tmp /= np.linalg.norm(vec_tmp)
        vec_tmp = vec_tmp
        vec_tmp /= np.linalg.norm(vec_tmp)
        movement += alpha * 1.0 / (distance**2) * interior_weight * vec_tmp
        if showQ:
          show_interior[XX, YY, ZZ] = 1.0 / (distance**2) * interior_weight

      rand_add = np.array([2. * np.random.rand() - 1., 2. * np.random.rand() - 1., 2. * np.random.rand() - 1.])
      rand_add = rand_add / np.linalg.norm(rand_add) + rand_add
      RW = 0.5
      movement = movement + 1.5 * (10.0**(-0.75 + -2. * np.random.rand())) * np.linalg.norm(movement) * rand_add
      stepps[i] = movement
      print('Ballistic size is: ', np.linalg.norm(movement))
    else:
      rs = np.array([np.random.rand() * 2 - 1, np.random.rand() * 2 - 1, np.random.rand() * 2 - 1])
      rs = rs / np.linalg.norm(rs)
      stepps[i] = 3.0 * (0.4 * 10.0**(-0.75 + -2. * np.random.rand())) * np.random.rand() * rs
      print('step size is: ', np.linalg.norm(stepps[i]))
  return stepps


def weighted_move(location, shell):
  me = np.copy(location)
  start = np.zeros(3)
  for j in range(len(shell)):
    them = shell[j]
    dist = np.ceil(np.linalg.norm(me - them))
    weight = 1 / dist**2
    start += weight * (them - me)
  weight_move = start / np.linalg.norm(start)
  return weight_move


def fun(r):
  return (-100 + r) * (100 + r) * (-0.002697076488101822 + 0.000019481665063774152 * r + \
         5.8309223845135165e-6 * r**2 + 3.784884075040913e-7 * r**3 - \
         1.5781149013318262e-8 * r**4 - 1.3230420835332004e-9 * r**5 + \
         1.6677465968641514e-11 * r**6 + 1.8871014935516763e-12 * r**7 - \
         9.219429140380462e-15 * r**8 - 1.500638181542457e-15 * r**9 + \
         2.7710736284434474e-18 * r**10 + 7.466561328990443e-19 * r**11 - \
         3.640235573641235e-22 * r**12 - 2.457565707237221e-22 * r**13 - \
         3.1725077201564805e-26 * r**14 + 5.491956791517286e-26 * r**15 + \
         2.078764148115423e-29 * r**16 - 8.375247516285485e-30 * r**17 - \
         3.6390466412366884e-33 * r**18 + 8.580028385401963e-34 * r**19 + \
         3.309018533342796e-37 * r**20 - 5.645760663661313e-38 * r**21 - \
         1.5828033560219243e-41 * r**22 + 2.1549930933938543e-42 * r**23 +\
         3.1525246605842057e-46 * r**24 - 3.6258343842476703e-47 * r**25)



def center_func(x):
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
  return 1.040396156714501 * (-100. + x) * (100. + x) * (-0.002240441472956154 + \
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


def shell(xx, yy, zz):
  constricted = False
  if constricted == False:
    x = xx - 100. * SCALE
    center = center_func(x)
    max_dist = rad_bent(x)

    y = yy - 31.0
    z = zz - 31.0
    rr = np.sqrt((y - center) * (y - center) + z * z)
    if rr <= max_dist:
      return True
    else:
      return False
  else:
    x = xx - 100. * SCALE
    y = yy - 25. * SCALE
    z = zz - 25. * SCALE
    rr = np.sqrt(y * y + z * z)
    max_dist = fun(x)
    if rr <= max_dist:
      return True
    else:
      return False


def downsample(shell):
  length = len(shell)
  if length < 2400:
    return shell
  else:
    permuted = np.random.permutation(range(length))[:2399]
    toreturn = np.array([shell[j] for j in permuted])
    return toreturn


def divide(positions):
  current_n = len(positions)
  to_return = np.zeros((2 * current_n, 3))
  for j in range(current_n):
    to_return[j] = positions[j]
  for j in range(current_n):
    flag = False
    me = positions[j]
    while not flag:
      me2 = me + [2 * np.random.rand() - 1., 2 * np.random.rand() - 1., 2 * np.random.rand() - 1.]
      if shell(me2[0], me2[1], me2[2]):
        flag = True
        me = np.copy(me2)
    to_return[j + current_n] = me
  return to_return


def divide2(positions,m):
  current_n = len(positions)
  to_return = np.zeros((current_n + 1, 3))
  for j in range(current_n):
    to_return[j] = positions[j]
  flag = False
  me = positions[m]
  while flag == False:
    me2 = me + [2 * np.random.rand() - 1., 2 * np.random.rand() - 1., 2 * np.random.rand() - 1.]
    if shell(me2[0], me2[1], me2[2]) == True:
      flag = True
      me = np.copy(me2)
    to_return[-1] = me
  return to_return


def con2(vec):
  x = vec[0]
  y = vec[1]
  z = vec[2]
  rx = 100. * SCALE
  ry = 25. * SCALE
  rz = 25. * SCALE
  return (x**2 / rx**2 + y**2 / ry**2 + z**2 / rz**2) - 0.995


def distance_matrix(rank):
  DS = 2
  dx = int((200 * SCALE) / DS)
  dy = int((92 * SCALE) / DS)
  dz = int((92 * SCALE) / DS)
  print(dx, dy, dz)
  distance = np.zeros((dx, dy, dz))
  for i in range(dx):
    if rank == 0:
      print(i)
    for j in range(dy):
      for k in range(dz):
        xx = DS * i - 100 * SCALE
        yy = DS * j - 25 * SCALE
        zz = DS * k - 25 * SCALE
        them = np.array([xx, yy, zz])
        surf = project2(them[0], them[1], them[2])
        distance_to_surface = np.linalg.norm(surf - them)
        distance[i][j][k] = distance_to_surface
        #print distance_to_surface
  if rank == 0:
    np.save('./DISTANCE.npy', distance)
  return distance


def centrioles(locations):
  cents = []
  for i in range(len(locations)):
    flag = False
    while flag == False:
      rx = np.random.randint(low=-1, high=2)
      ry = np.random.randint(low=-1, high=2)
      rz = np.random.randint(low=-1, high=2)
      if rx == 0 and ry == 0 and rz == 0:
        flag = False
      else:
        flag = True
        print(rx, ry, rz)
    cents.append([rx, ry, rz])
  print(cents)
  return cents


def shift_cents(cents, locations, shells):
  Num = len(cents)
  cents_new = []
  for i in range(Num):
    iamat = locations[i] + cents[i]
    tmp = np.random.rand(3)
    for j in range(len(shells[i])):
      vec = shells[i][j] - iamat
      tmp += vec / np.linalg.norm(vec)
    tmp = tmp / np.linalg.norm(tmp)
    for_append = cents[i] + 0.05 * tmp
    for_append = for_append / np.linalg.norm(for_append)
    cents_new.append(for_append)
  #print 'Just finished the update'
  #print cents_new
  return cents_new


def refine_shells(locations, shells, cents, nbhs, interior):
  cents = np.array(cents)
  num = len(locations)
  shells_new = [[] for i in range(num)]
  for i in range(num):
    # print 'I am on ',i
    me = locations[i]
    Near_1 = me + cents[i]
    Anti_1 = me - cents[i]
    for j in range(len(shells[i])):
      them = shells[i][j]
      dis_1 = np.linalg.norm(them - Near_1)
      dis_2 = np.linalg.norm(them - Anti_1)
      if dis_1 < 1.055 * dis_2:
        shells_new[i].append(shells[i][j])
  print('Length start: ', len(shells[0]))
  print('Length end: ', len(shells_new[0]))
  # exit()
  return shells_new


def setup_divide(locations, divide_at):
  rad_max = 15
  for i in range(len(locations)):
    count = 0
    for j in range(len(locations)):
      if np.linalg.norm(locations[i] - locations[j]) < rad_max:
        count += 1
    if count <= 2:
      divide_at[i] = 66 + np.random.randint(-3, 4)
    elif count <= 4:
      divide_at[i] = 67 + np.random.randint(-4, 5)
    elif count <= 6:
      divide_at[i] = 79 + np.random.randint(-4, 5)
    elif count <= 10:
      divide_at[i] = 84 + np.random.randint(-6, 7)
    elif count <= 15:
      divide_at[i] = 85 + np.random.randint(-6, 7)
    elif count <= 20:
      divide_at[i] = 89 + np.random.randint(-6, 7)
    elif count <= 30:
      divide_at[i] = 95 + np.random.randint(-6, 7)
    elif count <= 35:
      divide_at[i] = 115 + np.random.randint(-10, 11)
    elif count <= 40:
      divide_at[i] = 130 + np.random.randint(-10, 21)
    elif count <= 45:
      divide_at[i] = 140 + np.random.randint(-20, 21)
    else:
      divide_at[i] = 180 + np.random.randint(-25, 25)
  print('Divide at is: ', divide_at[:len(locations)])
  return divide_at


def divide_new(locations, divide_at, thetwo):
  rad_max = 15
  for i in range(2):
    me = locations[thetwo[i]]
    count = 0
    for j in range(len(locations)):
      if np.linalg.norm(me - locations[j]) < rad_max:
        count += 1
      else:
        count += np.exp(-(np.linalg.norm(me - locations[j])- rad_max) / (0.1 * rad_max))
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
  print('Divide at is: ', divide_at[:len(locations)])
  return divide_at


if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()

  for i in range(-100, 101):
    print(i, rad_bent(i))

  weight_small = np.ones(50)
  alphas = np.ones(100)
  length = len(alphas)
  falls = np.arange(0.1, 5, 4 / 32.)
  shell_top = np.arange(5, 25, 5)
  interior = np.zeros((202, 92, 92))
  for i in range(201):
    print(i)
    for j in range(92):
      for k in range(92):
        if shell(i, j, k):
          interior[i, j, k] = 1.0
  to_show = interior[:, :, 31]
  # plt.imshow(to_show)
  # plt.show()
  # print np.shape(to_show)
  # to_show = interior[:,31,:]
  # plt.imshow(to_show)
  # plt.show()
  print(np.shape(to_show))
  # np.save('./Bent.npy',interior)
  # interior = np.load('interior.npy')#distance_matrix(rank)
  # interior = interior[::2,::2,::2]
  print(np.shape(interior))
  print(np.shape(interior))
  # exit()
  # distance_to_surface_matrix_ones = np.ones(np.shape(distance_to_surface_matrix))
  UPPERMAX = 28.

  RANK2 = 14

  fall = falls[RANK2]
  alpha2 = alphas[RANK2]
  # old 0.89/2.
  weight_middle = 0.75 * 0.89 / 2. * 2.3 * np.array([0.00905143, 0.00905143, 0.00905143, 0.00905143, 0.00905143, \
0.00905143, 0.00905143, 0.00905143, 0.00905143, 0.00905143, \
0.00905143, 0.00905143, 0.00905143, 0.00905143, 0.00905143, \
0.00905143, 0.00905143, 0.00905143, 0.00905143, 0.00905143, \
0.00905143, 0.00905143, 0.00905143, 0.00905143, 0.00905143, \
0.00905143, 0.00905143, 0.00905143, 0.00905143, 0.00905143, \
0.00905143, 0.00905143, 0.00905143, 0.00905143, 0.00905143, \
0.00905143, 0.00905143, 0.00905143, 0.00905143, 0.00905143, \
0.00905143, 0.00905143, 0.00905143, 0.00905143, 0.00905143, \
0.00905143, 0.00905143, 0.00905143, 0.00905143, 0.00905143])

  weight_surface = 0.75*0.89/2.*2.3*np.array([0.0168312, 0.0168312, 0.0168312, 0.0168312, 0.0168312, 0.0168312, \
0.0168312, 0.0168312, 0.0168312, 0.0168312, 0.0168312, 0.0168312, \
0.0168312, 0.0168312, 0.0168312, 0.0168312, 0.0168312, 0.0168312, \
0.0168312, 0.0168312, 0.0168312, 0.0168312, 0.0168312, 0.0168312, \
0.0168312, 0.0168312, 0.0168312, 0.0168312, 0.0168312, 0.0168312, \
0.0168312, 0.0168312, 0.0168312, 0.0168312, 0.0168312, 0.0168312, \
0.0168312, 0.0168312, 0.0168312, 0.0168312, 0.0168312, 0.0168312, \
0.0168312, 0.0168312, 0.0168312, 0.0168312, 0.0168312, 0.0168312, \
0.0168312, 0.0168312])

  alpha_middle = weight_middle[RANK2] * alpha2 * weight_small[rank]
  alpha_surface = weight_surface[RANK2] * alpha2 * weight_small[rank]

  nbhds = make_nbhs()
  print(len(nbhds))
  number = 1
  tsd = np.zeros(number)
  write_Q = True
  dx = int(202 * SCALE)
  dy = int(92 * SCALE)
  dz = int(92 * SCALE)
  speed_ratio = 5.
  micro_vel = 1.0
  np.random.seed(rank)
  tsd_Shell = 3.0
  shell_bound = tsd * micro_vel
  locations, phi = setup(dx, dy, dz, number)
  cents = centrioles(locations)
  # div_method_2 = False
  tsd = np.zeros(4096)
  divide_at = np.zeros(4096)
  divide_at = setup_divide(locations, divide_at)
  for k in range(len(divide_at)):
    divide_at[k] = 10
  fates = []

  fates.append(np.array([0, 1]))

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
    # if rank == 0:
    #   for i in range(len(shells)):
    #     np.savetxt('./shell_'+str(rank)+'_'+str(time_step)+'_'+str(i)+'.csv', np.random.permutation(np.array(shells[i]).astype(int))[0:-1],fmt='%i',delimiter=',')
    locations = update_positions(locations, steps)
    phi = update_phi(dx, dy, dz, locations, cents)
    tsd_Shell += 1
    # if write_Q and rank == 0:
    #   np.savetxt('./locations_'+str(rank)+'_'+str(time_step)+'.csv',locations,delimiter=',')
    print('TSD: ', tsd[:len(locations)])
    print('Dividing at: ', divide_at[:len(locations)])
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
    # if rank == 0:
    #   np.savetxt('./fates_'+str(rank)+'_'+str(time_step)+'.csv', local_fates, delimiter=',')

    # if rank == 0:
    #   np.savetxt('./tsd_'+str(rank)+'_'+str(time_step)+'.csv', tsd, delimiter=',')



