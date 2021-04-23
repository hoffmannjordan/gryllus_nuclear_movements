import numpy as np

SCALE = 1.0

out_dir = "/home/jordanhoffmann/sim_run/"


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
