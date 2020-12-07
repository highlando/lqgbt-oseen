
## 2020-04-18 22:17:54+02:00

 * RE=35 and no FB gives nice wiggles

## 2020-04-19 07:59:19+02:00

 * RE=35 and reduced output feedback works
 * RE=40 too
 * RE=45 failed 
   * with perturbation `1e-5` of the initial value and perturbation of the input
   * gonna try only perturbation to the input with amplitude `1e-3`
   * gonna try only perturbation to the input with amplitude `1e-5`
   * works

## 2020-04-19 20:43:59+02:00

 * RE=50 too
 * going from 50 to 60 --- ADI did not converge

## 2020-04-21 10:16:30+02:00

 * plan: make it run on mechthild
 * TODO:
   * branch to include modules
   * what about the output
   * save it as a png??

## 2020-11-14 11:09:59+01:00

 * TODO: check -- bmat[:, 0] === 0 
 * branch `debugging`
   * a.o.: matlab (by Steffen) and python scripts to test the pymess matrices
## 2020-12-03 21:59:37+01:00

 * branch `debugging` checked everything seems ok with the mats
 * overall (reduced) closed loop is stable
 * still `hinf` explodes in no time
 * gonna check `pymess-LQG` next

## 2020-12-04 20:45:42+01:00

 * gonna check `pymess-LQG` next
 * used the ZDG formula -- same problem 
 * problem seems to be an error in the velocity in the start
   * in `heunpred` already an error of `1e-7` (should not happen)

## 2020-12-05 11:51:09+01:00

 * the initial velocity "error" is OK -- (if no input perturbation then it is of
   size `1e-13`)
 * problem is the stiff and oscillatory closed loop system 
   * see `cylinderwake_re20_rom_control.py` for example sims
 * solutions -- a controller with stricter spectral condition
 * implicit scheme for the controller

## 2020-12-07 11:32:41+01:00

 * branch `dbrotcyl` -- my LQG works really well (RE=50)

