
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
