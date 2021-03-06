# CarND-Controls-MPC
Self-Driving Car Engineer Nanodegree Program

---

## Documentation
This section answers the questions posted in the rubric. They are as follows:

***Student describes their model in detail. This includes the state, actuators and update equations.***

**States:**
1. X Position of the car in it's own space.
2. Y Position of the car in it's own space.
3. Psi angle (orientation angle) of the car in it's own space.
4. Velocity of the car in the world space.
5. Cross Track Error of the car in it's own space.
6. Orientation Error of the car with respect to the trajectory it is supposed to follow.

**Actuators:**
1. Throttle of the car
2. Steering of the car

**Update Equations:**

I followed the traditional model as discussed in the class and the quiz.
```
AD<double> delta0 = vars[ delta_start + t - 1 ];
AD<double> a0 = vars[ a_start + t - 1 ];

AD<double> f0 = coeffs[ 0 ] + coeffs[ 1 ] * x0 + coeffs[ 2 ] * x0 * x0 + coeffs[ 3 ] * x0 * x0 * x0;
AD<double> psides0 = CppAD::atan( coeffs[ 1 ] + (coeffs[ 2 ] * x0 * 2) + (3 * coeffs[ 3 ] * (x0 * x0)));

fg[ 1 + x_start + t ] = x1 - (x0 + v0 * CppAD::cos( psi0 ) * dt);
fg[ 1 + y_start + t ] = y1 - (y0 + v0 * CppAD::sin( psi0 ) * dt);
fg[ 1 + psi_start + t ] = psi1 - (psi0 + v0 * delta0 / Lf * dt);
fg[ 1 + v_start + t ] = v1 - (v0 + a0 * dt);
fg[ 1 + cte_start + t ] = cte1 - ((f0 - y0) + (v0 * CppAD::sin( epsi0 ) * dt));
fg[ 1 + epsi_start + t ] = epsi1 - ((psi0 - psides0) + v0 * delta0 / Lf * dt);
```

**Constraints:**

These are also the same as those discussed in class and the quiz with some amount of tweaking. The constants shown were obtained through some common sense and hit and miss attempts.
```
for ( size_t t = 0; t < N; t++ )
{
  static double const CTE_FACTOR = 0.5;
  static double const EPSI_FACTOR = 3.0;
  static double const VELOCITY_FACTOR = 5.0;
  static double const STRAIGHT_BASE_VELOCITY = 30; //mph

  fg[ 0 ] += CTE_FACTOR * CppAD::pow( vars[ cte_start + t ], 2 );
  fg[ 0 ] += EPSI_FACTOR * CppAD::pow( vars[ epsi_start + t ], 2 );
  fg[ 0 ] += VELOCITY_FACTOR * CppAD::pow( vars[ v_start + t ] - STRAIGHT_BASE_VELOCITY, 2 );
}
```
```
for ( size_t t = 0; t < N - 1; t++ )
{
  static double const DELTA_FACTOR = 10.0;
  fg[ 0 ] += DELTA_FACTOR * CppAD::pow( vars[ delta_start + t ], 2 );
  fg[ 0 ] += CppAD::pow( vars[ a_start + t ], 2 );
}
```
```
for ( size_t t = 0; t < N - 2; t++ )
{
  static double const DELTA_FACTOR = 1000;
  static double const ACCEL_FACTOR = 0.01;
  fg[ 0 ] += DELTA_FACTOR * CppAD::pow( vars[ delta_start + t + 1 ] - vars[ delta_start + t ], 2 );
  fg[ 0 ] += ACCEL_FACTOR * CppAD::pow( vars[ a_start + t + 1 ] - vars[ a_start + t ], 2 );
}
```
<br>

***Student discusses the reasoning behind the chosen N (timestep length) and dt (elapsed duration between timesteps) values. Additionally the student details the previous values tried.***

* N: 20
* dt = 0.05

I used these values of N and dt because they seemed to work for me consistently. Using small N's ( less than 10 )or dt's larger than 0.1 resulted in a loss of stability of the controller.

<br>

***A polynomial is fitted to waypoints. If the student preprocesses waypoints, the vehicle state, and/or actuators prior to the MPC procedure it is described.***

I transform the waypoints into the space of the car in order to be able to draw them. I apply a cubic polynomial on the waypoints before sending them over to my controller for fitting.

<br>

***The student implements Model Predictive Control that handles a 100 millisecond latency. Student provides details on how they deal with latency.***

I do handle latency in my system. I predict the state staring at the next time frame assumed to be at 300ms after the current frame or the amount of time it takes between two frames provided that they are less than 300 ms.
The updated equations are as follows:

```
x = v * latency;
y = 0;
psi = (v / 2.57) * (-1 * steering_angle) * latency;
v = v + throttle * latency;
cte = cte + v * sin(epsi) * latency;
epsi = epsi + (v/2.67) * (-1 * steering_angle) * latency;
```


## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Fortran Compiler
  * Mac: `brew install gcc` (might not be required)
  * Linux: `sudo apt-get install gfortran`. Additionall you have also have to install gcc and g++, `sudo apt-get install gcc g++`. Look in [this Dockerfile](https://github.com/udacity/CarND-MPC-Quizzes/blob/master/Dockerfile) for more info.
* [Ipopt](https://projects.coin-or.org/Ipopt)
  * Mac: `brew install ipopt`
  * Linux
    * You will need a version of Ipopt 3.12.1 or higher. The version available through `apt-get` is 3.11.x. If you can get that version to work great but if not there's a script `install_ipopt.sh` that will install Ipopt. You just need to download the source from the Ipopt [releases page](https://www.coin-or.org/download/source/Ipopt/) or the [Github releases](https://github.com/coin-or/Ipopt/releases) page.
    * Then call `install_ipopt.sh` with the source directory as the first argument, ex: `bash install_ipopt.sh Ipopt-3.12.1`. 
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [CppAD](https://www.coin-or.org/CppAD/)
  * Mac: `brew install cppad`
  * Linux `sudo apt-get install cppad` or equivalent.
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/self-driving-car-sim/releases).
* Not a dependency but read the [DATA.md](./DATA.md) for a description of the data sent back from the simulator.


## Basic Build Instructions


1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./mpc`.

## Tips

1. It's recommended to test the MPC on basic examples to see if your implementation behaves as desired. One possible example
is the vehicle starting offset of a straight line (reference). If the MPC implementation is correct, after some number of timesteps
(not too many) it should find and track the reference line.
2. The `lake_track_waypoints.csv` file has the waypoints of the lake track. You could use this to fit polynomials and points and see of how well your model tracks curve. NOTE: This file might be not completely in sync with the simulator so your solution should NOT depend on it.
3. For visualization this C++ [matplotlib wrapper](https://github.com/lava/matplotlib-cpp) could be helpful.

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!

More information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/f1820894-8322-4bb3-81aa-b26b3c6dcbaf/lessons/b1ff3be0-c904-438e-aad3-2b5379f0e0c3/concepts/1a2255a0-e23c-44cf-8d41-39b8a3c8264a)
for instructions and the project rubric.

## Hints!

* You don't have to follow this directory structure, but if you do, your work
  will span all of the .cpp files here. Keep an eye out for TODOs.

## Call for IDE Profiles Pull Requests

Help your fellow students!

We decided to create Makefiles with cmake to keep this project as platform
agnostic as possible. Similarly, we omitted IDE profiles in order to we ensure
that students don't feel pressured to use one IDE or another.

However! I'd love to help people get up and running with their IDEs of choice.
If you've created a profile for an IDE that you think other students would
appreciate, we'd love to have you add the requisite profile files and
instructions to ide_profiles/. For example if you wanted to add a VS Code
profile, you'd add:

* /ide_profiles/vscode/.vscode
* /ide_profiles/vscode/README.md

The README should explain what the profile does, how to take advantage of it,
and how to install it.

Frankly, I've never been involved in a project with multiple IDE profiles
before. I believe the best way to handle this would be to keep them out of the
repo root to avoid clutter. My expectation is that most profiles will include
instructions to copy files to a new location to get picked up by the IDE, but
that's just a guess.

One last note here: regardless of the IDE used, every submitted project must
still be compilable with cmake and make./