#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }

double deg2rad( double x ) { return x * pi() / 180; }

double rad2deg( double x ) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData( string s )
{
	auto found_null = s.find( "null" );
	auto b1 = s.find_first_of( "[" );
	auto b2 = s.rfind( "}]" );
	if ( found_null != string::npos )
	{
		return "";
	}
	else if ( b1 != string::npos && b2 != string::npos )
	{
		return s.substr( b1, b2 - b1 + 2 );
	}
	return "";
}

// Evaluate a polynomial.
double polyeval( Eigen::VectorXd coeffs, double x )
{
	double result = 0.0;
	for ( int i = 0; i < coeffs.size(); i++ )
	{
		result += coeffs[ i ] * pow( x, i );
	}
	return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit( Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                         int order )
{
	assert( xvals.size() == yvals.size());
	assert( order >= 1 && order <= xvals.size() - 1 );
	Eigen::MatrixXd A( xvals.size(), order + 1 );

	for ( int i = 0; i < xvals.size(); i++ )
	{
		A( i, 0 ) = 1.0;
	}

	for ( int j = 0; j < xvals.size(); j++ )
	{
		for ( int i = 0; i < order; i++ )
		{
			A( j, i + 1 ) = A( j, i ) * xvals( j );
		}
	}

	auto Q = A.householderQr();
	auto result = Q.solve( yvals );
	return result;
}

int main()
{
	uWS::Hub h;

	// MPC is initialized here!
	MPC mpc;
	using namespace std::chrono;
	time_point<system_clock> lastTime = system_clock::now();
	time_point<system_clock> currentTime = system_clock::now();

	h.onMessage( [&mpc, &lastTime, &currentTime]( uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
	                     uWS::OpCode opCode )
	             {
		             // "42" at the start of the message means there's a websocket message event.
		             // The 4 signifies a websocket message
		             // The 2 signifies a websocket event
		             string sdata = string( data ).substr( 0, length );
		             cout << sdata << endl;
		             if ( sdata.size() > 2 && sdata[ 0 ] == '4' && sdata[ 1 ] == '2' )
		             {
			             string s = hasData( sdata );
			             if ( s != "" )
			             {
				             auto j = json::parse( s );
				             string event = j[ 0 ].get<string>();
				             if ( event == "telemetry" )
				             {
					             using std::vector;
					             // j[1] is the data JSON object
					             vector<double> ptsx = j[ 1 ][ "ptsx" ];
					             vector<double> ptsy = j[ 1 ][ "ptsy" ];
					             double px = j[ 1 ][ "x" ];
					             double py = j[ 1 ][ "y" ];
					             double psi = j[ 1 ][ "psi" ];
					             double v = j[ 1 ][ "speed" ];
					             double steering_angle = j[ 1 ][ "steering_angle" ];
					             double throttle = j[ 1 ][ "throttle" ];

					             //Ignore the compilation warnings.
					             double unity_psi = j[ 1 ][ "psi_unity" ];
					             (void) unity_psi;

					             static size_t const LEN = ptsx.size();


					             std::vector<double> ptsx_car( LEN );
					             std::vector<double> ptsy_car( LEN );

					             Eigen::VectorXd xVals( LEN );
					             Eigen::VectorXd yVals( LEN );

					             for ( size_t i = 0; i < LEN; ++i )
					             {
						             double const x = ptsx[ i ] - px;
						             double const y = ptsy[ i ] - py;

						             ptsx_car[ i ] = x * cos( -psi ) - y * sin( -psi );
						             ptsy_car[ i ] = x * sin( -psi ) + y * cos( -psi );

						             xVals[ i ] = ptsx_car[ i ];
						             yVals[ i ] = ptsy_car[ i ];
					             }

					             //Using a third degree polynomial
					             auto const &coeffs = polyfit( xVals, yVals, 3 );

					             // What is the value at x = 0?
					             // It is the zeroth value since the other items will be multiplied by zero
					             double const posAtZero = coeffs[ 0 ];

					             //Cross Track error is the difference between the position of the car (0) and the trajectory
					             double const cte = posAtZero;

					             // Psi difference
					             // In order to calculate this first find the differential of the curve at the point of the car
					             // This will be equivalent to the 1st coefficients arctan
					             double const epsi = -atan( coeffs[ 1 ] );

					             currentTime = system_clock::now();
					             double latency(0.3f);
					             auto const time_difference = duration_cast<milliseconds>(currentTime - lastTime);
					             if(time_difference < milliseconds(300))
					             {
						             latency = time_difference.count() / 1000.0;
					             }
					             std::cout<<"Time diff:"<<time_difference.count()<<"\n\n";

					             lastTime = currentTime;

					             Eigen::VectorXd state( 6 );
					             state[ 0 ] = v * latency;
					             state[ 1 ] = 0;
					             state[ 2 ] = (v / 2.57) * (-1 * steering_angle) * latency;
					             state[ 3 ] = v + throttle * latency;
					             state[ 4 ] = cte + v * sin(epsi) * latency;
					             state[ 5 ] = epsi + (v/2.67) * (-1 * steering_angle) * latency;

					             /*
								 * TODO: Calculate steering angle and throttle using MPC.
								 *
								 * Both are in between [-1, 1].
								 *
								 */
					             auto vars = mpc.Solve( state, coeffs );

					             json msgJson;
					             // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
					             // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
					             msgJson[ "steering_angle" ] = -vars[ 0 ] / deg2rad( 25.0 );
					             msgJson[ "throttle" ] = vars[ 1 ];

					             //Display the MPC predicted trajectory
					             //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
					             // the points in the simulator are connected by a Green line

					             msgJson[ "mpc_x" ] = mpc.x_values;
					             msgJson[ "mpc_y" ] = mpc.y_values;

					             //Display the waypoints/reference line
					             vector<double> next_x_vals = ptsx_car;
					             vector<double> next_y_vals = ptsy_car;

					             //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
					             // the points in the simulator are connected by a Yellow line

					             msgJson[ "next_x" ] = next_x_vals;
					             msgJson[ "next_y" ] = next_y_vals;


					             auto msg = "42[\"steer\"," + msgJson.dump() + "]";
					             std::cout << msg << std::endl;
					             // Latency
					             // The purpose is to mimic real driving conditions where
					             // the car does actuate the commands instantly.
					             //
					             // Feel free to play around with this value but should be to drive
					             // around the track with 100ms latency.
					             //
					             // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
					             // SUBMITTING.
					             this_thread::sleep_for( chrono::milliseconds( 100 ));
					             ws.send( msg.data(), msg.length(), uWS::OpCode::TEXT );
				             }
			             }
			             else
			             {
				             // Manual driving
				             std::string msg = "42[\"manual\",{}]";
				             ws.send( msg.data(), msg.length(), uWS::OpCode::TEXT );
			             }
		             }
	             } );

	// We don't need this since we're not using HTTP but if it's removed the
	// program
	// doesn't compile :-(
	h.onHttpRequest( []( uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
	                     size_t, size_t )
	                 {
		                 const std::string s = "<h1>Hello world!</h1>";
		                 if ( req.getUrl().valueLength == 1 )
		                 {
			                 res->end( s.data(), s.length());
		                 }
		                 else
		                 {
			                 // i guess this should be done more gracefully?
			                 res->end( nullptr, 0 );
		                 }
	                 } );

	h.onConnection( [&h]( uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req )
	                {
		                std::cout << "Connected!!!" << std::endl;
	                } );

	h.onDisconnection( [&h]( uWS::WebSocket<uWS::SERVER> ws, int code,
	                         char *message, size_t length )
	                   {
		                   ws.close();
		                   std::cout << "Disconnected" << std::endl;
	                   } );

	int port = 4567;
	if ( h.listen( port ))
	{
		std::cout << "Listening to port " << port << std::endl;
	}
	else
	{
		std::cerr << "Failed to listen to port" << std::endl;
		return -1;
	}
	h.run();
}
