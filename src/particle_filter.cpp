/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    //set initialization to false:
    is_initialized = false;

    //set num_particles to 100:
    num_particles = 100;

    // Create a normal (Gaussian) distribution for x, y, theta
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // initialize particles
    for(size_t i=0; i< num_particles; i++)
    {
        Particle p;
        p.id = int(i);
        p.weight = 1.0;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        particles.push_back(p);
        weights.push_back(p.weight);
    }

    //set initialization to true:
    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    // for each particle
    for(auto& p : particles)
    {
        // yaw_rate is not constant
        if(fabs(yaw_rate) > 1e-5)
        {
            p.x += velocity/yaw_rate*(sin(p.theta+yaw_rate*delta_t)-sin(p.theta));
            p.y += velocity/yaw_rate*(cos(p.theta)-cos(p.theta+yaw_rate*delta_t));
            p.theta += yaw_rate*delta_t;

            // yaw_rate is constant
        }else
        {
            p.x += velocity*delta_t*cos(p.theta);
            p.y += velocity*delta_t*sin(p.theta);
        }


        // Create a normal (Gaussian) distribution for x, y, theta
        normal_distribution<double> dist_x(p.x, std_pos[0]);
        normal_distribution<double> dist_y(p.y, std_pos[1]);
        normal_distribution<double> dist_theta(p.theta, std_pos[2]);

        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);

    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    double distance;
    double min_dist;
    LandmarkObs min_obs;
    for(auto& obs: observations)
    {
        min_dist = numeric_limits<double>::max();
        for(auto& pred: predicted)
        {
            distance = dist(obs.x, obs.y, pred.x, pred.y);
            if(distance < min_dist)
            {
                min_obs = pred;
                min_dist = distance;
            }
        }
        obs = min_obs;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
        std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    LandmarkObs tmp_landmark;
    for(auto& p: particles)
    {
        std::vector<LandmarkObs> map_observations;
        // transform all observations from vehicle coordinate system to MAP coordinate system
        for(auto& obj: observations)
        {
            LandmarkObs map_observation;
            map_observation.x = p.x + obj.x * cos(p.theta) - obj.y * sin(p.theta);
            map_observation.y = p.y + obj.x * sin(p.theta) + obj.y * cos(p.theta);
            map_observation.id = obj.id;

            map_observations.push_back(map_observation);
        }

        // find all landmarks within sensor range
        std::vector<LandmarkObs> in_range_landmarks;

        for (auto landmark: map_landmarks.landmark_list){

            double distance = dist(p.x,p.y,landmark.x_f,landmark.y_f);
            if (distance < sensor_range) {
                tmp_landmark.id = landmark.id_i;
                tmp_landmark.x = landmark.x_f;
                tmp_landmark.y = landmark.y_f;
                in_range_landmarks.push_back(tmp_landmark);
            }
        }

        // call dataAssociation to find predicted landmark that is closest to observed
        std::vector<LandmarkObs> original_observations = map_observations;
        dataAssociation(in_range_landmarks, map_observations);

        // calculate probability
        double probability = 1;
        for (int j=0; j < original_observations.size(); ++j){
            double dx = original_observations.at(j).x - map_observations.at(j).x;
            double dy = original_observations.at(j).y - map_observations.at(j).y;
            probability *= 1.0/(2*M_PI*std_landmark[0]*std_landmark[1]) * exp(-dx*dx / (2*std_landmark[0]*std_landmark[0]))* exp(-dy*dy / (2*std_landmark[1]*std_landmark[1]));
        }

        // update weights
        p.weight = probability;

    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    discrete_distribution<int> discrete_weight_dist(weights.begin(), weights.end());

    std::vector<Particle> weighted_sample;

    Particle p;
    for(size_t i = 0; i < num_particles; i++)
    {
        int j = discrete_weight_dist(gen);
        weighted_sample.push_back(particles[j]);
    }

    particles = weighted_sample;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
