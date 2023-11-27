#ifndef CUDA_PARTICLE_SIMULATION_TEST_H
#define CUDA_PARTICLE_SIMULATION_TEST_H

#include "raylib.h"
#include <iostream>

#define MAX_PARTICLES 50000
#define MAX_SPEED 5.0f

#define SCREEN_WIDTH 1600.0f
#define SCREEN_HEIGHT 900.0f

struct Particle{
    Vector2 position;
    Vector2 speed;
    Color color;
};

extern Particle particles[MAX_PARTICLES];

void updateParitclesGPU(float deltatime);

#endif
