#ifndef CUDA_PARTICLE_SIMULATION_TEST_H
#define CUDA_PARTICLE_SIMULATION_TEST_H

#include "raylib.h"
#include "cuda_runtime.h"
#include "test.h"


struct Particle{
    Vector2 position;
    Vector2 speed;
    Color color;
};

__global__ void updateParticlesKernel(Particle *particles, float screenWidth, float screenHeight);

#endif //CUDA_PARTICLE_SIMULATION_TEST_H
