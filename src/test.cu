#include "test.h"

__global__ void updateParticlesKernel(Particle *particles, float screenWidth, float screenHeight) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    particles[i].position.x += particles[i].speed.x;
    particles[i].position.y += particles[i].speed.y;

    // Check screen boundaries
    if (particles[i].position.x > screenWidth) particles[i].position.x = 0;
    if (particles[i].position.x < 0) particles[i].position.x = screenWidth;
    if (particles[i].position.y > screenHeight) particles[i].position.y = 0;
    if (particles[i].position.y < 0) particles[i].position.y = screenHeight;
}