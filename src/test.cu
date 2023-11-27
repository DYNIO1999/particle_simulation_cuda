#include "test.h"

Particle particles[MAX_PARTICLES];

__global__ void updateParticlesKernel(Particle *particles, float deltaTime) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < MAX_PARTICLES) {
        particles[i].position.x += particles[i].speed.x * deltaTime;
        particles[i].position.y += particles[i].speed.y * deltaTime;

        particles[i].position.x = fminf(fmaxf(particles[i].position.x, 0.0f), SCREEN_WIDTH);
        particles[i].position.y = fminf(fmaxf(particles[i].position.y, 0.0f), SCREEN_HEIGHT);

        float angleChange = 1.0f;
        float angle = atan2f(particles[i].speed.y, particles[i].speed.x) + angleChange;

        particles[i].speed.x = cosf(angle);
        particles[i].speed.y = sinf(angle);

        particles[i].speed.x += 1000.0f * cosf(angle) * deltaTime;
        particles[i].speed.y += 1000.0f * sinf(angle) * deltaTime;
    }
}

void updateParitclesGPU(float deltatime) {
    Particle *d_particles;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMalloc((void**)&d_particles, MAX_PARTICLES * sizeof(Particle));
    cudaMemcpy(d_particles, particles, MAX_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (MAX_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;

    updateParticlesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_particles, deltatime);

    cudaMemcpy(particles, d_particles, MAX_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaFree(d_particles);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Elapsed Time on GPU: " << elapsedTime << " ms" << std::endl;

}

