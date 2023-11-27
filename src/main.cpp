#include "test.h"

#define MAX_PARTICLES 50000
#define MAX_SPEED 5.0f

constexpr float screenWidth = 1600.0f;
constexpr float screenHeight = 900.0f;


Particle particles[MAX_PARTICLES];

void updateParticlesCPU() {
    for (int i = 0; i < MAX_PARTICLES; i++) {
        particles[i].position.x += particles[i].speed.x;
        particles[i].position.y += particles[i].speed.y;

        if (particles[i].position.x > screenWidth)
            particles[i].position.x = 0;

        if (particles[i].position.x < 0)
            particles[i].position.x = screenWidth;

        if (particles[i].position.y > screenHeight)
            particles[i].position.y = 0;

        if (particles[i].position.y < 0)
            particles[i].position.y = screenHeight;
    }
}


void updateParitclesGPU() {
    Particle *d_particles;

    cudaMalloc((void**)&d_particles, MAX_PARTICLES * sizeof(Particle));
    cudaMemcpy(d_particles, particles, MAX_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (MAX_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;

    updateParticlesKernel<<blocksPerGrid, threadsPerBlock>>>(d_particles, screenWidth, screenHeight);

    cudaMemcpy(particles, d_particles, MAX_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaFree(d_particles);
}

int main() {

    InitWindow((int)screenWidth, (int)screenHeight, "Particle Simulation");

    for (int i = 0; i < MAX_PARTICLES; i++) {
        particles[i].position = (Vector2){ float(GetRandomValue(0, screenWidth)), float(GetRandomValue(0, screenHeight)) };
        particles[i].speed = (Vector2){ float(GetRandomValue(-MAX_SPEED, MAX_SPEED)), float(GetRandomValue(-MAX_SPEED, MAX_SPEED))};
        particles[i].color = (Color){
                (unsigned char)GetRandomValue(50, 255),
                (unsigned char)GetRandomValue(50, 255),
                (unsigned char)GetRandomValue(50, 255),
                255
        };

    }

    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        BeginDrawing();

        ClearBackground(RAYWHITE);

        updateParticlesCPU();

        for (int i = 0; i < MAX_PARTICLES; i++) {

            DrawCircle(particles[i].position.x, particles[i].position.y, 3, particles[i].color);
        }

        EndDrawing();
    }

    CloseWindow();

    return 0;
}